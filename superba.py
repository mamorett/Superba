import argparse
import math
import os
import sys
from PIL import Image, ImageDraw, ImageFilter
import torch
from diffusers import FluxImg2ImgPipeline, FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from tqdm import tqdm
from safetensors.torch import load_file
from enum import Enum
from huggingface_hub import hf_hub_download
from optimum.quanto import freeze, qfloat8, quantize
import datetime

# Memory optimization
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

dtype = torch.bfloat16
bfl_repo = "black-forest-labs/FLUX.1-dev"
revision = "main"

class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2

class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3

class FluxUpscaler:
    def __init__(self, acceleration=None, loras=None, safetensor_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = dtype
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
        self.vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision).to(self.device)
        
        # Load transformer with optional safetensor
        self.transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, revision=revision)
        
        # Load safetensor if provided
        if safetensor_path:
            print(f"Loading transformer from safetensor: {safetensor_path}")
            state_dict = load_file(safetensor_path, device=self.device)
            self.transformer.load_state_dict(state_dict, strict=False)
            self.transformer.eval()
        
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision)
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", revision=revision)
        
        self.pipe = FluxImg2ImgPipeline(
            scheduler=self.scheduler,
            vae=self.vae,
            transformer=self.transformer,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            text_encoder_2=self.text_encoder_2,
            tokenizer_2=self.tokenizer_2,
        )
        
        # Apply LoRAs if provided
        if loras:
            for lorafile in loras:
                lora_name = os.path.basename(lorafile)
                print(f"Applying LoRA: {lora_name}")
                self.pipe.load_lora_weights(lorafile, weight_name=lora_name)
                self.pipe.fuse_lora(lora_scale=1.0)

        # Apply acceleration first
        self.default_steps = self._apply_acceleration(acceleration)
        
        # Apply quantization only if no safetensor was provided
        if not safetensor_path:
            print(f"{datetime.datetime.now()} Quantizing transformer to qfloat8")
            quantize(self.transformer, weights=qfloat8)
            freeze(self.transformer)

        self.pipe.enable_model_cpu_offload()
        print(f"FluxUpscaler initialized with acceleration={acceleration}, default_steps={self.default_steps}")


    def _apply_acceleration(self, acceleration):
        if acceleration == "hyper":
            repo_name = "ByteDance/Hyper-SD"
            ckpt_name = "Hyper-FLUX.1-dev-8steps-lora.safetensors"
            try:
                self.pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
                self.pipe.fuse_lora(lora_scale=0.125)
                print(f"Successfully loaded Hyper-SD adapter: {ckpt_name}")
                return 10
            except Exception as e:
                print(f"Failed to load Hyper-SD: {str(e)}")
                return 25
        elif acceleration == "alimama":
            adapter_id = "alimama-creative/FLUX.1-Turbo-Alpha"
            try:
                self.pipe.load_lora_weights(adapter_id)
                self.pipe.fuse_lora(lora_scale=1.0)
                print(f"Successfully loaded Alimama adapter: {adapter_id}")
                return 10
            except Exception as e:
                print(f"Failed to load Alimama: {e}")
                return 25
        return 25

class SimpleUpscaler:
    def __init__(self, name="Lanczos"):
        self.name = name

    def upscale(self, image, scale_factor):
        if self.name == "None":
            return image
        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)
        return image.resize((new_width, new_height), resample=Image.LANCZOS)

class USDURedraw:
    def __init__(self):
        self.tile_width = 1024
        self.tile_height = 1024
        self.padding = 32
        self.mode = USDUMode.LINEAR
        self.enabled = True
        self.tile_overlap = 64
        self.current_pass = 0

    def _is_valid_size(self, size):
        return all(dim % 8 == 0 for dim in size)

    def start(self, pipe, image, rows, cols, prompt, strength, steps):
        if not self.enabled:
            return image
            
        print(f"Starting {self.mode.name} redraw with {rows}x{cols} tiles, steps={steps}")
        self.current_pass += 1
        
        if self.mode == USDUMode.LINEAR:
            return self._linear_pass(pipe, image, rows, cols, prompt, strength, steps)
        elif self.mode == USDUMode.CHESS:
            return self._chess_pass(pipe, image, rows, cols, prompt, strength, steps)
        return image

    def _linear_pass(self, pipe, image, rows, cols, prompt, strength, steps):
        total_tiles = rows * cols
        with tqdm(total=total_tiles, desc=f"Linear Pass {self.current_pass}") as pbar:
            for yi in range(rows):
                for xi in range(cols):
                    x1, y1, x2, y2 = self.calc_rectangle(xi, yi, image.width, image.height)
                    image = self._process_tile(pipe, image, x1, y1, x2, y2, str(prompt), strength, steps)
                    pbar.update(1)
        return image

    def _chess_pass(self, pipe, image, rows, cols, prompt, strength, steps):
        total_tiles = rows * cols
        with tqdm(total=total_tiles, desc=f"Chess Pass {self.current_pass}") as pbar:
            for phase in [True, False]:
                for yi in range(rows):
                    for xi in range(cols):
                        if (xi + yi) % 2 == int(phase):
                            x1, y1, x2, y2 = self.calc_rectangle(xi, yi, image.width, image.height)
                            image = self._process_tile(pipe, image, x1, y1, x2, y2, str(prompt), strength, steps)
                            pbar.update(1)
        return image

    def calc_rectangle(self, xi, yi, img_width, img_height):
        x1 = max(0, xi * (self.tile_width - self.tile_overlap))
        y1 = max(0, yi * (self.tile_height - self.tile_overlap))
        x2 = min(x1 + self.tile_width + self.tile_overlap, img_width)
        y2 = min(y1 + self.tile_height + self.tile_overlap, img_height)
        adj_width = ((x2 - x1) // 8) * 8
        adj_height = ((y2 - y1) // 8) * 8
        return (x1, y1, x1 + adj_width, y1 + adj_height)

    def _process_tile(self, pipe, base_image, x1, y1, x2, y2, prompt, strength, steps):
        expanded_x1 = max(0, x1 - self.tile_overlap)
        expanded_y1 = max(0, y1 - self.tile_overlap)
        expanded_x2 = min(base_image.width, x2 + self.tile_overlap)
        expanded_y2 = min(base_image.height, y2 + self.tile_overlap)
        tile = base_image.crop((expanded_x1, expanded_y1, expanded_x2, expanded_y2))
        if not self._is_valid_size(tile.size):
            new_size = ((tile.width // 8) * 8, (tile.height // 8) * 8)
            tile = tile.resize(new_size, Image.LANCZOS)
        mask = Image.new("L", tile.size, 0)
        draw = ImageDraw.Draw(mask)
        feather = self.tile_overlap
        draw.rectangle((feather, feather, mask.width - feather, mask.height - feather), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(feather // 2))
        print(f"Processing tile with steps={steps}")
        result = pipe(prompt=prompt, image=tile, strength=float(strength), num_inference_steps=int(steps), guidance_scale=7.5).images[0]
        if result.size != tile.size:
            result = result.resize(tile.size, Image.LANCZOS)
        base_image = base_image.copy()
        base_image.paste(result, (expanded_x1, expanded_y1), mask=mask)
        return base_image

class USDUSeamsFix:
    def __init__(self):
        self.tile_width = 1024
        self.tile_height = 1024
        self.padding = 16
        self.denoise = 0.35
        self.mask_blur = 4
        self.width = 64
        self.mode = USDUSFMode.NONE
        self.enabled = False

    def process_tile(self, pipe, image, x1, y1, x2, y2, prompt, steps):
        tile = image.crop((x1, y1, x2, y2))
        result = pipe(prompt=prompt, image=tile, strength=self.denoise, num_inference_steps=steps, guidance_scale=7.5).images[0]
        output = image.copy()
        output.paste(result, (x1, y1))
        return output

    def half_tile_process(self, pipe, image, rows, cols, prompt, steps):
        total_seams = (rows - 1) * cols + rows * (cols - 1)
        with tqdm(total=total_seams, desc="Seams Fix") as pbar:
            for yi in range(rows - 1):
                for xi in range(cols):
                    x1 = xi * self.tile_width
                    y1 = yi * self.tile_height + self.tile_height // 2 - self.width // 2
                    x2 = min(x1 + self.tile_width, image.width)
                    y2 = min(y1 + self.width, image.height)
                    image = self.process_tile(pipe, image, x1, y1, x2, y2, prompt, steps)
                    pbar.update(1)
            for yi in range(rows):
                for xi in range(cols - 1):
                    x1 = xi * self.tile_width + self.tile_width // 2 - self.width // 2
                    y1 = yi * self.tile_height
                    x2 = min(x1 + self.width, image.width)
                    y2 = min(y1 + self.tile_height, image.height)
                    image = self.process_tile(pipe, image, x1, y1, x2, y2, prompt, steps)
                    pbar.update(1)
        return image

    def start(self, pipe, image, rows, cols, prompt, steps):
        if self.mode == USDUSFMode.HALF_TILE:
            return self.half_tile_process(pipe, image, rows, cols, prompt, steps)
        return image

class USDUpscaler:
    def __init__(self, image, upscaler, tile_width, tile_height, save_redraw, save_seams_fix):
        self.image = image
        self.upscaler = upscaler
        self.redraw = USDURedraw()
        self.redraw.tile_width = tile_width
        self.redraw.tile_height = tile_height
        self.redraw.save = save_redraw
        self.seams_fix = USDUSeamsFix()
        self.seams_fix.tile_width = tile_width
        self.seams_fix.tile_height = tile_height
        self.seams_fix.save = save_seams_fix

    def upscale(self, upscale_ratio):
        self.scale_factor = upscale_ratio
        target_width = int(self.image.width * upscale_ratio)
        target_height = int(self.image.height * upscale_ratio)
        self.image = self.upscaler.upscale(self.image, self.scale_factor)
        self.image = self.image.resize((target_width, target_height), resample=Image.LANCZOS)
        self.rows = math.ceil(target_height / self.redraw.tile_height)
        self.cols = math.ceil(target_width / self.redraw.tile_width)

    def setup_redraw(self, redraw_mode, padding):
        self.redraw.mode = USDUMode(redraw_mode)
        self.redraw.enabled = self.redraw.mode != USDUMode.NONE
        self.redraw.padding = padding

    def setup_seams_fix(self, padding, denoise, mask_blur, width, mode):
        self.seams_fix.padding = padding
        self.seams_fix.denoise = denoise
        self.seams_fix.mask_blur = mask_blur
        self.seams_fix.width = width
        self.seams_fix.mode = USDUSFMode(mode)
        self.seams_fix.enabled = self.seams_fix.mode != USDUSFMode.NONE

    def save_image(self, output_dir, filename, suffix=""):
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}{suffix}.png"
        path = os.path.join(output_dir, output_filename)
        self.image.save(path)
        print(f"Saved image to {path}")

    def process(self, pipe, upscale_ratio, prompt, strength, steps, output_dir, filename):
        self.upscale(upscale_ratio)
        if self.redraw.enabled:
            print(f"Starting redraw with {self.redraw.mode.name} mode")
            self.image = self.redraw.start(pipe, self.image, self.rows, self.cols, prompt, strength, steps)
            if self.redraw.save:
                self.save_image(output_dir, filename, "_upscaled")
        if self.seams_fix.enabled:
            print(f"Starting seams fix with {self.seams_fix.mode.name} mode")
            self.image = self.seams_fix.start(pipe, self.image, self.rows, self.cols, prompt, steps)
            if self.seams_fix.save:
                self.save_image(output_dir, filename, "_seams_fixed")
        self.save_image(output_dir, filename)

def process_directory(input_dir, output_dir, upscale_ratio, tile_width, tile_height, padding, redraw_mode, save_redraw,
                     seams_fix_mode, seams_fix_padding, seams_fix_denoise, save_seams_fix, prompt, safetensor, lora,
                     denoise, steps, acceleration):
    print("Loading FLUX pipeline...")
    flux = FluxUpscaler(acceleration=acceleration, loras=[lora] if lora else None, safetensor_path=safetensor)
    pipe = flux.pipe
    
    # Determine steps: user-provided overrides, otherwise use acceleration default adjusted by denoise
    final_steps = steps if steps is not None else int(flux.default_steps / denoise)
    print(f"Final steps set to: {final_steps} (user-provided={steps}, acceleration default={flux.default_steps}, denoise={denoise})")

    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    with tqdm(total=len(image_files), desc="Processing Directory") as pbar:
        for filename in image_files:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Check if output file already exists
            if os.path.exists(output_path):
                print(f"Skipping {filename} - output file already exists")
                pbar.update(1)
                continue
                
            try:
                init_img = Image.open(input_path).convert("RGB")
                upscaler = SimpleUpscaler()
                flux_upscaler = USDUpscaler(init_img, upscaler, tile_width, tile_height, save_redraw, save_seams_fix)
                flux_upscaler.setup_redraw(redraw_mode, padding)
                flux_upscaler.setup_seams_fix(seams_fix_padding, seams_fix_denoise, 4, 64, seams_fix_mode)
                flux_upscaler.process(pipe, upscale_ratio, prompt, denoise, final_steps, output_dir, filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
            pbar.update(1)


def main():
    parser = argparse.ArgumentParser(description="Ultimate FLUX Upscale script with directory processing")
    parser.add_argument("-i", "--input", required=True, help="Path to input image or directory")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("-r", "--upscale_ratio", type=float, default=2.0, help="Upscale ratio (e.g., 2.0, 3.0)")
    parser.add_argument("-tw", "--tile_width", type=int, default=1024, help="Tile width")
    parser.add_argument("-th", "--tile_height", type=int, default=1024, help="Tile height")
    parser.add_argument("-p", "--padding", type=int, default=32, help="Padding for redraw")
    parser.add_argument("-rm", "--redraw_mode", type=int, default=0, choices=[0, 1, 2], help="0=Linear, 1=Chess, 2=None")
    parser.add_argument("-sr", "--save_redraw", action="store_true", help="Save upscaled image")
    parser.add_argument("-sfm", "--seams_fix_mode", type=int, default=0, choices=[0, 1, 2, 3], help="0=None, 1=Band Pass, 2=Half Tile, 3=Half Tile + Intersections")
    parser.add_argument("-sfp", "--seams_fix_padding", type=int, default=16, help="Padding for seams fix")
    parser.add_argument("-sfd", "--seams_fix_denoise", type=float, default=0.35, help="Denoise strength for seams fix")
    parser.add_argument("-ssf", "--save_seams_fix", action="store_true", help="Save seams-fixed image")
    parser.add_argument("-pr", "--prompt", default="upscaled image", help="Prompt for FLUX")
    parser.add_argument("-st", "--safetensor", type=str, help="Path to safetensor file (disables quantization)")
    parser.add_argument("-l", "--lora", type=str, help="Path to LoRA file")
    parser.add_argument("-d", "--denoise", type=float, default=0.3, help="Denoise strength")
    parser.add_argument("-s", "--steps", type=int, help="Number of steps (overrides acceleration default)")
    parser.add_argument("-a", "--acceleration", type=str, default="none", choices=["none", "hyper", "alimama"], help="Acceleration method")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        process_directory(
            args.input, args.output, args.upscale_ratio, args.tile_width, args.tile_height, args.padding,
            args.redraw_mode, args.save_redraw, args.seams_fix_mode, args.seams_fix_padding, args.seams_fix_denoise,
            args.save_seams_fix, args.prompt, args.safetensor, args.lora, args.denoise, args.steps, args.acceleration
        )
    else:
        try:
            init_img = Image.open(args.input).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {e}")
            sys.exit(1)
        
        print("Loading FLUX pipeline...")
        flux = FluxUpscaler(acceleration=args.acceleration, loras=[args.lora] if args.lora else None, safetensor_path=args.safetensor)
        pipe = flux.pipe
        
        # Determine steps: user-provided overrides, otherwise use acceleration default adjusted by denoise
        final_steps = args.steps if args.steps is not None else int(flux.default_steps / args.denoise)
        print(f"Final steps set to: {final_steps} (user-provided={args.steps}, acceleration default={flux.default_steps}, denoise={args.denoise})")

        upscaler = SimpleUpscaler()
        flux_upscaler = USDUpscaler(init_img, upscaler, args.tile_width, args.tile_height, args.save_redraw, args.save_seams_fix)
        flux_upscaler.setup_redraw(args.redraw_mode, args.padding)
        flux_upscaler.setup_seams_fix(args.seams_fix_padding, args.seams_fix_denoise, 4, 64, args.seams_fix_mode)
        flux_upscaler.process(pipe, args.upscale_ratio, args.prompt, args.denoise, final_steps, args.output, os.path.basename(args.input))

    print("Processing complete.")

if __name__ == "__main__":
    main()