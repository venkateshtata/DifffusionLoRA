from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from pathlib import Path
from PIL import Image
import os

save_dir = "./outputs"
os.makedirs(save_dir, exist_ok=True)



logging_dir = Path("/notebooks/lora/diff_lora/text_to_image/realms", "logs")
accelerator_project_config = ProjectConfiguration(project_dir="/notebooks/lora/diff_lora/text_to_image/realms", logging_dir=logging_dir)

accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",
        project_config=accelerator_project_config,
    )

# seed = 100
# torch.manual_seed(seed)

# # model_path = "./onepiece-latest/"
# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker = None)
# # pipe.unet.load_attn_procs(model_path)
# pipe.to("cuda")

# prompt = "miner"
# image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
# image.save("piece.png")

pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision=None, variant=None, torch_dtype=torch.float32)

pipeline = pipeline.to(accelerator.device)

# load attention processors
pipeline.load_lora_weights("/notebooks/lora/diff_lora/text_to_image/realms")

# run inference
generator = torch.Generator(device='cuda')

# if run_config['seed'] is not None:
#     generator = generator.manual_seed(run_config['seed'])

images = []

autocast_ctx = torch.autocast(accelerator.device.type)

with autocast_ctx:
    for _ in range(4):
        images.append(pipeline("a girl in pink dress", num_inference_steps=30, generator=generator).images[0])
        

for idx, img in enumerate(images):
    # Assuming 'img' is a PIL Image object, construct a filename for each image
    filename = f"image_{idx + 1}.png"
    filepath = os.path.join(save_dir, filename)
    # Save the image
    img.save(filepath)

print(f"Saved {len(images)} images to '{save_dir}'.")