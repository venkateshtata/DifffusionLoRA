from diffusers import StableDiffusionPipeline
import torch

# seed = 100
# torch.manual_seed(seed)

model_path = "./onepiece-latest/"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker = None)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

prompt = "a women in pink dress"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("piece.png")