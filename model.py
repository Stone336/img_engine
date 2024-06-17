from diffusers import AutoPipelineForText2Image
import torch

def load_model(model_id="runwayml/stable-diffusion-v1-5"):
    pipeline = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    return pipeline

def generate_image(pipeline, prompt, height=512, width=512):
    image = pipeline(prompt, height=height, width=width).images[0]
    return image
