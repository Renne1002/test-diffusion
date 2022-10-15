import torch
from torch import autocast
from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline
from japanese_stable_diffusion import JapaneseStableDiffusionPipeline
import matplotlib.pyplot as plt
import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

access_token = os.environ.get("ACCESS_TOKEN") #@param {type:"string"}

# model_id = "rinna/japanese-stable-diffusion"
model_id = "CompVis/stable-diffusion-v1-4"
devise = "cuda" if torch.cuda.is_available() else "cpu"

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, use_auth_token=access_token)
pipe = pipe.to(devise)

prompt = "assaultlily" # @param {type:"string"}

num = 1

for i in range(num):
  with autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5, width=768, height=512)["sample"][0]
  image.save(f"outputs/{i:04}.png")

for i in range(num):
  plt.imshow(plt.imread(f"outputs/{i:04}.png"))
  plt.axis("off")
  plt.show()
