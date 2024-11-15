!pip install diffusers transformers gradio accelerate
!pip show torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import torch
import gradio as gr

model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"
model_id2 = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionPipeline.from_pretrained(model_id1, torch_dtype=torch.float16, use_safetensors=True)
pipe = pipe.to("cuda")

prompt = """ rose
"""

image = pipe(prompt).images[0]

print("[PROMPT]: ",prompt)
plt.imshow(image);
plt.axis('off');

prompt2 = """dreamlike, Goddess Durga coming down from the heaven with a weapon in one hand and other hand in the pose of blessing. Anger and divine energy
reflecting from her eyes.
She is in the form of a soldier and savior coming to protect the world from misery. She is accompanied by her tiger. Make sure to keep it cinematic and color to be golden iris
"""

image = pipe(prompt2).images[0]

print('[PROMPT]: ',prompt2)
plt.imshow(image)
plt.axis('off')

def generate_image(pipe, prompt, params):
  img = pipe(prompt, **params).images

  num_images = len(img)
  if num_images>1:
    fig, ax = plt.subplots(nrows=1, ncols=num_images)
    for i in range(num_images):
      ax[i].imshow(img[i]);
      ax[i].axis('off');

  else:
    fig = plt.figure()
    plt.imshow(img[0]);
    plt.axis('off');
  plt.tight_layout()

  prompt = "dreamlike, beautiful girl playing the festival of colors, draped in traditional Indian attire, throwing colors"

params = {}

generate_image(pipe, prompt, params)

#num inference steps
params = {'num_inference_steps': 100}

generate_image(pipe, prompt, params)

def generate_image_interface(prompt, negative_prompt, num_inference_steps=50, weight=640):
      params = {'prompt': prompt, 'num_inference_steps': num_inference_steps, 'num_images_per_prompt':2, 'height':int(1.2*weight),
                'weight': weight, 'negative_prompt': negative_prompt}

      img = pipe(prompt).images[0]
      return img

demo = gr.Interface(
        fn=generate_image_interface,
        inputs=[
            "text",  # For the prompt
            "text",  # For the negative prompt
            gr.Slider(1, 100),  # For num_inference_steps
            gr.Slider(512, 640)  # For weight
        ],
        outputs=["image"]  # For the generated images
    )
demo.launch()