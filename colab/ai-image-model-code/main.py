# Install required libraries
!pip install -q diffusers transformers accelerate torch pillow gradio

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import gradio as gr

# Check GPU availability
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU not available. Please enable GPU: Runtime → Change runtime type → T4 GPU")

# Load Stable Diffusion model (using a publicly available model)
# Using CompVis SD 1.4 - no authentication required
model_id = "CompVis/stable-diffusion-v1-4"
# Alternative models that don't require auth:
# model_id = "prompthero/openjourney"
# model_id = "dreamlike-art/dreamlike-photoreal-2.0"

print(f"Loading image generation model: {model_id}")
print("This may take 2-3 minutes on first run...")

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None,  # Disable for faster inference
    requires_safety_checker=False
)

# Optimize for speed
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Enable memory optimization
pipe.enable_attention_slicing()

# Try to enable xformers for even faster generation
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("✓ xformers enabled for faster generation")
except:
    print("✓ Model loaded (xformers not available)")

print("✅ Model ready! You can now generate images.")

# Image generation function
def generate_image(prompt, negative_prompt="", num_steps=25, guidance_scale=7.5, width=512, height=512):
    """
    Generate an image from text prompt
    
    Args:
        prompt: Description of the image you want
        negative_prompt: Things you don't want in the image
        num_steps: More steps = better quality but slower (20-50 recommended)
        guidance_scale: How closely to follow the prompt (7-12 recommended)
        width/height: Image dimensions (must be multiples of 8)
    """
    print(f"Generating: {prompt}")
    
    try:
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height
            ).images[0]
        
        return image
    
    except Exception as e:
        print(f"Error: {e}")
        return None

# Create Gradio interface
interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(
            lines=3, 
            label="Prompt", 
            placeholder="Describe the image you want to create...",
            value="A beautiful sunset over mountains, highly detailed, 4k"
        ),
        gr.Textbox(
            lines=2, 
            label="Negative Prompt (Optional)", 
            placeholder="Things to avoid in the image...",
            value="blurry, low quality, distorted"
        ),
        gr.Slider(15, 50, value=25, step=1, label="Steps (Higher = Better Quality)"),
        gr.Slider(5, 15, value=7.5, step=0.5, label="Guidance Scale"),
        gr.Slider(256, 768, value=512, step=64, label="Width"),
        gr.Slider(256, 768, value=512, step=64, label="Height")
    ],
    outputs=gr.Image(type="pil", label="Generated Image"),
    title="🎨 AI Image Generator",
    description="Generate images from text descriptions using Stable Diffusion. Takes ~10-30 seconds per image.",
    examples=[
        ["A majestic lion in the savanna at golden hour, photorealistic", "blurry, low quality", 25, 7.5, 512, 512],
        ["A futuristic cyberpunk city with neon lights, night scene, detailed", "blurry, people", 30, 8, 512, 512],
        ["A cute corgi puppy playing in a flower garden, sunny day", "ugly, deformed", 25, 7.5, 512, 512],
        ["An astronaut riding a horse on Mars, digital art", "blurry, low resolution", 25, 7.5, 512, 512],
        ["A magical forest with glowing mushrooms, fantasy art style", "dark, scary", 30, 8, 512, 512]
    ],
    cache_examples=False
)

# Launch the interface
print("\n" + "="*60)
print("🚀 Launching Image Generator Interface...")
print("="*60)
interface.launch(share=True, debug=True)

# Alternative: Simple function to generate and display
def quick_generate(prompt, steps=25):
    """Quick generation function for notebook use"""
    image = generate_image(prompt, num_steps=steps)
    if image:
        display(image)
        return image
    return None

print("\n💡 Pro Tips:")
print("- Use detailed prompts for better results")
print("- Add art style keywords: 'digital art', 'photorealistic', 'oil painting', etc.")
print("- Use negative prompts to avoid unwanted elements")
print("- Lower steps (15-20) for faster generation")
print("- Higher steps (30-50) for better quality")
print("\n📝 Example in code:")
print('quick_generate("a magical castle in the clouds, fantasy art", steps=25)')