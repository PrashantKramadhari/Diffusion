from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch

# stage 1
stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-M-v1.0", variant="fp16", torch_dtype=torch.float16)
stage_1.enable_model_cpu_offload()

# stage 2
stage_2 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-M-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
)
stage_2.enable_model_cpu_offload()

# # stage 3
# safety_modules = {
#     "feature_extractor": stage_1.feature_extractor,
#     "safety_checker": stage_1.safety_checker,
#     "watermarker": stage_1.watermarker,
# }
# stage_3 = DiffusionPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
# )
# stage_3.enable_model_cpu_offload()

prompt = 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"'
generator = torch.manual_seed(1)

# text embeds
prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)

# stage 1
stage_1_output = stage_1(
    prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt"
).images
pt_to_pil(stage_1_output)[0].save("./if_stage_I.png")

# stage 2
stage_2_output = stage_2(
    image=stage_1_output,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    generator=generator,
    output_type="pt",
).images
pt_to_pil(stage_2_output)[0].save("./if_stage_II.png")

# stage 3
# stage_3_output = stage_3(prompt=prompt, image=stage_2_output, noise_level=100, generator=generator).images
# #stage_3_output[0].save("./if_stage_III.png")
#make_image_grid([pt_to_pil(stage_1_output)[0], pt_to_pil(stage_2_output)[0], stage_3_output[0]], rows=1, rows=3)