import torch
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer


class StableDiffusion(StableDiffusionPipeline):
    def __init__(
        self, 
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        
        super().__init__(
            vae, text_encoder, tokenizer, unet, 
            scheduler, safety_checker, 
            feature_extractor, requires_safety_checker
        )
        
        self.scheduler = scheduler


    def compute_noise_preds(
        self, 
        xts, 
        ts, 
        prompt_embeds,
        guidance_scale,
        **kwargs,
    ):
        
        C, H, W = xts.shape[-3], xts.shape[-2], xts.shape[-1]
        xts = xts.reshape(-1, C, H, W)

        noise_pred_dict = {}
        for mode, prompt in zip(["uncond", "text"], prompt_embeds):
            noise_pred_list = []
            
            xt_input = self.scheduler.scale_model_input(xts, ts)
            prompt_embeds_batch = [prompt] * xt_input.shape[0]
            prompt_embeds_batch = torch.stack(prompt_embeds_batch, dim=0)
            
            noise_preds = self.unet(
                xt_input.half(),
                ts,
                encoder_hidden_states=prompt_embeds_batch,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]
                
            noise_pred_list.append(noise_preds)
            noise_pred_dict[mode] = torch.cat(noise_pred_list, dim=0)

        noise_preds_stack = noise_pred_dict["uncond"] + guidance_scale * (noise_pred_dict["text"] - noise_pred_dict["uncond"])

        return noise_preds_stack