import os 
import torch
from abc import *
from pathlib import Path
from datetime import datetime

from diffusers import StableDiffusionPipeline, DDIMScheduler, DiffusionPipeline

from diffusion.stable_diffusion import StableDiffusion


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now

    
class BaseModel(metaclass=ABCMeta):
    def __init__(self):
        self.init_model()
        self.init_mapper()
        
    def initialize(self):
        now = get_current_time()
        save_top_dir = self.config.save_top_dir
        tag = self.config.tag
        save_dir_now = self.config.save_dir_now 
        
        if save_dir_now:
            self.output_dir = Path(save_top_dir) / f"{tag}/{now}"
        else:
            self.output_dir = Path(save_top_dir) / f"{tag}"
        
        if not os.path.isdir(self.output_dir):
            self.output_dir.mkdir(exist_ok=True, parents=True)
        else:
            print(f"Results exist in the output directory, use time string to avoid name collision.")
            exit(0)
            
        print("[*] Saving at ", self.output_dir)
    
    
    @abstractmethod
    def init_mapper(self, **kwargs):
        pass
    
    
    @abstractmethod
    def forward_mapping(self, z_t, **kwargs):
        pass
    
    
    @abstractmethod
    def inverse_mapping(self, x_ts, **kwargs):
        pass
    
    
    @abstractmethod
    def compute_noise_preds(self, xts, ts, **kwargs):
        pass
        
    
    def init_model(self):
        if self.config.model == "sd":
            pipe = StableDiffusionPipeline.from_pretrained(
                self.config.sd_path,
                torch_dtype=torch.float16,
            ).to(self.device)
            
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

            # Remove image_encoder and other non-component parameters
            components = dict(pipe.components)
            components.pop('image_encoder', None)  # Remove image_encoder if present
            
            # Get requires_safety_checker separately  
            requires_safety_checker = getattr(pipe, 'requires_safety_checker', True)
            
            # Pass components in the correct order expected by StableDiffusion constructor
            self.model = StableDiffusion(
                vae=components['vae'],
                text_encoder=components['text_encoder'],
                tokenizer=components['tokenizer'],
                unet=components['unet'],
                scheduler=components['scheduler'],
                safety_checker=components['safety_checker'],
                feature_extractor=components['feature_extractor'],
                requires_safety_checker=requires_safety_checker,
            )
            
            del pipe
            
        elif self.config.model == "deepfloyd":
            self.stage_1 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-I-M-v1.0", 
                variant="fp16", 
                torch_dtype=torch.float16,
            )
            self.stage_2 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-II-M-v1.0",
                text_encoder=None,
                variant="fp16",
                torch_dtype=torch.float16,
            )
            
            scheduler = DDIMScheduler.from_config(self.stage_1.scheduler.config)
            self.stage_1.scheduler = self.stage_2.scheduler = scheduler
            
        else:
            raise NotImplementedError(f"Invalid model: {self.config.model}")
        
        
        if self.config.model in ["sd"]:
            self.model.text_encoder.requires_grad_(False)
            self.model.unet.requires_grad_(False)
            if hasattr(self.model, "vae"):
                self.model.vae.requires_grad_(False)
        else:
            self.stage_1.text_encoder.requires_grad_(False)
            self.stage_2.unet.requires_grad_(False)
            self.stage_2.unet.requires_grad_(False)
            
            self.stage_1 = self.stage_1.to(self.device)
            self.stage_2 = self.stage_2.to(self.device)
                
                
    def compute_tweedie(self, xts, eps, timestep, alphas, sigmas, **kwargs):
        """
        Input:
            xts, eps: [B,*]
            timestep: [B]
            x_t = alpha_t * x0 + sigma_t * eps
        Output:
            pred_x0s: [B,*]
        """
        
        # Get alpha and sigma for the current timestep
        alpha_t = alphas[timestep].to(xts.device)
        sigma_t = sigmas[timestep].to(xts.device)
        
        # Reshape alpha_t and sigma_t to match xts dimensions
        # alpha_t and sigma_t are scalars, need to expand to match batch and spatial dimensions
        alpha_t = alpha_t.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        sigma_t = sigma_t.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        
        # Apply Tweedie formula: x0 = (xt - sigma_t * eps) / alpha_t
        pred_x0s = (xts - sigma_t * eps) / alpha_t
        
        return pred_x0s

        
    def compute_prev_state(
        self, xts, pred_x0s, timestep, **kwargs,
    ):
        """
        Input:
            xts: [N,C,H,W]
        Output:
            pred_prev_sample: [N,C,H,W]
        """
        
        # DDIM reverse step formula: x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1 - alpha_{t-1}) * eps
        # But we need to get the noise eps from the original equation: eps = (xt - sqrt(alpha_t) * x0) / sqrt(1 - alpha_t)
        
        # Get scheduler parameters
        alphas_cumprod = self.model.scheduler.alphas_cumprod
        
        # Get alpha values for current and previous timestep
        alpha_prod_t = alphas_cumprod[timestep].to(xts.device)
        if timestep == 0:
            alpha_prod_t_prev = torch.ones_like(alpha_prod_t)
        else:
            alpha_prod_t_prev = alphas_cumprod[timestep - 1].to(xts.device)
        
        # Reshape for broadcasting
        alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
        alpha_prod_t_prev = alpha_prod_t_prev.view(-1, 1, 1, 1)
        
        # Extract noise from current state using predicted x0
        sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
        sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - alpha_prod_t)
        pred_epsilon = (xts - sqrt_alpha_prod_t * pred_x0s) / sqrt_one_minus_alpha_prod_t
        
        # DDIM reverse step
        sqrt_alpha_prod_t_prev = torch.sqrt(alpha_prod_t_prev)
        sqrt_one_minus_alpha_prod_t_prev = torch.sqrt(1 - alpha_prod_t_prev)
        pred_prev_sample = sqrt_alpha_prod_t_prev * pred_x0s + sqrt_one_minus_alpha_prod_t_prev * pred_epsilon
        
        return pred_prev_sample
        
    def one_step_process(
        self, input_params, timestep, alphas, sigmas, **kwargs
    ):
        """
        Input:
            latents: either xt or zt. [B,*]
        Output:
            output: the same with latent.
        """
        
        xts = input_params["xts"]

        eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
        x0s = self.compute_tweedie(
            xts, eps_preds, timestep, alphas, sigmas, **kwargs
        )
        
        # Synchronization using SyncTweedies 
        z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs) # Comment out to skip synchronization
        x0s = self.forward_mapping(z0s, bg=x0s, **kwargs) # Comment out to skip synchronization
        
        x_t_1 = self.compute_prev_state(xts, x0s, timestep, **kwargs)

        out_params = {
            "x0s": x0s,
            "z0s": None,
            "x_t_1": x_t_1,
            "z_t_1": None,
        }

        return out_params