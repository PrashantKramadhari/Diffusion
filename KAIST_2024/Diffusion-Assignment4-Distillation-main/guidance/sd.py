from diffusers import DDIMScheduler, StableDiffusionPipeline

import torch
import torch.nn as nn
import torch.nn.functional as F


class StableDiffusion(nn.Module):
    def __init__(self, args, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = args.device
        self.dtype = args.precision
        print(f'[INFO] loading stable diffusion...')

        model_key = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype,
        )

        pipe.to(self.device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.t_range = t_range
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')
        print(f'[INFO] min_step: {self.min_step}, max_step: {self.max_step}')
        print(f"[INFO] device: {self.device}, dtype: {self.dtype}")

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings
    
    
    def get_noise_preds(self, latents_noisy, t, text_embeddings, guidance_scale=100):
        latent_model_input = torch.cat([latents_noisy] * 2)
            
        tt = torch.cat([t] * 2)
        noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        
        return noise_pred


    def get_sds_loss(self, latents, text_embeddings, guidance_scale=100, grad_scale=1):
        
        # TODO: Implement the loss function for SDS

        noise = torch.randn_like(latents, device=self.device, dtype=self.dtype)


        t = torch.randint(
            self.min_step, self.max_step, (latents.shape[0],), 
            device=self.device, dtype=torch.long,
        )
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)

        # both works    
        #latents_noisy = self.scheduler.add_noise(latents, noise, t) 
        latents_noisy = (alpha_t.sqrt() * latents) + ((1 - alpha_t).sqrt() * noise)

        noise_pred = self.get_noise_preds(latents_noisy, t, text_embeddings, guidance_scale=guidance_scale)

        # this block is required, otherwise the training only produce noise
        #0.5 is the mandatory thing
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)  # [B, 1, 1, 1], for broadcasting easier, 
        grad =  w * (noise_pred - noise) # even if I drop the w term still can sample
        target = (latents - grad).detach()
        loss = 0.5 * nn.functional.mse_loss(latents, target, reduction="mean")

        return loss

    def compute_posterior_mean(self, xt, noise_pred, t, t_prev):
        """
        Computes an estimated posterior mean \mu_\phi(x_t, y; \epsilon_\phi).
        """
        device = self.device
        beta_t = self.scheduler.betas[t.cpu()].to(device)
        alpha_t = self.scheduler.alphas[t.cpu()].to(device)
        alpha_bar_t = self.scheduler.alphas_cumprod[t.cpu()].to(device)
        alpha_bar_t_prev = self.scheduler.alphas_cumprod[t_prev.cpu()].to(device)

        pred_x0 = (xt - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(
            alpha_bar_t
        )
        c0 = torch.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t)
        c1 = torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)

        mean_func = c0 * pred_x0 + c1 * xt
        return mean_func

    def get_pds_loss(self, src_latents, tgt_latents, src_text_embedding, tgt_text_embedding,guidance_scale=7.5, grad_scale=1):
        # For pds, the task is to match the stochastic term of both source and target latents
        # TODO: Implement the loss function for PDS
        # set up time dependent variables
        device = self.device
        t = torch.randint(
            self.min_step, self.max_step, (src_latents.shape[0],), 
            device=self.device, dtype=torch.long,
        )
        
        t_prev = t - 1
        
        beta_t = self.scheduler.betas[t.cpu()].to(self.device)
        alpha_bar_t = self.scheduler.alphas_cumprod[t.cpu()].to(self.device)
        alpha_bar_t_prev = self.scheduler.alphas_cumprod[t_prev.cpu()].to(self.device)
        sigma_t = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t) ** (0.5)

        noise_t = torch.randn_like(src_latents, device=self.device, dtype=self.dtype)
        # Use the same noise for both source and target at time t_prev  
        noise_t_prev = torch.randn_like(src_latents, device=self.device, dtype=self.dtype)

        src_latents_noisy = self.scheduler.add_noise(src_latents, noise_t, t)
        tgt_latents_noisy = self.scheduler.add_noise(tgt_latents, noise_t, t)

        src_latents_prev = self.scheduler.add_noise(src_latents, noise_t_prev, t_prev)
        tgt_latents_prev = self.scheduler.add_noise(tgt_latents, noise_t_prev, t_prev)

        noise_pred_src = self.get_noise_preds(src_latents_noisy, t, src_text_embedding, guidance_scale=guidance_scale)
        noise_pred_tgt = self.get_noise_preds(tgt_latents_noisy, t, tgt_text_embedding, guidance_scale=guidance_scale)

    
        # compute posterior mean
        src_mu = self.compute_posterior_mean(src_latents_noisy, noise_pred_src, t, t_prev)
        tgt_mu = self.compute_posterior_mean(tgt_latents_noisy, noise_pred_tgt, t, t_prev)

        zt_src = (src_latents_prev - src_mu) / sigma_t
        zt_tgt = (tgt_latents_prev - tgt_mu) / sigma_t

        grad = (zt_tgt- zt_src) * grad_scale
        grad = torch.nan_to_num(grad)

        target = (tgt_latents - grad).detach()
        loss = 0.5 * F.mse_loss(tgt_latents, target, reduction="mean")

        return loss.to(self.dtype)

    def get_dds_loss(self, src_latents, tgt_latents, src_text_embedding, tgt_text_embedding, guidance_scale=7.5, grad_scale=1):
        # For DDS, the task is to match the noise predict of (src_latents, src_text_embedding) and (tgt_latents, tgt_text_embedding)

        noise = torch.randn_like(src_latents, device=self.device, dtype=self.dtype)
        t = torch.randint(self.min_step, self.max_step, (src_latents.shape[0],), device=self.device, dtype=torch.long)

        src_latents_noisy = self.scheduler.add_noise(src_latents, noise, t)
        tgt_latents_noisy = self.scheduler.add_noise(tgt_latents, noise, t)

        src_noise_pred = self.get_noise_preds(src_latents_noisy, t, src_text_embedding, guidance_scale=guidance_scale)
        tgt_noise_pred = self.get_noise_preds(tgt_latents_noisy, t, tgt_text_embedding, guidance_scale=guidance_scale)

        w_t = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w_t * (tgt_noise_pred - src_noise_pred) * grad_scale
        grad = torch.nan_to_num(grad)
        target = (tgt_latents - grad).detach()
        loss = 0.5 * F.mse_loss(tgt_latents, target, reduction="mean")
        return loss
    
    
    @torch.no_grad()
    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents