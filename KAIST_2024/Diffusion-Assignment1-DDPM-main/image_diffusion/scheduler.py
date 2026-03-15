from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


class BaseScheduler(nn.Module):
    def __init__(
        self, num_train_timesteps: int, beta_1: float, beta_T: float, mode="linear"
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def uniform_sample_t(
        self, batch_size, device: Optional[torch.device] = None
    ) -> torch.IntTensor:
        """
        Uniformly sample timesteps.
        """
        ts = np.random.choice(np.arange(self.num_train_timesteps), batch_size)
        ts = torch.from_numpy(ts)
        if device is not None:
            ts = ts.to(device)
        return ts

class DDPMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float,
        beta_T: float,
        mode="linear",
        sigma_type="small",
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)
    
        # sigmas correspond to $\sigma_t$ in the DDPM paper.
        self.sigma_type = sigma_type
        if sigma_type == "small":
            # when $\sigma_t^2 = \tilde{\beta}_t$.
            alphas_cumprod_t_prev = torch.cat(
                [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
            )
            sigmas = (
                (1 - alphas_cumprod_t_prev) / (1 - self.alphas_cumprod) * self.betas
            ) ** 0.5
        elif sigma_type == "large":
            # when $\sigma_t^2 = \beta_t$.
            sigmas = self.betas ** 0.5

        self.register_buffer("sigmas", sigmas)

    def step(self, x_t: torch.Tensor, t: int, eps_theta: torch.Tensor):
        """
        One step denoising function of DDPM: x_t -> x_{t-1}.

        Input:
            x_t (`torch.Tensor [B,C,H,W]`): samples at arbitrary timestep t.
            t (`int`): current timestep in a reverse process.
            eps_theta (`torch.Tensor [B,C,H,W]`): predicted noise from a learned model.
        Ouptut:
            sample_prev (`torch.Tensor [B,C,H,W]`): one step denoised sample. (= x_{t-1})
        """

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 1. Implement the DDPM reverse step.
        device = x_t.device
        t_tensor = torch.tensor(t, device=device, dtype=torch.long)
        sqrt_alphas_cumprod_t = self._get_teeth(self.alphas_cumprod.sqrt(), t_tensor)
        sqrt_one_minus_alphas_cumprod_t = self._get_teeth((1 - self.alphas_cumprod).sqrt(), t_tensor)
        x0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * eps_theta) / sqrt_alphas_cumprod_t

        betas_t = self._get_teeth(self.betas, t_tensor)
        alphas_t = self._get_teeth(self.alphas, t_tensor)
        alphas_cumprod_t = self._get_teeth(self.alphas_cumprod, t_tensor)
        alphas_cumprod_t_prev = self._get_teeth(
            torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]]), t_tensor
        )

        # Posterior mean
        coef1 = betas_t * alphas_cumprod_t_prev.sqrt() / (1 - alphas_cumprod_t)
        coef2 = alphas_t.sqrt() * (1 - alphas_cumprod_t_prev) / (1 - alphas_cumprod_t)
        posterior_mean = coef1 * x0_pred + coef2 * x_t

        # Posterior variance
        posterior_var = betas_t * (1 - alphas_cumprod_t_prev) / (1 - alphas_cumprod_t)
        posterior_var = torch.clamp(posterior_var, min=1e-20)

        # Sample noise
        #noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        noise = torch.randn_like(x_t) 
        if t_tensor.ndim == 0 or t_tensor.shape[0] == 1:
            if t == 0 or (t_tensor.ndim == 1 and t_tensor[0] == 0):
                noise = torch.zeros_like(x_t)
        else:
            noise = noise * (t_tensor != 0).float().view(-1, 1, 1, 1)

        sample_prev = posterior_mean + posterior_var.sqrt() * noise if t > 0 else posterior_mean        
        #######################
        return sample_prev
    
    # https://nn.labml.ai/diffusion/ddpm/utils.html
    def _get_teeth(self, consts: torch.Tensor, t: torch.Tensor): # get t th const 
        const = consts.gather(-1, t)
        return const.reshape(-1, 1, 1, 1)
    
    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.IntTensor,
        eps: Optional[torch.Tensor] = None,
    ):
        """
        A forward pass of a Markov chain, i.e., q(x_t | x_0).

        Input:
            x_0 (`torch.Tensor [B,C,H,W]`): samples from a real data distribution q(x_0).
            t: (`torch.IntTensor [B]`)
            eps: (`torch.Tensor [B,C,H,W]`, optional): if None, randomly sample Gaussian noise in the function.
        Output:
            x_t: (`torch.Tensor [B,C,H,W]`): noisy samples at timestep t.
            eps: (`torch.Tensor [B,C,H,W]`): injected noise.
        """
        
        if eps is None:
            #eps       = torch.randn(x_0.shape, device='cuda:0')
            eps = torch.randn_like(x_0)

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 1. Implement the DDPM forward step.
        sqrt_alphas_cumprod_t = self._get_teeth(self.alphas_cumprod.sqrt(), t)
        sqrt_one_minus_alphas_cumprod_t = self._get_teeth((1 - self.alphas_cumprod).sqrt(), t)
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * eps
        #######################

        return x_t, eps
