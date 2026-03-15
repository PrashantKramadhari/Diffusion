import argparse

import numpy as np
import torch
from dataset import tensor_to_pil_image
from fm import FlowMatching
from pathlib import Path


def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    device = f"cuda:{args.gpu}"
    
    fm = FlowMatching(None, None)
    fm.load(args.ckpt_path)
    fm.eval()
    fm = fm.to(device)

    total_num_samples = 500
    num_batches = int(np.ceil(total_num_samples / args.batch_size))

    for i in range(num_batches):
        sidx = i * args.batch_size
        eidx = min(sidx + args.batch_size, total_num_samples)
        B = eidx - sidx

        if args.use_cfg:  # Enable CFG sampling
            assert fm.network.use_cfg, f"The model was not trained to support CFG."
            shape = (B, 3, fm.image_resolution, fm.image_resolution)
            samples = fm.sample(
                shape,
                num_inference_timesteps=20,
                class_label=torch.randint(1, 4, (B,)).to(device),
                guidance_scale=args.cfg_scale,
            )
        else:
            raise NotImplementedError("In Assignment 7, we sample images with CFG setup only.")

        pil_images = tensor_to_pil_image(samples)

        for j, img in zip(range(sidx, eidx), pil_images):
            img.save(save_dir / f"{j}.png")
            print(f"Saved the {j}-th image.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_scale", type=float, default=7.5)

    args = parser.parse_args()
    main(args)
