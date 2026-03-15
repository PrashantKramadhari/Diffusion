import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from dataset import AFHQDataModule, get_data_iterator, tensor_to_pil_image
from dotmap import DotMap
from fm import FlowMatching, FMScheduler
from network import UNet
from pytorch_lightning import seed_everything
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

matplotlib.use("Agg")


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now


def main(args):
    """config"""
    config = DotMap()
    config.update(vars(args))
    config.device = f"cuda:{args.gpu}"

    now = get_current_time()
    assert args.use_cfg, f"In Assignment 7, we sample images with CFG setup only."

    if args.use_cfg:
        save_dir = Path(f"results/cfg_fm-{now}")
    else:
        save_dir = Path(f"results/fm-{now}")
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"save_dir: {save_dir}")

    seed_everything(config.seed)

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    """######"""

    image_resolution = 64
    ds_module = AFHQDataModule(
        "./data",
        batch_size=config.batch_size,
        num_workers=4,
        max_num_images_per_cat=config.max_num_images_per_cat,
        image_resolution=image_resolution
    )

    train_dl = ds_module.train_dataloader()
    train_it = get_data_iterator(train_dl)

    # Set up the scheduler
    fm_scheduler = FMScheduler(sigma_min=args.sigma_min)

    network = UNet(
        image_resolution=image_resolution,
        ch=128,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=4,
        dropout=0.1,
        use_cfg=args.use_cfg,
        cfg_dropout=args.cfg_dropout,
        num_classes=getattr(ds_module, "num_classes", None),
    )

    fm = FlowMatching(network, fm_scheduler)
    fm = fm.to(config.device)

    optimizer = torch.optim.Adam(fm.network.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / config.warmup_steps, 1.0)
    )

    step = 0
    losses = []
    with tqdm(initial=step, total=config.train_num_steps) as pbar:
        while step < config.train_num_steps:
            if step % config.log_interval == 0:
                fm.eval()
                plt.plot(losses)
                plt.savefig(f"{save_dir}/loss.png")
                plt.close()
                shape = (4, 3, fm.image_resolution, fm.image_resolution)
                if args.use_cfg:
                    class_label = torch.tensor([1,1,2,3]).to(config.device)
                    samples = fm.sample(shape, class_label=class_label, guidance_scale=7.5, verbose=False)
                else:
                    samples = fm.sample(shape, return_traj=False, verbose=False)
                pil_images = tensor_to_pil_image(samples)
                for i, img in enumerate(pil_images):
                    img.save(save_dir / f"step={step}-{i}.png")

                fm.save(f"{save_dir}/last.ckpt")
                fm.train()

            img, label = next(train_it)
            img, label = img.to(config.device), label.to(config.device)
            if args.use_cfg:  # Conditional, CFG training
                loss = fm.get_loss(img, class_label=label)
            else:  # Unconditional training
                loss = fm.get_loss(img)
            pbar.set_description(f"Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

            step += 1
            pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--train_num_steps",
        type=int,
        default=100000,
        help="the number of model training steps.",
    )
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument(
        "--max_num_images_per_cat",
        type=int,
        default=3000,
        help="max number of images per category for AFHQ dataset",
    )
    parser.add_argument("--sigma_min", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--image_resolution", type=int, default=64)
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
