import argparse
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import PretrainedConfig, CLIPTextModel, CLIPTokenizer
from model.PolarControlnet import PolarControl
from model.utils import load_params, print_model_size
from module.tools import save_checkpoint, load_checkpoint
from dataloader.Dataset_Polarization import PolarDataset
from dataloader.utils import custom_collate_fn

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from itertools import chain
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Train Polarization ControlNet")
    parser.add_argument('--num_epochs', type=int, default=4000, help='训练总轮次')
    parser.add_argument('--batch_size', type=int, default=1, help='训练 batch size')
    parser.add_argument('--lr', type=float, default=4e-5, help='初始学习率')
    parser.add_argument('--save_ckpt_freq', type=int, default=100, help='每多少 epoch 保存一次 checkpoint')
    parser.add_argument('--polarization_dir', type=str, default='./data/Polarization_Encoding', help='Polarization 文件夹路径')
    parser.add_argument('--rgb_dir', type=str, default='./data/RGB', help='RGB 文件夹路径')
    parser.add_argument('--enable_xformers_memory_efficient_attention',
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help='是否启用xformers高效注意力，默认开启')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='保存中间checkpoint的文件夹')
    parser.add_argument('--continue_checkpoint_path', type=str, default='./checkpoints/ckpt.pth', help='Checkpoint文件路径')
    parser.add_argument('--model_dir', type=str, default='./model', help='保存最终model的文件夹')
    return parser.parse_args()

def build_models(enable_xformers):
    checkpoint = 'runwayml/stable-diffusion-v1-5'
    encoder    = CLIPTextModel.from_pretrained(checkpoint, subfolder='text_encoder')
    tokenizer  = CLIPTokenizer.from_pretrained(checkpoint, subfolder='tokenizer')
    vae        = AutoencoderKL.from_pretrained(checkpoint, subfolder='vae')
    unet       = UNet2DConditionModel.from_pretrained(checkpoint, subfolder='unet')
    scheduler  = DDPMScheduler.from_pretrained(checkpoint, subfolder='scheduler')
    controlnet = PolarControl(PretrainedConfig())
    load_params(controlnet, unet)

    if enable_xformers:
        unet.enable_xformers_memory_efficient_attention()
        for m in controlnet.modules():
            if hasattr(m, 'enable_xformers_memory_efficient_attention'):
                m.enable_xformers_memory_efficient_attention()

    for name, module in zip(['encoder', 'vae', 'unet', 'controlnet'],
                            [encoder, vae, unet, controlnet]):
        print_model_size(name, module)

    # 冻结部分参数
    vae.requires_grad_(False)
    encoder.requires_grad_(False)
    unet.requires_grad_(True)
    controlnet.requires_grad_(True)

    return encoder, tokenizer, vae, unet, scheduler, controlnet

def get_loss(data, encoder, vae, scheduler, unet, controlnet, device):
    for key in ['input_ids', 'polarization', 'rgb']:
        data[key] = data[key].to(device)
    out_encoder = encoder(data['input_ids'])[0]
    out_vae = vae.encode(data['polarization']).latent_dist.sample() * 0.18215

    noise = torch.randn_like(out_vae)
    noise_step = torch.randint(0, 1000, (1,), device=device).long()
    out_vae_noise = scheduler.add_noise(out_vae, noise, noise_step)

    control_down, control_mid = controlnet(
        out_vae_noise, noise_step, out_encoder, condition=data['rgb']
    )

    out_unet = unet(
        out_vae_noise, noise_step,
        encoder_hidden_states=out_encoder,
        down_block_additional_residuals=control_down,
        mid_block_additional_residual=control_mid
    ).sample

    return torch.nn.functional.mse_loss(out_unet, noise)

def train(num_epochs, dataloader, encoder, vae, scheduler, unet, controlnet, optimizer,
          checkpoint_dir, continue_checkpoint_path, model_dir, save_ckpt_freq, accelerator):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    summaries_dir = os.path.join(model_dir, 'summaries')
    if accelerator.is_main_process:
        os.makedirs(summaries_dir, exist_ok=True)

    start_epoch, epoch_losses = load_checkpoint(continue_checkpoint_path, unet, controlnet, optimizer)
    total_steps = 0
    start_time = time.time()

    dataloader = accelerator.prepare(dataloader)
    if accelerator.is_main_process:
        writer = SummaryWriter(summaries_dir)
    else:
        writer = None

    try:
        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0
            grad_norm = 0
            progress = tqdm(enumerate(dataloader), total=len(dataloader),
                            desc=f"Epoch {epoch+1}/{num_epochs}") if accelerator.is_main_process else enumerate(dataloader)

            for i, data in progress:
                loss = get_loss(data, encoder, vae, scheduler, unet, controlnet, accelerator.device) / 4
                accelerator.backward(loss)
                epoch_loss += accelerator.gather(loss).sum().item()

                if i % 4 == 0:
                    params = list(chain(unet.parameters(), controlnet.parameters()))
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_steps += 1
                if writer:
                    writer.add_scalar("train/loss_step", loss.item(), total_steps)
                    writer.add_scalar("train/grad_norm", grad_norm if isinstance(grad_norm, float) else grad_norm.item(), total_steps)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], total_steps)
                    if i % 200 == 0:
                        rgb_img = data['rgb'][0].detach().cpu()
                        writer.add_image('train/input_rgb', (rgb_img - rgb_img.min())/(rgb_img.max() - rgb_img.min() + 1e-8), total_steps)
                if accelerator.is_main_process:
                    progress.set_postfix(loss=epoch_loss)

            if accelerator.is_main_process:
                epoch_losses.append(epoch_loss)
                avg_epoch_loss = epoch_loss / len(dataloader)
                time_per_epoch = (time.time() - start_time) / (epoch + 1)
                print(f"Epoch {epoch+1} Done, Loss: {epoch_loss:.4f}, Avg: {avg_epoch_loss:.4f}, ETA: {time_per_epoch * (num_epochs-epoch-1)/60:.2f}min")
                if writer:
                    writer.add_scalar("train/loss_epoch", avg_epoch_loss, epoch + 1)

                if (epoch + 1) % save_ckpt_freq == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"ckpt_epoch{epoch+1}.pth")
                    save_checkpoint(epoch, unet, controlnet, optimizer, epoch_losses, checkpoint_path)

        if accelerator.is_main_process:
            final_model_path = os.path.join(model_dir, 'PA_Final_Model.pth')
            torch.save({
                'epoch': num_epochs,
                'unet_state_dict': unet.cpu().state_dict(),
                'controlnet_state_dict': controlnet.cpu().state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, final_model_path)
            print('Training complete. Final model saved to:', final_model_path)
    finally:
        if writer:
            writer.close()

def main():
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    # 模型和优化器
    encoder, tokenizer, vae, unet, scheduler, controlnet = build_models(
        args.enable_xformers_memory_efficient_attention
    )
    optimizer = torch.optim.AdamW(
        chain(unet.parameters(), controlnet.parameters()),
        lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-8
    )
    encoder, vae, controlnet, unet, optimizer = accelerator.prepare(
        encoder, vae, controlnet, unet, optimizer
    )

    # 数据集
    dataset = PolarDataset(args.polarization_dir, args.rgb_dir, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn
    )

    train(
        num_epochs=args.num_epochs,
        dataloader=dataloader,
        encoder=encoder,
        vae=vae,
        scheduler=scheduler,
        unet=unet,
        controlnet=controlnet,
        optimizer=optimizer,
        checkpoint_dir=args.checkpoint_dir,
        continue_checkpoint_path=args.continue_checkpoint_path,
        model_dir=args.model_dir,
        save_ckpt_freq=args.save_ckpt_freq,
        accelerator=accelerator
    )

if __name__ == "__main__":
    main()
