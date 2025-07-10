import argparse
from diffusers import StableDiffusionControlNetPipeline, UNet2DConditionModel, ControlNetModel
from transformers import PretrainedConfig
from model.PolarControlnet import PolarControl
from model.utils import load_params, remove_module_prefix

from datetime import datetime
import torch
import os
import cv2
import numpy as np

class PolarControlTest(ControlNetModel):
    def __init__(self, unet):
        super().__init__(cross_attention_dim=768)
        self.controlnet = PolarControl(PretrainedConfig())
        load_params(self.controlnet, unet)

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        controlnet_cond,
        conditioning_scale=1.0,
        class_labels=None,
        timestep_cond=None,
        attention_mask=None,
        cross_attention_kwargs=None,
        return_dict=True,
        guess_mode=None,
    ):
        timestep = timestep.reshape(1)
        out_down, out_mid = self.controlnet(
            out_vae_noise=sample,
            noise_step=timestep,
            out_encoder=encoder_hidden_states,
            condition=controlnet_cond
        )
        if return_dict:
            return {"down_block_res_samples": out_down, "mid_block_res_sample": out_mid}
        return out_down, out_mid

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    h, w = (h // 8) * 8, (w // 8) * 8
    image = cv2.resize(image, (w, h))
    image = image / np.max(image)
    image = image.astype(np.float32)
    tensor_image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    tensor_image = tensor_image * 2 - 1
    return tensor_image, (h, w)

def save_output(output, save_path):
    output = np.clip(output, 0, 1)
    output = (output * 65535).astype(np.uint16)
    cv2.imwrite(save_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

def test(pipeline, input_folder, save_folder, steps=20):
    prompt = 'denoised polarized images'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = pipeline.to(device)
    os.makedirs(save_folder, exist_ok=True)
    round_folder = os.path.join(save_folder, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(round_folder)

    img_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for filename in img_files:
        img_path = os.path.join(input_folder, filename)
        tensor_img, (h, w) = preprocess_image(img_path)
        tensor_img = tensor_img.to(device)
        result = pipeline(
            prompt,
            tensor_img,
            height=h,
            width=w,
            num_inference_steps=steps,
            output_type="np"
        )
        save_output(result.images[0], os.path.join(round_folder, filename))

    print(f"Results saved to {round_folder}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Polar ControlNet Test Pipeline")
    parser.add_argument('--checkpoint', type=str, default='runwayml/stable-diffusion-v1-5',
                        help='Base Stable Diffusion checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default='./model/PA_Final_Model.pth',
                        help='Checkpoint .pth file for model weights')
    parser.add_argument('--input_folder', type=str, default='./data/RGB',
                        help='Folder containing input images')
    parser.add_argument('--results_folder', type=str, default='./results',
                        help='Folder to save results')
    parser.add_argument('--steps', type=int, default=20, help='Number of denoising steps')
    args = parser.parse_args()

    unet = UNet2DConditionModel.from_pretrained(args.checkpoint, subfolder='unet')
    controlnet = PolarControlTest(unet)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.checkpoint,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None
    )
    pipeline.unet.requires_grad_(False)
    pipeline.controlnet.requires_grad_(False)

    checkpoint = torch.load(args.checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    pipeline.unet.load_state_dict(remove_module_prefix(checkpoint['unet_state_dict']))
    pipeline.controlnet.controlnet.load_state_dict(remove_module_prefix(checkpoint['controlnet_state_dict']))

    test(pipeline, args.input_folder, args.results_folder, args.steps)

