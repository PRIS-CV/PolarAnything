import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import cv2
import os

class PolarDataset(Dataset):
    def __init__(self, image_dir, conditioning_image_dir, tokenizer):
        self.image_dir = image_dir
        self.conditioning_image_dir = conditioning_image_dir
        self.tokenizer = tokenizer

        # 获取两个目录中的所有文件名，并取交集，确保图片和条件图片的文件名完全一致
        image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        conditioning_image_filenames = sorted([f for f in os.listdir(conditioning_image_dir) if f.endswith('.png')])

        # 取主图片和条件图片文件名的交集，确保文件名匹配
        self.image_filenames = sorted(list(set(image_filenames) & set(conditioning_image_filenames)))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        conditioning_image_filename = image_filename  # 使用相同的文件名加载条件图片

        img_path = os.path.join(self.image_dir, image_filename)
        conditioning_img_path = os.path.join(self.conditioning_image_dir, conditioning_image_filename)

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        conditioning_image = cv2.imread(conditioning_img_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError(f"无法读取图片: {image_filename}")
        if conditioning_image is None:
            raise ValueError(f"无法读取条件图片: {conditioning_image_filename}")

        if image.shape != conditioning_image.shape:
            raise ValueError(
                f"图片 {image_filename} 和条件图片 {conditioning_image_filename} 的尺寸不一致: {image.shape} vs {conditioning_image.shape}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        conditioning_image = cv2.cvtColor(conditioning_image, cv2.COLOR_BGR2RGB)

        image, conditioning_image = self.random_crop(image, conditioning_image, 512, 512)

        image_max = np.max(image)

        image = (image.astype(np.float32) / image_max) * 2 - 1
        conditioning_image = (conditioning_image.astype(np.float32) / 65535.0) * 2 - 1

        pixel_values = torch.from_numpy(image).permute(2, 0, 1).float()
        conditioning_pixel_values = torch.from_numpy(conditioning_image).permute(2, 0, 1).float()

        # valid_mask = torch.ones_like(pixel_values).bool()  # empty mask

        text = self.tokenizer.batch_encode_plus(['denoised polarized images'], max_length=77,
                                                     padding='max_length', truncation=True, return_tensors='pt').input_ids.squeeze()
        return {
            'polarization': pixel_values,
            'rgb': conditioning_pixel_values,
            # 'val_mask': valid_mask,
            'input_ids': text,
        }

    def random_crop(self, image, conditioning_image, crop_height, crop_width):
        max_x = max(image.shape[1] - crop_width, 0)
        max_y = max(image.shape[0] - crop_height, 0)
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        image_cropped = image[y:y + crop_height, x:x + crop_width]
        conditioning_image_cropped = conditioning_image[y:y + crop_height, x:x + crop_width]
        return image_cropped, conditioning_image_cropped

if __name__ == "__main__":
    from utils import show_batch_images, custom_collate_fn
    image_dir = './data/Polarization_Encoding'
    conditioning_image_dir = './data/RGB'
    checkpoint = 'runwayml/stable-diffusion-v1-5'
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(checkpoint, subfolder='tokenizer')
    dataset = PolarDataset(image_dir, conditioning_image_dir, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    try:
        for idx, batch in enumerate(dataloader):
            if batch is None:
                print(f"跳过空的 batch")
                continue

            print("Input IDs shape:", batch['input_ids'].shape)
            print("Pixel values shape:", batch['pixel_values'].shape)
            print("Conditioning pixel values shape:", batch['conditioning_pixel_values'].shape)

            show_batch_images(batch['pixel_values'], "Pixel Values")
            show_batch_images(batch['conditioning_pixel_values'], "Conditioning Pixel Values")
    except ValueError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"Runtime error: {e}")
