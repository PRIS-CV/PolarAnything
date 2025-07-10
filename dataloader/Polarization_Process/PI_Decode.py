import os
import cv2
import numpy as np
import argparse

def PI_decode(input_folder, output_folder):
    # 输出子文件夹
    aolp_16bit_folder = os.path.join(output_folder, 'AoLP_16bit')
    dolp_16bit_folder = os.path.join(output_folder, 'DoLP_16bit')
    os.makedirs(aolp_16bit_folder, exist_ok=True)
    os.makedirs(dolp_16bit_folder, exist_ok=True)

    # 检查点文件用于断点续跑
    checkpoint_file = os.path.join(output_folder, 'processed.txt')
    processed_files = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed_files = set(line.strip() for line in f if line.strip())

    # 遍历所有图片
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for filename in image_files:
        if filename in processed_files:
            print(f"Skip: {filename} (already processed)")
            continue

        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim < 3 or image.shape[-1] < 3:
            print(f"Skip: {filename} (not found or less than 3 channels)")
            continue

        # 判断输入图片为8位还是16位
        img_dtype = image.dtype
        maxval = 255 if img_dtype == np.uint8 else 65535

        # 分离通道
        sin_aolp_2, cos_aolp_2, DoLP = cv2.split(image)
        sin_aolp_2 = sin_aolp_2.astype(np.float32) / maxval * 2 - 1
        cos_aolp_2 = cos_aolp_2.astype(np.float32) / maxval * 2 - 1

        denominator = np.sqrt(sin_aolp_2 ** 2 + cos_aolp_2 ** 2)
        denominator[denominator == 0] = 1e-8
        sin_AoLP_2, cos_AoLP_2 = sin_aolp_2 / denominator, cos_aolp_2 / denominator

        AoLP = 0.5 * np.arctan2(sin_AoLP_2, cos_AoLP_2)
        im_aolp_normalized = (AoLP + 0.5 * np.pi) / np.pi
        im_aolp_16bit = (im_aolp_normalized * 65535).round().astype(np.uint16)
        im_dolp_16bit = (DoLP.astype(np.float32) / maxval * 65535).round().astype(np.uint16)

        # 保存
        cv2.imwrite(os.path.join(aolp_16bit_folder, filename), im_aolp_16bit)
        cv2.imwrite(os.path.join(dolp_16bit_folder, filename), im_dolp_16bit)

        with open(checkpoint_file, 'a') as f:
            f.write(filename + '\n')

        print(f"Processed: {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Polarization Image Decoding')
    parser.add_argument('--input_folder', type=str, default=r'..\..\results\20250709_034046', help='输入文件夹')
    parser.add_argument('--output_folder', type=str, default=r'..\..\results\20250709_034046_decode', help='输出根目录')
    args = parser.parse_args()

    PI_decode(args.input_folder, args.output_folder)
