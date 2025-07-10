import os
import cv2
import numpy as np
import argparse
from utils import calcStokes

def encode_from_pol(base_folder, subfolders, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    images_dict = {}
    for subfolder in subfolders:
        folder_path = os.path.join(base_folder, subfolder)
        images_dict[subfolder] = sorted([
            f for f in os.listdir(folder_path) if f.lower().endswith('.png')
        ])
    common_images = set(images_dict[subfolders[0]])
    for subfolder in subfolders[1:]:
        common_images &= set(images_dict[subfolder])
    common_images = sorted(list(common_images))
    if not common_images:
        print("No common images found in all subfolders!")
        return
    for image_name in common_images:
        group_images = []
        for subfolder in subfolders:
            image_path = os.path.join(base_folder, subfolder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"Warning: Failed to read {image_path}")
                break
            group_images.append(image)
        else:
            s0, s1, s2, _ = calcStokes(group_images)
            im_dolp = (np.sqrt(s1 ** 2 + s2 ** 2) / (s0 + 1e-8)).clip(0, 1)
            im_aolp = 0.5 * np.arctan2(s2, (s1 + 1e-8))
            sin_aolp = np.sin(im_aolp * 2)
            cos_aolp = np.cos(im_aolp * 2)
            sin_aolp_16bit = ((sin_aolp + 1) / 2 * 65535).astype(np.uint16)
            cos_aolp_16bit = ((cos_aolp + 1) / 2 * 65535).astype(np.uint16)
            im_dolp_16bit = (im_dolp * 65535).astype(np.uint16)
            Polarization_Encoding = np.stack((sin_aolp_16bit, cos_aolp_16bit, im_dolp_16bit), axis=-1)
            save_path = os.path.join(output_folder, image_name)
            cv2.imwrite(save_path, Polarization_Encoding)
            print(f"Saved Polarization_Encoding: {image_name}")

def encode_from_aolp_dolp(input_folder_aolp, input_folder_dolp, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    aolp_files = sorted([f for f in os.listdir(input_folder_aolp) if f.lower().endswith('.png')])
    dolp_files = sorted([f for f in os.listdir(input_folder_dolp) if f.lower().endswith('.png')])
    common_files = sorted(list(set(aolp_files) & set(dolp_files)))
    if not common_files:
        print("No common images found in both AoLP and DoLP folders!")
        return
    for image_name in common_files:
        aolp_path = os.path.join(input_folder_aolp, image_name)
        dolp_path = os.path.join(input_folder_dolp, image_name)
        aolp_16bit = cv2.imread(aolp_path, cv2.IMREAD_UNCHANGED)
        dolp_16bit = cv2.imread(dolp_path, cv2.IMREAD_UNCHANGED)
        if aolp_16bit is None or dolp_16bit is None:
            print(f"Warning: Failed to read {aolp_path} or {dolp_path}")
            continue

        aolp_normalized = aolp_16bit.astype(np.float64) / 65535  # [0, 1]
        aolp = aolp_normalized * np.pi - 0.5 * np.pi             # [-0.5π, 0.5π]

        sin_aolp = np.sin(2 * aolp)
        cos_aolp = np.cos(2 * aolp)
        sin_aolp_16bit = ((sin_aolp + 1) / 2 * 65535).round().astype(np.uint16)
        cos_aolp_16bit = ((cos_aolp + 1) / 2 * 65535).round().astype(np.uint16)

        polarization_encoding = np.stack((sin_aolp_16bit, cos_aolp_16bit, dolp_16bit), axis=-1)
        save_path = os.path.join(output_folder, image_name)
        cv2.imwrite(save_path, polarization_encoding)
        print(f"Saved Polarization_Encoding (from AoLP+DoLP): {image_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polarization encoding from two types of input.")
    parser.add_argument('--input_type', choices=['pol', 'ad'], default='ad',
                        help="输入类型：'pol'为四通道偏振图，'ad'为AoLP_16bit+DoLP_16bit单通道图")
    parser.add_argument('--base_folder', type=str,
                        default=r'D:\1First_pro\__000FinalRelease\code\PolarAnything\data\val_data\GT\Polarization_Encoding_GT_decode',
                        help='根目录，包含pol/AoLP_16bit/DoLP_16bit等子文件夹')
    parser.add_argument('--subfolders', nargs=4, type=str,
                        default=['pol000', 'pol045', 'pol090', 'pol135'],
                        help='偏振子文件夹名称，按顺序为0,45,90,135')
    parser.add_argument('--output', type=str,
                        default=r'D:\1First_pro\__000FinalRelease\code\PolarAnything\data\val_data\GT\Polarization_Encoding_GT_decode_test',
                        help='输出三通道编码目录')
    parser.add_argument('--aolp_folder', type=str, default='AoLP_16bit', help='AoLP_16bit单通道输入文件夹')
    parser.add_argument('--dolp_folder', type=str, default='DoLP_16bit', help='DoLP_16bit单通道输入文件夹')
    args = parser.parse_args()

    if args.input_type == 'pol':
        encode_from_pol(args.base_folder, args.subfolders, args.output)
    elif args.input_type == 'ad':
        aolp_folder = os.path.join(args.base_folder, args.aolp_folder)
        dolp_folder = os.path.join(args.base_folder, args.dolp_folder)
        encode_from_aolp_dolp(aolp_folder, dolp_folder, args.output)
    else:
        print("Unknown input_type!")
