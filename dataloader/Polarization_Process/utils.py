import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def ensure_dir(path):
    """确保文件夹存在。"""
    os.makedirs(path, exist_ok=True)

def get_common_image_files(dir1, dir2, exts=('.png', '.jpg', '.jpeg')):
    """获取两个目录下的交集图片文件名（无序）。"""
    files1 = {f for f in os.listdir(dir1) if f.lower().endswith(exts)}
    files2 = {f for f in os.listdir(dir2) if f.lower().endswith(exts)}
    return sorted(files1 & files2)

def read_image(path, mode=cv2.IMREAD_UNCHANGED):
    """安全读取图像。"""
    img = cv2.imread(path, mode)
    if img is None:
        print(f"Warning: Cannot read image {path}")
    return img

def read_mask(mask_path, reference_shape):
    """读取mask，若无则返回全True。"""
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return mask > 0
    else:
        print(f"Warning: No mask found for {os.path.basename(mask_path)}. Using full mask.")
        return np.ones(reference_shape, dtype=bool)

def compute_error(gt_img, pred_img, mask, compare_type):
    """计算误差与可视化图。"""
    if compare_type == 'AoLP':
        gt_val = (gt_img.astype(np.float32) / 65535) * 180 - 90
        pred_val = (pred_img.astype(np.float32) / 65535) * 180 - 90
        diff1 = np.abs(pred_val - gt_val)
        diff2 = np.abs(pred_val - gt_val + 180)
        diff3 = np.abs(pred_val - gt_val - 180)
        errormap = np.minimum(np.minimum(diff1, diff2), diff3)
    else:
        gt_val = gt_img.astype(np.float32) / 65535
        pred_val = pred_img.astype(np.float32) / 65535
        errormap = (pred_val - gt_val) ** 2

    masked_errormap = errormap.copy()
    masked_errormap[~mask] = np.nan
    avg_error = np.nan if mask.sum() == 0 else np.nanmean(errormap[mask])
    return gt_val, pred_val, errormap, masked_errormap, avg_error

def visualize_map(image, save_path, cmap, vmin, vmax, mask):
    """保存带mask的可视化图。"""
    visualize_and_save(image, save_path, cmap=cmap, vmin=vmin, vmax=vmax, mask=mask)

def visualize_and_save(image, save_path, cmap='hsv', vmin=0, vmax=360, mask=None):
    print('visualize_and_save: image shape =', image.shape)
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    image_vis = image.astype(float).copy()
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        image_vis[~mask] = np.nan
    im = plt.imshow(image_vis, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()


def save_comparison_figure(
    filename, compare_type, GT_val, res_val, masked_errormap, average_error, type_folder
):
    if compare_type == 'AoLP':
        error_val_str = f"Mean Error: {average_error:.2f}°"
    else:
        error_val_str = f"Mean Error: {average_error:.4f}"

    fig = plt.figure(figsize=(18, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 0.03, 1, 0.03, 1, 0.03])

    ax0 = fig.add_subplot(gs[0, 0])
    cax0 = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[0, 2])
    cax1 = fig.add_subplot(gs[0, 3])
    ax2 = fig.add_subplot(gs[0, 4])
    cax2 = fig.add_subplot(gs[0, 5])

    if compare_type == 'AoLP':
        im0 = ax0.imshow(GT_val, cmap='hsv', vmin=-90, vmax=90)
        im1 = ax1.imshow(res_val, cmap='hsv', vmin=-90, vmax=90)
        im2 = ax2.imshow(masked_errormap, cmap='jet', vmin=0, vmax=40)
        cb0 = fig.colorbar(im0, cax=cax0)
        cb1 = fig.colorbar(im1, cax=cax1)
        cb2 = fig.colorbar(im2, cax=cax2)
        cb0.set_label('AoLP GT (°)')
        cb1.set_label('AoLP Est (°)')
        cb2.set_label('Angle Error (°)')
    else:
        im0 = ax0.imshow(GT_val, cmap='GnBu', vmin=0, vmax=1)
        im1 = ax1.imshow(res_val, cmap='GnBu', vmin=0, vmax=1)
        im2 = ax2.imshow(masked_errormap, cmap='jet', vmin=0, vmax=1)
        cb0 = fig.colorbar(im0, cax=cax0)
        cb1 = fig.colorbar(im1, cax=cax1)
        cb2 = fig.colorbar(im2, cax=cax2)
        cb0.set_label('DoLP GT')
        cb1.set_label('DoLP Est')
        cb2.set_label('MSE of DoLP')

    ax0.set_title('Original Image')
    ax1.set_title('Output Image')
    ax2.set_title('Error Map')
    for a in [ax0, ax1, ax2]:
        a.axis('off')

    fig.suptitle(f'Comparison for {filename} | {error_val_str}', fontsize=16)
    error_path = os.path.join(type_folder, f'errormap_{filename}')
    fig.savefig(error_path, bbox_inches='tight')
    plt.close(fig)

def calcStokes(images):
    I_0, I_45, I_90, I_135 = [img.astype(np.float32) for img in images]
    S0_img = (I_0 + I_90 + I_45 + I_135) / 4
    I_0, I_45, I_90, I_135 = [img.mean(-1) if img.ndim == 3 else img for img in images]
    S0 = (I_0 + I_90 + I_45 + I_135) / 2
    S1 = I_0 - I_90
    S2 = I_45 - I_135
    return S0, S1, S2, S0_img