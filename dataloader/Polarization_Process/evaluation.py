import argparse
import os
import pandas as pd
from tqdm import tqdm
from utils import ensure_dir, read_image, read_mask, compute_error, visualize_map, save_comparison_figure, get_common_image_files

def compare_polar_results_with_GT(
    GT_folder, results_folder, output_folder, mask_folder, compare_type, visualize_maps=True
):
    # === 1. 路径 ===
    ensure_dir(output_folder)
    errormap_folder = os.path.join(output_folder, 'errormap')
    ensure_dir(errormap_folder)

    # 选择模式参数
    if compare_type == 'AoLP':
        gt_dir = os.path.join(GT_folder, 'AoLP_16bit')
        pred_dir = os.path.join(results_folder, 'AoLP_16bit')
        vmin, vmax, cmap = -90, 90, 'hsv'
        error_tag = 'AoLP_MAE'
    elif compare_type == 'DoLP':
        gt_dir = os.path.join(GT_folder, 'DoLP_16bit')
        pred_dir = os.path.join(results_folder, 'DoLP_16bit')
        vmin, vmax, cmap = 0, 1, 'GnBu'
        error_tag = 'DoLP_MSE'
    else:
        raise ValueError("compare_type must be 'AoLP' or 'DoLP'")

    type_folder = os.path.join(errormap_folder, error_tag)
    ensure_dir(type_folder)
    vis_gt_folder = os.path.join(errormap_folder, f'{compare_type}_GT_vis')
    vis_est_folder = os.path.join(errormap_folder, f'{compare_type}_EST_vis')
    if visualize_maps:
        ensure_dir(vis_gt_folder)
        ensure_dir(vis_est_folder)

    # === 2. 共有文件 ===
    common_files = get_common_image_files(gt_dir, pred_dir)
    results_list = []

    # === 3. 进度条程序 ===
    for fname in tqdm(common_files, desc=f'Comparing {compare_type}'):
        gt_img = read_image(os.path.join(gt_dir, fname))
        pred_img = read_image(os.path.join(pred_dir, fname))
        if gt_img is None or pred_img is None:
            continue

        mask = read_mask(os.path.join(mask_folder, fname), gt_img.shape)
        gt_val, pred_val, errormap, masked_errormap, avg_error = compute_error(
            gt_img, pred_img, mask, compare_type
        )
        results_list.append({'filename': fname, 'average_error': avg_error})

        # 可视化
        if visualize_maps:
            visualize_map(gt_val, os.path.join(vis_gt_folder, f'{fname}_GT_vis.png'), cmap, vmin, vmax, mask)
            visualize_map(pred_val, os.path.join(vis_est_folder, f'{fname}_EST_vis.png'), cmap, vmin, vmax, mask)

        save_comparison_figure(
            fname, compare_type, gt_val, pred_val, masked_errormap, avg_error, type_folder
        )

    # === 4. 汇总csv输出 ===
    df = pd.DataFrame(results_list)
    if not df.empty:
        overall = df['average_error'].mean()
        df.loc[len(df)] = {'filename': 'Overall', 'average_error': overall}
    output_csv = os.path.join(type_folder, f'{compare_type}_error_results.csv')
    df.to_csv(output_csv, index=False)
    print(f"[{compare_type}] Results saved to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compare polarization result maps with GT and output error maps and stats."
    )
    parser.add_argument('--GT_folder', type=str,
                        default=r'..\..\data\demo\GT\Polarization_Encoding_GT_decode',
                        help='Path to ground truth folder')
    parser.add_argument('--results_folder', type=str,
                        default=r'..\..\results\20250709_034046_decode',
                        help='Path to predicted results folder')
    parser.add_argument('--output_folder', type=str,
                        default=r'..\..\evaluation',
                        help='Output folder')
    parser.add_argument('--mask_folder', type=str,
                        default=r'..\..\data\demo\GT\mask',
                        help='Mask folder')
    parser.add_argument('--compare_type', type=str, choices=['AoLP', 'DoLP', 'all'], default='all',
                        help="Compare type: 'AoLP', 'DoLP', or 'all'")
    parser.add_argument(
        '--visualize_maps',
        type=bool,
        default=False,
        help="Whether to save AoLP/DoLP visualizations"
    )
    args = parser.parse_args()

    if args.compare_type in ['AoLP', 'all']:
        compare_polar_results_with_GT(
            args.GT_folder, args.results_folder, args.output_folder,
            args.mask_folder, compare_type='AoLP', visualize_maps=args.visualize_maps
        )
    if args.compare_type in ['DoLP', 'all']:
        compare_polar_results_with_GT(
            args.GT_folder, args.results_folder, args.output_folder,
            args.mask_folder, compare_type='DoLP', visualize_maps=args.visualize_maps
        )
