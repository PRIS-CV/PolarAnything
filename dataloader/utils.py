import matplotlib.pyplot as plt
import torch

def show_image(tensor, title):
    image = tensor.cpu().detach().numpy().transpose(1, 2, 0)
    image = (image + 1) / 2.0

    if image.shape[2] == 1:
        image = image[:, :, 0]
        plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    else:
        plt.imshow(image, vmin=0, vmax=1)

    plt.title(title)
    plt.axis('off')


def show_batch_images(batch_images, title, max_images_per_row=4):
    batch_size = batch_images.shape[0]
    plt.figure(figsize=(12, 12))

    num_rows = (batch_size + max_images_per_row - 1) // max_images_per_row

    for i in range(batch_size):
        plt.subplot(num_rows, max_images_per_row, i + 1)
        show_image(batch_images[i], f"{title} {i + 1}")

    plt.tight_layout()
    plt.show()


def custom_collate_fn(batch):
    # 过滤掉任何为 None 的条目
    batch = [item for item in batch if item is not None]

    # 如果 batch 是空的，则返回 None
    if len(batch) == 0:
        return None

    try:
        return torch.utils.data.default_collate(batch)
    except RuntimeError as e:
        print(f"Batch 中的元素尺寸不一致: {e}")
        for i, item in enumerate(batch):
            print(
                f"元素 {i} 的尺寸: pixel_values: {item['pixel_values'].shape}, conditioning_pixel_values: {item['conditioning_pixel_values'].shape}")
        raise e
