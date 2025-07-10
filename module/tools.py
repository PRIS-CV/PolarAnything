import torch
import os

def save_checkpoint(epoch, model, controlnet, optimizer, epoch_losses, checkpoint_path):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if isinstance(controlnet, torch.nn.DataParallel):
        controlnet = controlnet.module

    state = {
        'epoch': epoch,
        'unet_state_dict': model.state_dict(),
        'controlnet_state_dict': controlnet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch_losses': epoch_losses
    }
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch + 1}")

def load_checkpoint(checkpoint_path, model, controlnet, optimizer):
    if os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location='cuda')

        # 直接加载模型权重
        model.load_state_dict(state['unet_state_dict'])
        controlnet.load_state_dict(state['controlnet_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])

        # 使用 .get() 来安全获取 'epoch' 和 'epoch_losses'
        epoch = state.get('epoch', 0)  # 如果 'epoch' 不存在，默认返回 0
        epoch_losses = state.get('epoch_losses', [])  # 如果 'epoch_losses' 不存在，默认返回空列表

        print(f"Checkpoint loaded from epoch {epoch + 1}")
        return epoch, epoch_losses
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0, []