


def load_params(controlnet, unet):
    controlnet.out_vae_noise_embed.load_state_dict(unet.conv_in.state_dict())
    controlnet.time_embed.load_state_dict(unet.time_embedding.state_dict())
    controlnet.unet_down.load_state_dict(unet.down_blocks.state_dict())
    controlnet.unet_mid.load_state_dict(unet.mid_block.state_dict())

def print_model_size(name, model):
    print(name, sum(i.numel() for i in model.parameters()) / 1000)

def remove_module_prefix(state_dict):
    """移除 DataParallel 模型保存时带的 'module.' 前缀"""
    new_state_dict = {}
    for k, v in state_dict.items():
        # 去除 'module.' 前缀
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict