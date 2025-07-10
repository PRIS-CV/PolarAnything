from diffusers.models.embeddings import TimestepEmbedding, get_timestep_embedding
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, DownBlock2D, UNetMidBlock2DCrossAttn
from transformers import PreTrainedModel, PretrainedConfig, CLIPTextModel
import torch


class PolarControl(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)

        #入参embed部分
        self.out_vae_noise_embed = torch.nn.Conv2d(4,
                                                   320,
                                                   kernel_size=3,
                                                   padding=1)

        self.time_embed = TimestepEmbedding(320, 1280, act_fn='silu')

        self.condition_embed = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 96, kernel_size=3, stride=2, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(96, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 320, kernel_size=3, stride=1, padding=1),
        )

        #unet的down部分
        self.unet_down = torch.nn.ModuleList([])
        for i in range(3):
            self.unet_down.append(
                CrossAttnDownBlock2D(num_layers=2,
                                     in_channels=[320, 320, 640][i],
                                     out_channels=[320, 640, 1280][i],
                                     temb_channels=1280,
                                     add_downsample=True,
                                     resnet_eps=1e-5,
                                     resnet_act_fn='silu',
                                     resnet_groups=32,
                                     downsample_padding=1,
                                     cross_attention_dim=768,
                                     attn_num_head_channels=8,
                                     dual_cross_attention=False,
                                     use_linear_projection=False,
                                     only_cross_attention=False,
                                     upcast_attention=False,
                                     resnet_time_scale_shift='default'))
        self.unet_down.append(
            DownBlock2D(num_layers=2,
                        in_channels=1280,
                        out_channels=1280,
                        temb_channels=1280,
                        add_downsample=False,
                        resnet_eps=1e-5,
                        resnet_act_fn='silu',
                        resnet_groups=32,
                        downsample_padding=1,
                        resnet_time_scale_shift='default'))

        #unet的mid部分
        self.unet_mid = UNetMidBlock2DCrossAttn(
            in_channels=1280,
            temb_channels=1280,
            resnet_eps=1e-5,
            resnet_act_fn='silu',
            output_scale_factor=1,
            resnet_time_scale_shift='default',
            cross_attention_dim=768,
            attn_num_head_channels=8,
            resnet_groups=32,
            use_linear_projection=False,
            upcast_attention=False)

        #control的down部分
        self.control_down = torch.nn.ModuleList([
            torch.nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(640, 640, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(640, 640, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(640, 640, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0),
        ])

        #control的mid部分
        self.control_mid = torch.nn.Conv2d(1280, 1280, kernel_size=1)

    def forward(self, out_vae_noise, noise_step, out_encoder, condition):
        #out_vae_noise -> [1, 4, 64, 64]
        #noise_step -> [1]
        #out_encoder -> [1, 77 ,768]
        #condition -> [1, 3, 512, 512]

        #编码noise_step
        #[1] -> [1, 320]
        noise_step = get_timestep_embedding(noise_step,
                                            320,
                                            flip_sin_to_cos=True,
                                            downscale_freq_shift=0)
        #[1, 320] -> [1, 1280]
        noise_step = self.time_embed(noise_step, None)

        #out_vae_noise升到高维
        #[1, 4, 64, 64] -> [1, 320, 64, 64]
        out_vae_noise = self.out_vae_noise_embed(out_vae_noise)

        #condition投影到和out_vae_noise同一维度空间
        #[1, 3, 512, 512] -> [1, 320, 64, 64]
        condition = self.condition_embed(condition)

        #向out_vae_noise中添加condition信息
        #[1, 320, 64, 64]
        out_vae_noise += condition

        #unet的down部分计算,每一层当中包括了3个串行的注意力计算,所以每一层都有3个计算结果.
        #[1, 320, 64, 64]
        #[1, 320, 64, 64]
        #[1, 320, 64, 64]
        #[1, 320, 32, 32]
        #[1, 640, 32, 32]
        #[1, 640, 32, 32]
        #[1, 640, 16, 16]
        #[1, 1280, 16, 16]
        #[1, 1280, 16, 16]
        #[1, 1280, 8, 8]
        #[1, 1280, 8, 8]
        #[1, 1280, 8, 8]
        out_unet_down = [out_vae_noise]
        for i in range(4):
            if i < 3:
                #这里只记录了out_vae_noise的维度变换,输出的维度看上面的out_unet_down
                #[1, 320, 64, 64] -> [1, 320, 32, 32]
                #[1, 320, 32, 32] -> [1, 640, 16, 16]
                #[1, 640, 16, 16] -> [1, 1280, 8, 8]

                out_vae_noise, out = self.unet_down[i](
                    hidden_states=out_vae_noise,
                    temb=noise_step,
                    encoder_hidden_states=out_encoder,
                    attention_mask=None,
                    cross_attention_kwargs=None)
            else:
                #[1, 1280, 8, 8] -> [1, 1280, 8, 8]
                out_vae_noise, out = self.unet_down[i](
                    hidden_states=out_vae_noise, temb=noise_step)

            out_unet_down.extend(out)

        #unet的mid计算,维度不变
        #[1, 1280, 8, 8] -> [1, 1280, 8, 8]
        out_vae_noise = self.unet_mid(out_vae_noise,
                                      noise_step,
                                      encoder_hidden_states=out_encoder,
                                      attention_mask=None,
                                      cross_attention_kwargs=None)

        #control的down的部分计算,维度不变,两两组合,分别计算即可
        out_control_down = [
            self.control_down[i](out_unet_down[i]) for i in range(12)
        ]

        #control的mid的部分计算,维度不变
        out_control_mid = self.control_mid(out_vae_noise)

        return out_control_down, out_control_mid