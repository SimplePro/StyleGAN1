import torch
import torch.nn as nn
import torch.nn.functional as F

from time import time


GEN_CHANNEL_LIST = [
    (512, 512), # 8x8
    (512, 512), # 16x16
    (512, 256), # 32x32
    (256, 256), # 64x64
    (256, 128), # 128x128
    (128, 64), # 256x256
    (64, 64), # 512x512
]

DISC_CHANNEL_LIST = [(out_channels, in_channels) for in_channels, out_channels in GEN_CHANNEL_LIST[::-1]]

class WSConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bias = self.conv.bias
        self.conv.bias = None

        self.scale = (2/(in_channels * kernel_size**2)) ** 0.5
    
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.size(0), 1, 1)


class WSLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.scale = (2 / in_features) ** 0.5

    def forward(self, x):
        return self.linear(x * self.scale)


class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)
    

class MappingNetwork(nn.Module):

    def __init__(self, z_dim, w_dim):
        super().__init__()

        self.mapping_network = nn.Sequential(
            PixelNorm(),
            WSLinear(z_dim, w_dim),
            nn.ReLU(),
            *[
                nn.Sequential(
                    WSLinear(w_dim, w_dim),
                    nn.ReLU()
                )
                for _ in range(7)
            ]
        )

    def forward(self, x):
        return self.mapping_network(x)


class AdaIN(nn.Module):

    def __init__(self, channels, w_dim):
        super().__init__()

        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale = nn.Linear(w_dim, channels)
        self.style_bias = nn.Linear(w_dim, channels)

    def forward(self, x, w):
        out = self.instance_norm(x)
        style_scale = self.style_scale(w).view(-1, x.size(1), 1, 1)
        style_bias = self.style_bias(w).view(-1, x.size(1), 1, 1)

        out = out * style_scale + style_bias

        return out


class InjectNoise(nn.Module):
    
    def __init__(self, channels):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, x):
        noise = torch.randn((x.size(0), 1, x.size(2), x.size(3)), device=x.device)
        return x + self.weight * noise


class GenBlock(nn.Module):

    def __init__(self, in_channels, out_channels, w_dim):
        super().__init__()

        self.conv1 = WSConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = WSConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.inject_noise1 = InjectNoise(out_channels)
        self.inject_noise2 = InjectNoise(out_channels)

        self.adain1 = AdaIN(out_channels, w_dim)
        self.adain2 = AdaIN(out_channels, w_dim)

        self.leaky = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, w):
        out = self.conv1(x)
        out = self.inject_noise1(out)
        out = self.leaky(out)
        out = self.adain1(out, w)

        out = self.conv2(out)
        out = self.inject_noise2(out)
        out = self.leaky(out)
        out = self.adain2(out, w)

        return out


class Generator(nn.Module):

    def __init__(self, z_dim, w_dim, const_channels=512):
        super().__init__()

        self.const = nn.Parameter(torch.ones((1, const_channels, 4, 4)))

        self.mapping_network = MappingNetwork(z_dim, w_dim)

        self.initial_inject_noise1 = InjectNoise(const_channels)
        self.initial_adain1 = AdaIN(const_channels, w_dim)
        self.initial_conv = WSConv2d(const_channels, const_channels, kernel_size=3, stride=1, padding=1)
        self.initial_inject_noise2 = InjectNoise(const_channels)
        self.initial_adain2 = AdaIN(const_channels, w_dim)
        self.initial_to_rgb = WSConv2d(in_channels=const_channels, out_channels=3, kernel_size=1, stride=1, padding=0)

        self.leaky = nn.LeakyReLU(0.2, inplace=True)

        self.prog_blocks = nn.ModuleList([])
        self.to_rgbs = nn.ModuleList([
            WSConv2d(in_channels=const_channels, out_channels=3, kernel_size=1, stride=1, padding=0)
        ])

        for in_channels, out_channels in GEN_CHANNEL_LIST:
            self.prog_blocks.append(
                GenBlock(in_channels, out_channels, w_dim)
            )

            self.to_rgbs.append(
                WSConv2d(in_channels=out_channels, out_channels=3, kernel_size=1, stride=1, padding=0)
            )

    
    def fade_in(self, alpha, upsampled, generated):
        return alpha * generated + (1 - alpha) * upsampled

    
    def forward(self, z, alpha, step):
        w = self.mapping_network(z)

        out = self.initial_inject_noise1(self.const)
        out = self.initial_adain1(out, w)
        out = self.initial_conv(out)
        out = self.initial_inject_noise2(out)
        out = self.leaky(out)
        out = self.initial_adain2(out, w)

        if step == 0:
            return self.to_rgbs[0](out)

        for i in range(step):
            upsampled = F.interpolate(out, scale_factor=2.0, mode="bilinear")
            out = self.prog_blocks[i](upsampled, w)
        
        upsampled = self.to_rgbs[step-1](upsampled)
        out = self.to_rgbs[step](out)

        return self.fade_in(alpha, upsampled, out)

        
class DiscriminatorBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            WSConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )

    
    def forward(self, x):
        return self.conv(x)


class MinibatchStd(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_statistics = (torch.std(x, dim=0).mean().repeat(x.size(0), 1, x.size(2), x.size(3)))

        return torch.cat((x, batch_statistics), dim=1)


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.from_rgbs = nn.ModuleList([])
        self.prog_blocks = nn.ModuleList([])

        self.leaky = nn.LeakyReLU(0.2)
        self.avg_pool = nn.AvgPool2d(2, 2)

        for in_channels, out_channels in DISC_CHANNEL_LIST:
            self.prog_blocks.append(
                DiscriminatorBlock(in_channels, out_channels)
            )
            self.from_rgbs.append(
                WSConv2d(in_channels=3, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
            )

        self.from_rgbs.append(
            WSConv2d(in_channels=3, out_channels=DISC_CHANNEL_LIST[-1][-1], kernel_size=1, stride=1, padding=0)
        )

        self.final_block = nn.Sequential(
                MinibatchStd(),
                WSConv2d(in_channels=DISC_CHANNEL_LIST[-1][-1] + 1, out_channels=DISC_CHANNEL_LIST[-1][-1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                WSConv2d(in_channels=DISC_CHANNEL_LIST[-1][-1], out_channels=DISC_CHANNEL_LIST[-1][-1], kernel_size=4, stride=1, padding=0),
                nn.LeakyReLU(0.2),
                WSConv2d(in_channels=DISC_CHANNEL_LIST[-1][-1], out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Flatten(),
                nn.Sigmoid()
            )


    def fade_in(self, alpha, downsampled, out):
        return alpha * out + (1 - alpha) * downsampled


    def forward(self, x, alpha, step):
        idx = len(self.prog_blocks) - step
        out = self.leaky(self.from_rgbs[idx](x))
        
        if step == 0:
            out = self.final_block(out)
            return out
        
        downsampled = self.leaky(self.from_rgbs[idx+1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[idx](out))

        out = self.fade_in(alpha, downsampled, out)

        for i in range(idx+1, len(self.prog_blocks)):
            out = self.prog_blocks[i](out)
            out = self.avg_pool(out)
        
        out = self.final_block(out)
        
        return out

    
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen = Generator(z_dim=512, w_dim=512, const_channels=512).to(device)
    disc = Discriminator().to(device)

    print(f"gen params: {sum([p.numel() for p in gen.parameters()])}, disc params: {sum([p.numel() for p in disc.parameters()])}\n")

    batch_sizes = [128, 64, 32, 32, 16, 16, 8, 4, 4]

    alpha = 1

    for step in range(8):
        start_time = time()
        z = torch.randn((batch_sizes[step], 512)).to(device)
        out = gen(z, alpha, step)
        print(f"STEP: {step}, gen_shape: {out.shape}, disc_shape: {disc(out, alpha, step).shape}, time: {time() - start_time}")