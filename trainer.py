import torch
from torch.optim import Adam
import torch.nn.functional as F

from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from tqdm import tqdm

from models import Generator, Discriminator

from utils import get_loader

import wandb


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = {
    "Z_DIM": 512,
    "W_DIM": 512,
    "LAMBDA_GP": 10,
    "EPOCHS": [20, 25, 35, 45, 55, 65, 80],
    "BATCH_SIZES": [256, 256, 128, 64, 32, 16, 8, 4],
    "IMAGE_SIZES": [4, 8, 16, 32, 64, 128, 256, 512],
}

# wgan-gp
def get_gradient_penalty(disc, real_img, fake_img, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real_img.shape
    beta = torch.randn((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real_img * beta + fake_img.detach() * (1 - beta)
    interpolated_images.requires_grad = True

    mixed_scores = disc(interpolated_images, alpha, train_step)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(BATCH_SIZE, -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty


class Trainer:

    def __init__(
        self,
        gen,
        disc,
        lr=1e-3
    ):

        self.gen = gen.to(DEVICE)
        self.disc = disc.to(DEVICE)

        self.opt_gen = Adam([
            {"params": [param for name, param in self.gen.named_parameters() if 'mapping_network' not in name]},
            {"params": self.gen.mapping_network.parameters(), 'lr': lr*(1e-2)}
        ], lr=lr, betas=(0.0, 0.99))
    
        self.opt_disc = Adam(self.disc.parameters(), lr=lr, betas=(0.0, 0.99))


        self.test_z = torch.randn((16, 512)).to(DEVICE)

        self.reset_alpha()

        self.step = 0

    
    def reset_alpha(self):
        self.alpha = 1e-7

    
    @torch.no_grad()
    def test_fn(self):
        self.gen.eval()
        self.disc.eval()

        pred = self.gen(self.test_z, self.alpha, self.step)
        pred = pred.detach().cpu()

        image = make_grid(pred, nrow=4, normalize=True)
        image = to_pil_image(image)

        return image


    def train_fn(self, epochs, loader):
        self.gen.train()
        self.disc.train()

        loop = tqdm(loader, leave=True)

        for real, _ in loop:

            real = real.to(DEVICE)
            
            z = torch.randn(real.size(0), args["Z_DIM"]).to(DEVICE)
            fake = self.gen(z, self.alpha, self.step)

            disc_real = self.disc(real, self.alpha, self.step)
            disc_fake = self.disc(fake.detach(), self.alpha, self.step)
            gp = get_gradient_penalty(self.disc, real, fake, self.alpha, self.step, DEVICE)
            disc_loss = (
                -(disc_real.mean() - disc_fake.mean())
                + args["LAMBDA_GP"] * gp
                + (0.001) * torch.mean(disc_real ** 2)
            )

            self.opt_disc.zero_grad()
            disc_loss.backward()
            self.opt_disc.step()

            gen_fake = self.disc(fake, self.alpha, self.step)
            gen_loss = -gen_fake.mean()

            self.opt_gen.zero_grad()
            gen_loss.backward()
            self.opt_gen.step()

            self.alpha += 1 / (epochs * len(loader) * 0.7)
            self.alpha = min(self.alpha, 1)

            loop.set_postfix(
                gp=gp.item(),
                gen_loss=gen_loss.item(),
                disc_loss=disc_loss.item()
            )

            wandb.log({"gp": gp.item(), "gen_loss": gen_loss.item(), "disc_loss": disc_loss.item()})


    def run(self, step, epochs, loader):
        print(f"\n\nImage Size: {args['IMAGE_SIZES'][step]}x{args['IMAGE_SIZES'][step]}\n")
        self.step = step
        wandb.log({"step": step})

        for epoch in range(epochs):
            wandb.log({"cur_epoch": epoch+1})
            print(f"EPOCH: {epoch+1}/{epochs}")
            self.train_fn(epochs, loader)
            test_image = self.test_fn()
            test_image.save(f"test_images/{step}/{epoch}.jpg")
            wandb.log({f"test_image{step}": wandb.Image(test_image)})

            torch.save(self.gen.state_dict(), "./gen_state_dict.pt")
            torch.save(self.disc.state_dict(), "./disc_state_dict.pt")
        
        
        self.reset_alpha()


if __name__ == '__main__':

    wandb.init(project="StyleGAN1", entity="donghwankim")

    wandb.run.name = f'lambda_gp:{args["LAMBDA_GP"]}/z_dim:{args["Z_DIM"]}/w_dim:{args["W_DIM"]}'
    wandb.save()

    wandb.config.update(args)

    gen = Generator(args["Z_DIM"], args["W_DIM"], const_channels=512)
    disc = Discriminator()

    trainer = Trainer(gen, disc)

    for step in range(8):
        loader, _ = get_loader(
            args["IMAGE_SIZES"][step],
            dataset_root="/home/kdhsimplepro/kdhsimplepro/AI/ffhq/",
            batch_size=args["BATCH_SIZES"][step]
        )
        trainer.run(step=step, epochs=args["EPOCHS"][step], loader=loader)