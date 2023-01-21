import torch
from torch.optim import Adam

from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from tqdm import tqdm

from models import Generator, Discriminator

from utils import get_loader


Z_DIM = 512
W_DIM = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMBDA_GP = 10
IMAGE_SIZE = [4, 8, 16, 32, 64, 128, 256, 512]
BATCH_SIZE = [128, 64, 32, 32, 16, 16, 8, 4, 4]
EPOCHS = [20, 30, 40, 50, 60, 70, 80, 100]


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
    ):

        self.gen = gen.to(DEVICE)
        self.disc = disc.to(DEVICE)

        self.opt_gen = Adam([
            {"params": [param for name, param in self.gen.named_parameters() if 'mapping_network' not in name]},
            {"params": self.gen.mapping_network.parameters(), 'lr': 1e-5}
        ], lr=1e-3, betas=(0.0, 0.99))
    
        self.opt_disc = Adam(self.disc.parameters(), lr=1e-3, betas=(0.0, 0.99))

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
            
            z = torch.randn(real.size(0), Z_DIM).to(DEVICE)
            fake = self.gen(z, self.alpha, self.step)

            disc_real = self.disc(real, self.alpha, self.step)
            disc_fake = self.disc(fake.detach(), self.alpha, self.step)
            gp = get_gradient_penalty(self.disc, real, fake, self.alpha, self.step, DEVICE)
            disc_loss = (
                -(torch.mean(disc_real) - torch.mean(disc_fake))
                + LAMBDA_GP * gp
                + (0.001) * torch.mean(disc_real ** 2)
            )

            self.opt_disc.zero_grad()
            disc_loss.backward()
            self.opt_disc.step()

            gen_fake = disc(fake, self.alpha, self.step)
            gen_loss = -torch.mean(gen_fake)

            self.opt_gen.zero_grad()
            gen_loss.backward()
            self.opt_gen.step()

            self.alpha += 1 / (epochs * len(loader) * 0.5)
            self.alpha = min(self.alpha, 1)

            loop.set_postfix(
                gp=gp.item(),
                gen_loss=gen_loss.item(),
                disc_loss=disc_loss.item()
            )


    def run(self, step, epochs, loader):
        print(f"\n\nImage Size: {IMAGE_SIZE[step]}x{IMAGE_SIZE[step]}\n")
        self.step = step

        for epoch in range(epochs):
            print(f"EPOCH: {epoch+1}/{epochs}")
            self.train_fn(epochs, loader)
            test_image = self.test_fn()
            test_image.save(f"test_images/{step}/{epoch}.jpg")

            self.reset_alpha()



if __name__ == '__main__':
    gen = Generator(Z_DIM, W_DIM, const_channels=512)
    disc = Discriminator()

    trainer = Trainer(gen, disc)

    for step in range(1):
        loader, _ = get_loader(
            IMAGE_SIZE[step],
            dataset_root="/home/kdhsimplepro/kdhsimplepro/AI/ffhq/",
            batch_size=BATCH_SIZE[step]
        )
        trainer.run(step=step, epochs=EPOCHS[step], loader=loader)