import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms 
from torchvision.utils import save_image
from dataset import ImgDataset
from generator import Generator
from discriminator import Discriminator

# hyperparameters

device = "cuda" if torch.cuda.is_available() else "cpu"

data_transform = transforms.Compose([
    transforms.Resize((256 , 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.ToTensor()
])

batch_size = 1

lrG = 0.0002
lrD = 0.0002
epochs = 10

# as i don't have gpu :(
num_workers = 0

training_directory = ""
validing_directory = ""

LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10

LOAD_MODEL = True
SAVE_MODEL = True

#CHECKPOINT_GEN_H = "genh.pth.tar"
#CHECKPOINT_GEN_Z = "genz.pth.tar"
#CHECKPOINT_CRITIC_H = "critich.pth.tar"
#CHECKPOINT_CRITIC_Z = "criticz.pth.tar"



def train_cyclGan(Dx, Dy, F, G, loader, disc_optimizer, gen_optimizer, l1, mse, d_scaler, g_scaler):

    B_reals = 0
    B_fakes = 0

    loop = tqdm(loader, leave=True)
    # loop over the training data
    for idx, (A_img, B_img) in enumerate(loop):
        A_img = A_img.to(device)
        B_img = B_img.to(device)

        # Train Discriminators Dx and Dy
        with torch.cuda.amp.autocast():
            # Dx on the first img A
            fake_B = G(A_img)
            disc_real_B = Dx(B_img)
            disc_fake_B = Dx(fake_B.detach())
            B_reals += disc_real_B.mean().item()
            B_fakes += disc_fake_B.mean().item()
            disc_B_real_loss = mse(disc_real_B, torch.ones_like(disc_real_B))
            disc_B_fake_loss = mse(disc_fake_B, torch.zeros_like(disc_fake_B))
            disc_B_loss = disc_B_real_loss + disc_B_fake_loss

            # Dy on the second img B
            fake_A  = F(B_img)
            disc_real_A = Dy(A_img)
            disc_fake_A = Dy(fake_A.detach())
            disc_A_real_loss = mse(disc_real_A, torch.ones_like(disc_real_A))
            disc_A_fake_loss = mse(disc_fake_A, torch.zeros_like(disc_fake_A))
            disc_A_loss = disc_A_real_loss + disc_A_fake_loss

            # all discriminator losses
            disc_loss = (disc_B_loss + disc_A_loss)/2

        disc_optimizer.zero_grad()
        d_scaler.scale(disc_loss).backward()
        d_scaler.step(disc_optimizer)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            disc_B_fake = Dx(fake_B)
            disc_A_fake = Dy(fake_A)
            gen_B_loss = mse(disc_B_fake, torch.ones_like(disc_B_fake))
            gen_A_loss = mse(disc_A_fake, torch.ones_like(disc_A_fake))

            # cycle loss
            cycle_A = F(fake_B)
            cycle_B = G(fake_A)
            cycle_A_loss = l1(A_img, cycle_A)
            cycle_B_loss = l1(B_img, cycle_B)

            # identity loss
            identity_A = F(A_img)
            identity_B = G(B_img)
            identity_A_loss = l1(A_img, identity_A)
            identity_B_loss = l1(B_img, identity_B)

            # final loss function
            G_loss = (
                gen_A_loss
                + gen_B_loss
                + cycle_A_loss * LAMBDA_CYCLE
                + cycle_B_loss * LAMBDA_CYCLE
                + identity_A_loss * LAMBDA_IDENTITY
                + identity_B_loss * LAMBDA_IDENTITY
            )

        gen_optimizer.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(gen_optimizer)
        g_scaler.update()

        #if idx % 200 == 0:
        #    save_image(fake_B*0.5+0.5, f"saved_images/horse_{idx}.png")
        #    save_image(fake_A*0.5+0.5, f"saved_images/zebra_{idx}.png")

        loop.set_postfix(B_real=B_reals/(idx+1), B_fake=B_fakes/(idx+1))



def main():

    # create the first img generator and discriminator
    D_A = Discriminator(img=3).to(device)
    G_A = Generator(img=3, residuals=9).to(device)

    # create the second img generator and discriminator
    D_B = Discriminator(img=3).to(device)
    G_B = Generator(img=3, residuals=9).to(device)

    # define each model's optimizer
    Discriminator_optimizer = optim.Adam(
        list(D_A.parameters()) + list(D_B.parameters()),
        lr=lrD,
        betas=(0.5, 0.999),
    )

    Generator_optimizer = optim.Adam(
        list(G_A.parameters()) + list(G_A.parameters()),
        lr=lrG,
        betas=(0.5, 0.999),
    )

    # Cycle Losses
    L1 = nn.L1Loss()
    # GAN losses
    mse = nn.MSELoss()

    # create training data loader to iterate over the training data
    train_dataset = ImgDataset( root_horse=training_directory+"/img1", root_zebra=training_directory+"/img2", transform=data_transform )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # create validation data loader to validate the model on unseen data
    val_dataset = ImgDataset( root_horse=validing_directory+"/img1", root_zebra=validing_directory+"/img2", transform=data_transform )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # train for specified number of epochs
    for epoch in range(epochs):

        # feed the model the crated objects to use them for training
        train_cyclGan(D_A, D_B, G_A, G_B, train_loader, Discriminator_optimizer, Generator_optimizer, L1, mse, d_scaler, g_scaler)

        # save the model weights
        #save_checkpoint(G_A, Generator_optimizer, filename=config.CHECKPOINT_GEN_H)
        #save_checkpoint(G_B, Generator_optimizer, filename=config.CHECKPOINT_GEN_Z)
        #save_checkpoint(D_A, Discriminator_optimizer, filename=config.CHECKPOINT_CRITIC_H)
        #save_checkpoint(D_B, Discriminator_optimizer, filename=config.CHECKPOINT_CRITIC_Z)

if __name__ == "__main__":
    main()