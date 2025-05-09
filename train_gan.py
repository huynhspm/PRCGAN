import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import MnistDataset
from gan import CGAN
import torchvision.utils as vutils
from torchmetrics.functional import structural_similarity_index_measure as ssim

# Hyperparameters
batch_size = 64
lr = 0.0002
img_dims = [1, 28, 28]
epochs = 300
dataset = "fashion"

# Load Dataset
train_dataset = MnistDataset(data_dir='data', mode='train', dataset=dataset)
test_dataset = MnistDataset(data_dir='data', mode='test', dataset=dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Initialize CGAN
cgan = CGAN(img_dims=img_dims, cond_dims=[1, 28, 28])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cgan.to(device)

log_dir = f"results/{dataset}"
os.makedirs(log_dir, exist_ok=True)

# Function to Evaluate
def evaluate_metrics(cgan, dataloader, device, save_path=None):
    total_ssim = 0
    total_mse = 0
    total_mae = 0

    with torch.no_grad():
        for idx, (real_images, magnitude, _) in enumerate(dataloader):
            real_images, magnitude = real_images.to(device), magnitude.to(device)
            fake_images = cgan.sample(num_sample=real_images.size(0), cond=magnitude, device=device).detach()

            # Compute SSIM
            batch_ssim = ssim(fake_images, real_images, data_range=1.0)
            total_ssim += batch_ssim.item()

            # Compute MSE
            batch_mse = F.mse_loss(fake_images, real_images, reduction='mean')
            total_mse += batch_mse.item()

            # Compute MAE
            batch_mae = F.l1_loss(fake_images, real_images, reduction='mean')
            total_mae += batch_mae.item()

            if save_path and idx == 0:
                grid = vutils.make_grid(fake_images, nrow=8, normalize=True, scale_each=True)
                vutils.save_image(grid, save_path)

    # Return average metrics
    return total_ssim / len(dataloader), total_mse / len(dataloader), total_mae / len(dataloader)


# Loss and Optimizers
criterion_adv = nn.BCELoss()
criterion_rec = torch.nn.L1Loss()
lambda_rec = 100.0
optimizer_g = optim.Adam(cgan.gen.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(cgan.disc.parameters(), lr=lr, betas=(0.5, 0.999))

# Training Loop
best_ssim = 0
log_file = open(f"{log_dir}/log.txt", "w")
for epoch in range(epochs):
    epoch_loss_d = 0
    epoch_loss_g = 0
    for real_images, magnitude, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True):
        real_images, magnitude = real_images.to(device), magnitude.to(device)
        
        # Train Discriminator
        fake_images = cgan.sample(num_sample=real_images.shape[0], cond=magnitude, device=device)
        real_labels = torch.ones(real_images.shape[0], 1, device=device, dtype=torch.float)
        fake_labels = torch.zeros(real_images.shape[0], 1, device=device, dtype=torch.float)

        real_output = cgan.classify(real_images, magnitude)
        fake_output = cgan.classify(fake_images.detach(), magnitude)

        loss_d_real = criterion_adv(real_output, real_labels)
        loss_d_fake = criterion_adv(fake_output, fake_labels)
        loss_d = loss_d_real + loss_d_fake

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        # === Train Generator ===
        fake_output = cgan.classify(fake_images, magnitude)
        loss_adv = criterion_adv(fake_output, real_labels)
        loss_rec = criterion_rec(fake_images, real_images)
        loss_g = loss_adv + lambda_rec * loss_rec

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        epoch_loss_d += loss_d.item()
        epoch_loss_g += loss_g.item()

    log_file.write(f"Epoch [{epoch+1}/{epochs}] Loss D: {epoch_loss_d/len(train_loader):.4f}, Loss G: {epoch_loss_g/len(train_loader):.4f}\n")
    log_file.flush()

    if (epoch + 1) % 20 == 0:
        cgan.eval()
        train_ssim, train_mse, train_mae = evaluate_metrics(cgan, train_loader, device, save_path=f"{log_dir}/epoch_{epoch+1}_train.png")
        test_ssim, test_mse, test_mae = evaluate_metrics(cgan, test_loader, device, save_path=f"{log_dir}/epoch_{epoch+1}_test.png")
        log_file.write(f"Epoch [{epoch+1}] Train SSIM: {train_ssim:.4f}, Train MSE: {train_mse:.4f}, Train MAE: {train_mae:.4f}\n")
        log_file.write(f"Epoch [{epoch+1}] Test SSIM: {test_ssim:.4f}, Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}\n")
        log_file.flush()

        if test_ssim > best_ssim:
            best_ssim = test_ssim
            torch.save(cgan.gen.state_dict(), f"{log_dir}/best_generator.pth")
            torch.save(cgan.disc.state_dict(), f"{log_dir}/best_discriminator.pth")
        
        cgan.train()

log_file.close()

# Save Models
torch.save(cgan.gen.state_dict(), f"{log_dir}/generator.pth")
torch.save(cgan.disc.state_dict(), f"{log_dir}/discriminator.pth")