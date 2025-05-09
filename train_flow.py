import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import MnistDataset
from flow import CondFlow
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


# Initialize Flow Model
flow = CondFlow(img_dims=img_dims)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flow.to(device)

log_dir = f"results/{dataset}_flow"
os.makedirs(log_dir, exist_ok=True)

def evaluate_metrics(flow, dataloader, device, n_steps=100, save_path=None):
    total_ssim = 0
    total_mse = 0
    total_mae = 0

    time_steps = torch.linspace(0, 1.0, n_steps + 1).to(device)
    with torch.no_grad():
        for idx, (real_images, magnitude, _) in enumerate(dataloader):
            real_images, cond = real_images.to(device), magnitude.to(device)
            cond = cond * 2 - 1

            # Initialize data
            x = torch.randn_like(real_images).to(device)

            # Perform flow matching
            for i in range(n_steps):
                x = flow.step(x, time_steps[i].view(-1, 1), time_steps[i + 1].view(-1, 1), cond)

            fake_images = (x.detach() + 1) / 2

            # Compute SSIM
            batch_ssim = ssim(fake_images, real_images, data_range=1.0)
            total_ssim += batch_ssim.item()

            # Compute MSE
            batch_mse = F.mse_loss(fake_images, real_images, reduction='mean')
            total_mse += batch_mse.item()

            # Compute MAE
            batch_mae = F.l1_loss(fake_images, real_images, reduction='mean')
            total_mae += batch_mae.item()

            # Save example results
            if save_path and idx == 0:
                grid = vutils.make_grid(fake_images, nrow=8, normalize=True, scale_each=True)
                vutils.save_image(grid, save_path)

    # Return average metrics
    return total_ssim / len(dataloader), total_mse / len(dataloader), total_mae / len(dataloader)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(flow.parameters(), lr=lr)

# Training Loop
best_ssim = 0
log_file = open(f"{log_dir}/log.txt", "w")
for epoch in range(epochs):
    epoch_loss = 0
    for real_images, magnitude, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True):
        x_1, cond = real_images.to(device), magnitude.to(device)
        x_1, cond = x_1 * 2 - 1, cond * 2 - 1
        
        x_0 = torch.randn_like(x_1)
        t = torch.rand(len(x_1), 1, device=device)
        t_expand = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        dx_t = x_1 - x_0
        
        # Compute loss
        loss = criterion(flow(x_t, t, cond), dx_t)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    log_file.write(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss/len(train_loader):.4f}")
    log_file.flush()

    if (epoch + 1) % 20 == 0:
        flow.eval()
        train_ssim, train_mse, train_mae = evaluate_metrics(flow, train_loader, device, save_path=f"{log_dir}/epoch_{epoch+1}_train.png")
        test_ssim, test_mse, test_mae = evaluate_metrics(flow, test_loader, device, save_path=f"{log_dir}/epoch_{epoch+1}_test.png")
        log_file.write(f"Epoch [{epoch+1}] Train SSIM: {train_ssim:.4f}, Train MSE: {train_mse:.4f}, Train MAE: {train_mae:.4f}\n")
        log_file.write(f"Epoch [{epoch+1}] Test SSIM: {test_ssim:.4f}, Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}\n")
        log_file.flush()

        if test_ssim > best_ssim:
            best_ssim = test_ssim
            torch.save(flow.state_dict(), f"{log_dir}/flow.pth")
        
        flow.train()

log_file.close()

# Save Models
torch.save(flow.state_dict(), f"{log_dir}/flow.pth")