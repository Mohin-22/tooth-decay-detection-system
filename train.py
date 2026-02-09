from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import torch
import torch.optim as optim
from unet_model import UNet

class DentalDecayDataset(Dataset):
    def __init__(self, img_dir, mask_txt_dir, transform=None, size=(256,256)):
        self.img_dir = img_dir
        self.mask_txt_dir = mask_txt_dir
        self.img_list = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_filename = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        txt_mask_filename = img_filename.rsplit('.', 1)[0] + '.txt'
        txt_mask_path = os.path.join(self.mask_txt_dir, txt_mask_filename)

        # Load and resize image
        image = Image.open(img_path).convert('L').resize(self.size)
        image_np = np.array(image, dtype=np.float32) / 255.0

        # Load and resize mask, then normalize
        # Load and resize mask, then normalize
        mask_np = np.zeros(self.size, dtype=np.float32)
        if os.path.exists(txt_mask_path):
            mask_lines = []
            with open(txt_mask_path, 'r') as f:
                for line in f:
                    split_line = line.strip().split()
                    # Make sure we skip empty lines
                    if split_line:
                        mask_lines.append(list(map(float, split_line)))
            if mask_lines:  # Only if non-empty!
                mask_np_from_file = np.array(mask_lines, dtype=np.float32)
                if mask_np_from_file.max() > 1.0:
                    mask_np_from_file = mask_np_from_file / 255.0
                mask_np_img = Image.fromarray(mask_np_from_file).resize(self.size)
                mask_np = np.array(mask_np_img, dtype=np.float32)
        mask_np = np.clip(mask_np, 0, 1)


        image_tensor = torch.tensor(image_np).unsqueeze(0)
        mask_tensor = torch.tensor(mask_np).unsqueeze(0)
        return image_tensor, mask_tensor

# -- Training loop --

img_dir = "Dataset/images"
mask_dir = "Dataset/masks"
dataset = DentalDecayDataset(img_dir, mask_dir)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device("cpu")
model = UNet(in_channels=1, out_channels=1).to(device)
criterion = torch.nn.BCELoss() # For binary masks
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (images, masks) in enumerate(loader):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = model(images)
        # BCELoss expects outputs AND masks in [0,1]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader):.4f}")

torch.save(model.state_dict(), "unet_dental.pth")
print("Trained model saved!")
