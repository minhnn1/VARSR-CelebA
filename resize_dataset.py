import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Paths
hr_dir = "dataset/testset/HR"
lr_dir = "dataset/testset/LR"
os.makedirs(lr_dir, exist_ok=True)

# Transformation: 256x256 → 64x64
resize_to_lr = transforms.Compose([
    transforms.Resize((64, 64), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.ToPILImage()
])

# Process each image in HR folder
for filename in tqdm(os.listdir(hr_dir), desc="Converting HR to LR"):
    hr_path = os.path.join(hr_dir, filename)
    lr_path = os.path.join(lr_dir, filename)

    try:
        image = Image.open(hr_path).convert("RGB")
        lr_image = resize_to_lr(image)
        lr_image.save(lr_path)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
