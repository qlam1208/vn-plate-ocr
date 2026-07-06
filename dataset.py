import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import glob

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def parse_yolo_label(txt_path):
    if not os.path.exists(txt_path): return False, "", ""
    with open(txt_path, "r") as f: boxes = [line.strip().split() for line in f.readlines()]
    boxes = [(int(b[0]), float(b[1]), float(b[2])) for b in boxes if len(b) >= 5]
    if not boxes: return False, "", ""

    y_centers = [b[2] for b in boxes]
    is_two_lines = (max(y_centers) - min(y_centers)) > 0.2

    if is_two_lines:
        mid_y = sum(y_centers) / len(y_centers)
        top_line = sorted([b for b in boxes if b[2] < mid_y], key=lambda x: x[1])
        bottom_line = sorted([b for b in boxes if b[2] >= mid_y], key=lambda x: x[1])
        label_top = "".join([CHARS[b[0]] for b in top_line if b[0] < len(CHARS)])
        label_bottom = "".join([CHARS[b[0]] for b in bottom_line if b[0] < len(CHARS)])
        return True, label_top, label_bottom
    else:
        label = "".join([CHARS[b[0]] for b in sorted(boxes, key=lambda x: x[1]) if b[0] < len(CHARS)])
        return False, label, ""

class CRNNDataset(Dataset):
    def __init__(self, data_dir, img_width=128, img_height=32):
        self.data_dir = data_dir
        self.img_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
        self.transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.valid_data = []
        for img_path in self.img_paths:
            is_two_lines, l1, l2 = parse_yolo_label(img_path.replace(".jpg", ".txt"))
            if is_two_lines:
                if l1: self.valid_data.append((img_path, l1, "top"))
                if l2: self.valid_data.append((img_path, l2, "bottom"))
            elif l1:
                self.valid_data.append((img_path, l1, "full"))
        print(f"Đã nạp {len(self.valid_data)} dòng ảnh hợp lệ.")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        img_path, label, crop_type = self.valid_data[idx]
        image = Image.open(img_path).convert('RGB')
        w, h = image.size

        if crop_type == "top":
            image = image.crop((0, 0, w, h // 2))
        elif crop_type == "bottom":
            image = image.crop((0, h // 2, w, h))

        return self.transform(image), label

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    
    # Encode labels
    targets = []
    target_lengths = []
    for label in labels:
        target = [CHARS.find(c) + 1 for c in label] 
        targets.extend(target)
        target_lengths.append(len(target))
        
    targets = torch.tensor(targets, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    input_lengths = torch.tensor([images.size(3) // 4] * len(images), dtype=torch.long)
    
    return images, targets, input_lengths, target_lengths

if __name__ == "__main__":
    dataset = CRNNDataset("yolo_plate_ocr_dataset")
    if len(dataset) > 0:
        img, label = dataset[0]
        print(f"Sample 0: Image size: {img.shape}, Label: {label}")
