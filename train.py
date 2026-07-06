import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# Import từ các file của mình
from model import CRNN
from dataset import CRNNDataset, collate_fn, CHARS

# Hyperparameters
DATA_DIR = "yolo_plate_ocr_dataset"
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 0.001
NUM_CLASSES = len(CHARS) + 1  

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dataset = CRNNDataset(DATA_DIR)
    
    if len(dataset) == 0:
        print("Error: No valid images found for training!")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    # Khoi tao model
    model = CRNN(img_channels=1, num_classes=NUM_CLASSES, hidden_size=256).to(device)
    
    # Khoi tao Loss & Optim
    # blank=0 vi class 0 trong lúc encode label danh cho blank
    criterion = nn.CTCLoss(blank=0, zero_infinity=True) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        loop = tqdm(dataloader, leave=True)
        for batch_idx, (images, targets, input_lengths, target_lengths) in enumerate(loop):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # (Batch, 1, 32, 128) -> Model -> (SequenceLength, Batch, NumClasses)
            outputs = model(images)
            
            # Loss CTC
            loss = criterion(outputs, targets, input_lengths, target_lengths)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # show
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())
            
        print(f"==> Epoch {epoch+1} completed. Avg Loss: {total_loss / len(dataloader):.4f}")
        
        # Save model checkpoint
        torch.save(model.state_dict(), f"crnn_epoch_{epoch+1}.pth")
        
    print("Training finished!")

if __name__ == "__main__":
    train()
