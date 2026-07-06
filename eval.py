import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
import shutil
import csv
import matplotlib.pyplot as plt
from model import CRNN
from dataset import CRNNDataset, collate_fn, CHARS

DATA_DIR = "yolo_plate_ocr_dataset"
MODEL_PATH = "crnn_epoch_60.pth"
BATCH_SIZE = 32
NUM_CLASSES = len(CHARS) + 1  

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def decode_predictions(predictions):
    decoded_texts = []
    for i in range(predictions.size(1)):
        pred = predictions[:, i]
        
        char_list = []
        for j in range(len(pred)):
            if pred[j] != 0 and (not (j > 0 and pred[j - 1] == pred[j])):
                char_list.append(CHARS[pred[j] - 1])
                
        decoded_texts.append("".join(char_list))
        
    return decoded_texts


def evaluate_model(model_path, dataloader, device):
    # Khoi tao model
    model = CRNN(img_channels=1, num_classes=NUM_CLASSES, hidden_size=256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    total_cer = 0.0
    correct_plates = 0
    total_plates = 0

    with torch.no_grad():
        for images, targets, input_lengths, target_lengths in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(2)
            pred_texts = decode_predictions(preds)

            target_texts = []
            start = 0
            for length in target_lengths:
                t = targets[start:start + length]
                target_texts.append("".join([CHARS[c - 1] for c in t]))
                start += length

            for pred_str, true_str in zip(pred_texts, target_texts):
                if len(true_str) == 0: continue
                total_plates += 1
                if pred_str == true_str:
                    correct_plates += 1
                dist = levenshtein_distance(pred_str, true_str)
                cer = dist / len(true_str)
                total_cer += cer
    accuracy = (correct_plates / total_plates) * 100 if total_plates > 0 else 0
    avg_cer = (total_cer / total_plates) * 100 if total_plates > 0 else 0
    return accuracy, avg_cer


def find_best_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dataset = CRNNDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    pth_files = glob.glob("crnn_epoch_*.pth")
    pth_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    pth_files = pth_files[-15:]
    best_acc = 0
    best_file = ""
    epochs = []
    accuracies = []
    cers = []

    with open("evaluation_results.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Accuracy (%)", "CER (%)"])
        for pth in pth_files:
            epoch_num = int(pth.split('_')[-1].split('.')[0])
            print(f"Đang chấm điểm {pth} ...")
            acc, cer = evaluate_model(pth, dataloader, device)
            print(f"-> Acc: {acc:.2f}% | CER: {cer:.2f}%\n")
            epochs.append(epoch_num)
            accuracies.append(acc)
            cers.append(cer)
            writer.writerow([epoch_num, acc, cer])
            if acc > best_acc:
                best_acc = acc
                best_file = pth
    print("=" * 40)
    print(f"MODEL ĐỈNH NHẤT LÀ: {best_file} (Acc = {best_acc:.2f}%)")
    print("=" * 40)
    if best_file:
        shutil.copy(best_file, "best_crnn.pth")
        print(f"Đã tự động nhân bản file 'best_crnn.pth' thành công!")
        print(f"Đã lưu kết quả chi tiết ra file 'evaluation_results.csv'")
    #Ve bieu do
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, accuracies, marker='o', color='b', label='Accuracy (%)')
        plt.plot(epochs, cers, marker='x', color='r', label='CER (%)')

        plt.title("Biểu đồ đánh giá mô hình CRNN")
        plt.xlabel("Epoch")
        plt.ylabel("Phần trăm (%)")
        plt.grid(True)
        plt.legend()

        plt.savefig("evaluation_chart.png")
        print("Đã xuất biểu đồ ra file 'evaluation_chart.png'")
    except Exception as e:
        print(f"Lỗi vẽ biểu đồ (có thể chưa cài matplotlib): {e}")

if __name__ == "__main__":
    find_best_model()