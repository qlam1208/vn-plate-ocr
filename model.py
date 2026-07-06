import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_channels=1, num_classes=37, hidden_size=256):
        super(CRNN, self).__init__()
        
        # 1. CNN Backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # H=16
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # H=8
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            # ep chieu cao, giu chieu rong
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)), # H=4
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)), # H=2
            
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        
        # RNN
        self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, num_layers=2, batch_first=True)
        
        # Fully Connected
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: (Batch, Channel, Height, Width) -> (B, 1, 32, 128)
        conv = self.cnn(x) 
        
        # convert tensor ve dang (Batch, Width, Channels) de cho vao RNN
        b, c, h, w = conv.size()

        assert h == 1, "Chiều cao của Feature map phải bằng 1, hãy kiểm tra lại code CNN."
        
        # bo h = 1, swap Width, Channels
        conv = conv.squeeze(2) # Output: (B, C, W)
        conv = conv.permute(0, 2, 1) # Output: (B, W, C)
        
        # RNN
        rnn_out, _ = self.rnn(conv) # rnn_out: (B, W, hidden_size * 2)

        output = self.fc(rnn_out) # (B, W, num_classes)
        
        # CTC Loss trong PyTorch can input (Width, Batch, num_classes)
        output = output.permute(1, 0, 2)
        
        return output

# Test
if __name__ == "__main__":
    model = CRNN(img_channels=1, num_classes=37, hidden_size=256)

    dummy_input = torch.randn(16, 1, 32, 128) 
    out = model(dummy_input)
    
    print(f"Kích thước Đầu ra của Model: {out.shape}")
    print("Ghi chú: (Khung_thời_gian=33, Batch=16, Số_lớp_ký_tự=37)")
    print("==> Mô hình CRNN đã khởi tạo thành công!")
