import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( # input [3, 128, 128]
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=1), # [16, 128, 128]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # [16, 32, 32]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3, stride=2, padding=1),  # [36, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # [36, 8, 8]
        )

        self.linear = nn.Linear(36 * 64, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 36 * 64)
        output = self.linear(x)
        return output
