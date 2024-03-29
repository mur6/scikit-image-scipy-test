import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        hidden_dim = 1024
        self.fc = nn.Sequential(
            nn.Linear(49*32*32*2, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, masks):
        masks = torch.unsqueeze(masks, 1)
        #print(masks.shape)
        out = self.layer1(masks.float())
        #print("1:", out.shape)
        out = self.layer2(out)
        #print("2:", out.shape)
        out = self.layer3(out)
        #print("3:", out.shape)
        out = self.layer4(out)
        #print("4:", out.shape)
        out = out.contiguous().view(masks.size(0), -1)
        #print("5:", out.shape)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    #from torchsummary import summary
    model = Net()
    # print(HomographyNet())
    #summary(model, (3, 128, 128))
    out = model(torch.rand(1, 3, 128, 128))
    print(out)
