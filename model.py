from torch import nn


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True),
              nn.Dropout2d(p=0.1)]

    if pool: layers.append(nn.MaxPool2d((2, 2), (2, 2)))
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1_1 = conv_block(3, 64)
        self.conv1_2 = conv_block(64, 64)
        self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))

        self.conv2_1 = conv_block(64, 128, )
        self.conv2_2 = conv_block(128, 128, pool=True)
        self.res2 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3_1 = conv_block(128, 256, )
        self.conv3_2 = conv_block(256, 256, pool=True)
        self.res3 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))

        self.conv4_1 = conv_block(256, 512, )
        self.conv4_2 = conv_block(512, 512, pool=True)
        self.res4 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(1),
            nn.Linear(32768, num_classes)
        )
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        x = self.conv1_2(self.conv1_1(x))
        x = self.res1(x) + x
        x = self.conv2_2(self.conv2_1(x))
        x = self.res2(x) + x
        x = self.conv3_2(self.conv3_1(x))
        x = self.res3(x) + x
        x = self.conv4_2(self.conv4_1(x))
        x = self.res4(x) + x
        return self.classifier(x)

if __name__ == '__main__':
    import torch
    image = torch.zeros((4, 3, 64, 64), dtype=torch.float) #
    model = ResNet()
    y = model(image)
    print(y.shape)