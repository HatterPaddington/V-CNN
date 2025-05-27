import torch
import torch.nn as nn

class net_model(torch.nn.Module):

    def __init__(self):
        super(net_model, self).__init__()

        self.layer_01 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                1, 32, (3, 3), stride=(1, 1), padding=0
            ),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=0),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU()
        )

        self.layer_01_maxpool = torch.nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2)
        )

        self.layer_02 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding=0),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU()
        )

        self.layer_02_maxpool = torch.nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2)
        )

        self.layer_03 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=0),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU()
        )

        self.layer_03_maxpool = torch.nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2)
        )

        self.layer_04 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=0),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU()
        )

        self.layer_04_maxpool = torch.nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2)
        )

        self.layer_05 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, (3, 3), stride=(1, 1), padding=0),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=2),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU()
        )

        self.layer_part1 = torch.nn.Sequential(
            self.layer_01, self.layer_01_maxpool,
            self.layer_02, self.layer_02_maxpool,
            self.layer_03, self.layer_03_maxpool,
            self.layer_04, self.layer_04_maxpool, self.layer_05
        )

        # -------------------------------------------------------

        # layer_06

        self.layer_06_01 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                512, 256, (3, 3), stride=(2, 2), padding=1,
                output_padding=1
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU()
        )

        self.layer_06_02 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, (3, 3), stride=(1, 1), padding=0),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU()
        )

        self.layer_06_03 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU()
        )

        # layer_07

        self.layer_07_01 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                256, 128, (3, 3), stride=(2, 2), padding=1,
                output_padding=1
            ),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU()
        )

        self.layer_07_02 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, (3, 3), stride=(1, 1), padding=0),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU()
        )

        self.layer_07_03 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU()
        )

        # layer_08

        self.layer_08_01 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                128, 64, (3, 3), stride=(2, 2), padding=1,
                output_padding=1
            ),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU()
        )

        self.layer_08_02 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, (3, 3), stride=(1, 1), padding=0),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU()
        )

        self.layer_08_03 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU()
        )

        # layer_09

        self.layer_09_01 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                64, 32, (3, 3), stride=(2, 2), padding=1,
                output_padding=1
            ),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU()
        )

        self.layer_09_02 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, (3, 3), stride=(1, 1), padding=0),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU()
        )

        self.layer_09_03 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU()
        )

        # layer_10

        self.layer_10 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 1, (3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(1),
            torch.nn.LeakyReLU()
        )

    def forward(self, x):
        # -------------------------------------------------------
        x6_0 = self.layer_06_01(self.layer_part1(x))
        x6_1 = self.layer_04(
            self.layer_03_maxpool(self.layer_03(self.layer_02_maxpool(
                self.layer_02(self.layer_01_maxpool(self.layer_01(x)))
            )))
        )
        x6_2 = torch.cat((x6_0, x6_1), 1)
        x6 = self.layer_06_03(self.layer_06_02(x6_2))

        # -------------------------------------------------------
        x7_0 = self.layer_07_01(x6)
        x7_1 = self.layer_03(self.layer_02_maxpool(
            self.layer_02(self.layer_01_maxpool(self.layer_01(x)))
        ))
        x7_2 = torch.cat((x7_0, x7_1), 1)
        x7 = self.layer_07_03(self.layer_07_02(x7_2))

        # -------------------------------------------------------
        x8_0 = self.layer_08_01(x7)
        x8_1 = self.layer_02(self.layer_01_maxpool(self.layer_01(x)))
        x8_2 = torch.cat((x8_0, x8_1), 1)
        x8 = self.layer_08_03(self.layer_08_02(x8_2))

        # -------------------------------------------------------
        x9_0 = self.layer_09_01(x8)
        x9_1 = self.layer_01(x)
        x9_2 = torch.cat((x9_0, x9_1), 1)
        x9 = self.layer_09_03(self.layer_09_02(x9_2))

        # -------------------------------------------------------
        x10 = self.layer_10(x9)

        return x10