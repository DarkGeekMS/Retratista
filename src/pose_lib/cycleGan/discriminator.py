import torch
import torch.nn as nn

"""

we used 9 residual blocks assuming the image dimensions is 256*256 or higher

"""
class Discriminator(nn.Module):
    """
        The Discriminator 
    """
    def __init__(self, img, k = 64):

        super().__init__()

        # -------------------------- c-k=64 -------------------------------------

        self.c64 = nn.Sequential(
            # 4*4 with stride 2 and 64 filters convolution layer followed by Leaky relu 
            nn.Conv2d(img, k, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(negative_slope = 0.02, inplace = True)
        )

        # ---------------------------- c-k=128 ----------------------------------------
        self.c128 = nn.Sequential(

            # 4*4 with stride 2 and 128 filter conv. layer followed by instance norm and relu
            nn.Conv2d(k,2*k, kernel_size=4, stride=1, padding_mode="reflect", bias=True ),
            nn.InstanceNorm2d(2*k),
            nn.LeakyReLU(negative_slope = 0.02, inplace = True)
        )

        # ----------------------------------- c-k=256 --------------------------------------
        self.c256 = nn.Sequential(

            # 4*4 with stride 2 and 128 filter conv. layer followed by instance norm and relu
            nn.Conv2d(2*k,4*k, kernel_size=4, stride=1, padding_mode="reflect", bias=True ),
            nn.InstanceNorm2d(4*k),
            nn.LeakyReLU(negative_slope = 0.02, inplace = True)
        )

        # ---------------------------------- c-k=512 ----------------------------------------
        self.c512 = nn.Sequential(

            # 4*4 with stride 2 and 128 filter conv. layer followed by instance norm and relu
            nn.Conv2d(4*k,8*k, kernel_size=4, stride=2, padding_mode="reflect", bias=True  ),
            nn.InstanceNorm2d(8*k),
            nn.LeakyReLU(negative_slope = 0.02, inplace = True)
        )

        self.lastL = nn.Conv2d(8*k, 1, kernel_size=4, stride=1, padding=1, padding_mode= "reflect")

    def forward(self, x):
        # apply the network layers
        x = self.c64(x)
        x = self.c128(x)
        x = self.c256(x)
        x = self.c512(x)
        return torch.sigmoid(self.lastL(x))

def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    disc = Discriminator(img_channels)
    print(disc(x).shape)

if __name__ == "__main__":
    test()