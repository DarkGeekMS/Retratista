import torch
import torch.nn as nn

"""

we used 9 residual blocks assuming the image dimensions is 256*256 or higher

"""
class Generator(nn.Module):
    """
        The generator 
    """
    def __init__(self, img, k = 64, residuals= 9):

        self.residuals = residuals
        super().__init__()

        # -------------------------- c7s1-k -------------------------------------

        self.c7s1kF = nn.Sequential(
            # 7*7 with stride 1 and 64 filters convolution layer followed by instance norm and relu 
            nn.Conv2d(img, k, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(k),
            nn.ReLU(inplace=True)
        )

        # ---------------------------- D-k ----------------------------------------
        self.DK = nn.Sequential(

            # 3*3 with stride 2 and 128 filter conv. layer followed by instance norm and relu
            nn.Conv2d(k,2*k, kernel_size=3, stride=2, padding=1, padding_mode="reflect" ),
            nn.InstanceNorm2d(2*k),
            nn.ReLU(inplace=True),

            # 3*3 with stride 3 and 256 filter conv. layer followed by instance norm and relu
            nn.Conv2d(2*k, 4*k, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(4*k),
            nn.ReLU(inplace=True)
        )

        # ----------------------------------- R-k --------------------------------------
        self.RK = nn.Sequential(

            # residual block consists of two 3*3 with stride 2 and 256 filter conv.layers
            nn.Conv2d(4*k,4*k, kernel_size=3, stride=1, padding=1, padding_mode="reflect" ),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*k,4*k, kernel_size=3, stride=1, padding=1, padding_mode="reflect" )
        )

        # ---------------------------------- U-k ----------------------------------------
        self.UK = nn.Sequential(

            # 3*3 with stride 2 and 128 filter fractonal-strided-conv. layer followed by instance norm and relu
            nn.ConvTranspose2d(4*k,2*k, kernel_size=3, stride=2, padding=1, output_padding=1 ),
            nn.InstanceNorm2d(2*k),
            nn.ReLU(inplace=True),

            # 3*3 with stride 3 and 64 filter fractonal-strided-conv. layer followed by instance norm and relu
            nn.ConvTranspose2d(2*k, 1*k, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(1*k),
            nn.ReLU(inplace=True),
        )
        # --------------------------------- c7s1-k -------------------------------------
        self.c7s1KL = nn.Conv2d(k*1, img, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        # apply the first conv2d
        x = self.c7s1kF(x)
        # apply the down blocks
        x = self.DK(x)
        # apply 9 residual blocks
        for i in range (self.residuals):
            x = self.RK(x)
        # apply the up blocks
        x = self.UK(x)
        # apply the final conv2d
        return torch.tanh(self.c7s1KL(x))

def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    print(gen(x).shape)

if __name__ == "__main__":
    test()