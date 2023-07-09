import torch
import torch.nn as nn
import torch.nn.functional as F


# PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
# Layer1 -
# X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
# R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
# Add(X, R1)
# Layer 2 -
# Conv 3x3 [256k]
# MaxPooling2D
# BN
# ReLU
# Layer 3 -
# X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
# R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
# Add(X, R2)
# MaxPooling with Kernel Size 4
# FC Layer 
# SoftMax

def getNormalisationLayer(normalisation_method, output_channel, groups=0):
      if normalisation_method == 'bn':
          return nn.BatchNorm2d(output_channel)
      elif normalisation_method == 'gn':
          return nn.GroupNorm(groups, output_channel)
      elif normalisation_method == 'ln':
          return nn.GroupNorm(1, output_channel)


class Model(nn.Module):
    def __init__(self, normalisation_method, groups=0):
        super(NormalisationModel, self).__init__()
        # PrepLayer
        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            getNormalisationLayer(normalisation_method, 64, groups),
            nn.ReLU(),
        ) # output_size = 26

        # Layer1 
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2,2),
            getNormalisationLayer(normalisation_method, 128, groups),
            nn.ReLU(),
        ) # output_size = 24
        # res block 
        # what should be the output for 

        # ResBlock todo: make sure to add as different class as specified in description
        self.res_block1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            getNormalisationLayer(normalisation_method, 128, groups),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            getNormalisationLayer(normalisation_method, 128, groups),
            nn.ReLU(),            
        )

        # Layer2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2,2),
            getNormalisationLayer(normalisation_method, 256, groups),
            nn.ReLU(),
        ) # output_size = 26

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2,2),
            getNormalisationLayer(normalisation_method, 256, groups),
            nn.ReLU(),
        ) # output_size = 26

        # ResBlock2 todo: make sure to add as different class as specified in description
        self.res_block2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            getNormalisationLayer(normalisation_method, 512, groups),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            getNormalisationLayer(normalisation_method, 512, groups),
            nn.ReLU(),            
        ) # output_size = 8

        self.maxPool2 = nn.MaxPool2d(4, 4) # output_size = 12

        self.output_linear = nn.Linear(512, 10, bias=False)

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.convblock1(x)
        x = x + self.res_block1(x)
        x = self.layer2(x)
        x = self.convblock2(x)
        x = x + self.res_block2(x)
        x = self.maxPool2(x)
        x = self.output_linear(x)

        return F.log_softmax(x, dim=-1)


