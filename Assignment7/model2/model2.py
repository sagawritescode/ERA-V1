import torch
import torch.nn as nn
import torch.nn.functional as F


# Target - to commit to a skeleton. 
# Intuition behind the model
    # After trying various architecture, I decided to commit to this architecture. 
    # Block 1 has 2 convolutional layers (instead of 1 that was in final model in class). Reason being I wanted Block 1 to have sufficient parameters to learn edges and gradients 
    # Transition block - helps reduce the numbers of parameters and implement squeeze 
    # Block 2 has 3 convolutional layers (instead of 4 in class final model). 3 layers seem sufficient enough to learn texture and patterns 
    # Global average pooling - I wanted this to be the part of the skeleton as I wanted to finalise the layers before proceeding. I tried experimenting with more/less and finalised the layers. Added GAP after finalising the layers i.e skeleton/architecture

# Result - train 99.29 test 99.86
# Analysis - The model is giving a decent accuracy but the number of parameters are high. Need to reduce so that they are under the constraint of 8k
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            
        ) # output_size = 26 

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 24

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # output_size = 22


        # TRANSITION BLOCK 1
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 22
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11

        
        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 9

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 7

        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 5

        # OUTPUT BLOCK        
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1
        


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool1(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

