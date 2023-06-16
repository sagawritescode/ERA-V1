import torch
import torch.nn as nn
import torch.nn.functional as F


# Target - to get the code structure right 
# Result - train - 99.87, test - 99.19
# Analysis - There was no accuracy specific target with this model, but from the results it can be said that the model was clearly overfitting. 
# The number of parameters been huge the model had an ability to overfit 
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()

        # INPUT BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU()
        ) # output_size = 28

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU()
        ) # output_size = 28


        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14

        # CONVOLUTION BLOCK 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU()
        ) # output_size = 14

        
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU()
        ) # output_size = 14

        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 7

        # OUTPUT BLOCK
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 5
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 3
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
        ) # output_size = 1
        


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        # x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
