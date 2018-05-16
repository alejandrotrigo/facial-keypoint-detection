## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # input size = 1 x 224 x 224 output size = 32 x 220 x 220
        # after pooling = 32 x 110 x 110
        self.conv0_bn = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 32, 5)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv1_bn = nn.BatchNorm2d(32)
        
        # input_size = 32 x 110 x 110 -> output_size = 64 x 108 x 108
        # after pooling = 64 x 54 x 54
        self.conv2 = nn.Conv2d(32, 64, 3) 
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm2d(64)
        
        self.drop2d = nn.Dropout2d(p=0.3)
        # input_size = 64 x 54 x 54 -> output_size = 128 x 52 x 52
        # after pooling = 128 x 26 x 26
        self.conv3 = nn.Conv2d(64, 128, 3)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.conv3_bn = nn.BatchNorm2d(128)
        
        # input_size = 128 x 26 x 26 -> output_size = 256 x 24 x 24
        # after pooling = 256 x 4 x 4
        self.conv4 = nn.Conv2d(128, 256, 3)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(6,6)
        
        self.fc1 = nn.Linear(256 * 4 * 4, 2048)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.drop = nn.Dropout(p=0.4)
        self.fc1_bn = nn.BatchNorm1d(2048)
        
        self.fc2 = nn.Linear(2048, 1024)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2_bn = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 256)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(p=0.2)
        
        self.fc4 = nn.Linear(256, 136)
        nn.init.xavier_uniform_(self.fc4.weight)
       
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.conv0_bn(x)
        x = self.pool1(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool1(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.drop2d(x)
        x = self.pool1(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool2(F.relu(self.conv4_bn(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.drop(x)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = self.drop2(x)
        x = self.fc4(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

class EasyNet(nn.Module):
    
    def __init__(self):
        super(EasyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool2_2 = nn.MaxPool2d(2,2)
        self.conv1_drop = nn.Dropout2d(p=0.3)
        
        # input_size = 110 x 110 x 1 -> 108 x 108 x 1 -> max_pooling 54 x 54 x 1
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv3_drop = nn.Dropout2d(p=0.3)
        
        # input_size = 54 x 54 x 1 -> 52 x 52 x 1 -> max_pooling_6_6 26 x 26 x 1
        self.conv4 = nn.Conv2d(32, 32, 3)
        self.pool6_6 = nn.MaxPool2d(13, 13)
        self.conv4_bn = nn.BatchNorm2d(32)
        
        self.fc1 = nn.Linear(4 * 4 * 32, 256)
        
        self.drop3 = nn.Dropout(p=0.3)
        
        self.fc3 = nn.Linear(256, 136)
       
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool2_2(F.relu(self.conv1(x)))
        x = self.conv1_drop(x)

        x = self.pool2_2(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.conv3_drop(x)
        x = self.pool6_6(F.relu(self.conv4_bn(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = F.relu(self.fc3(x))
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
