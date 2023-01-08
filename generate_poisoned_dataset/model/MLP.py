import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self,x):
        return x.reshape(x.shape[0],-1)

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.flatten = Flatten()
        self.fc = nn.Sequential(
            nn.Linear(4096,16384),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(16384,16384),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16384, 16384),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16384,10)
        )
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class three_layer_NN(nn.Module):
    def __init__(self, input_dims, output_dims=10):
        super(three_layer_NN, self).__init__()
        self.linear1 = nn.Linear(input_dims, input_dims)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(input_dims, int(input_dims/2))
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(int(input_dims/2), output_dims)

    def forward(self, x):
        x = x.view(len(x), -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

class Linear(nn.Module):
    def __init__(self, input_dims, output_dims=10, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)

    def forward(self, x):
        #print(x.size())
        x = x.view(-1, 3072)
        x = self.linear(x)
        return x

class two_layer_NN(nn.Module):
    def __init__(self, input_dims, output_dims=10, bias=True):
        super(two_layer_NN, self).__init__()
        self.linear1 = nn.Linear(input_dims, input_dims, bias=bias)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(input_dims, output_dims, bias=bias)

    def forward(self, x):
        x = x.view(len(x), -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x