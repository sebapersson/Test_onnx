import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(in_features=2, out_features=5)
        self.hidden_1 = nn.Linear(in_features=5, out_features=5)
        self.hidden_2 = nn.Linear(in_features=5, out_features=5)
        self.output = nn.Linear(in_features=5, out_features=2)
        
    def forward(self, x):
        x = F.tanh(self.input(x))
        x = F.tanh(self.hidden_1(x))
        x = F.tanh(self.hidden_2(x))
        return self.output(x)
    
model = Net()
x = torch.randn(2)
out = model(x)

input_names = ["prey", "predator"]
output_names = ["kout"]
torch.onnx.export(model, x, "NN1.onnx", input_names=input_names, output_names=output_names)

onnx_model = onnx.load("NN1.onnx")