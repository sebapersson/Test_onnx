import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(in_features=2, out_features=5), 
                       F.tanh,
                       nn.Linear(in_features=5, out_features=5), 
                       F.tanh, 
                       nn.Linear(in_features=5, out_features=5), 
                       F.tanh,
                       nn.Linear(in_features=5, out_features=2)]
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
model = Net()
x = torch.randn(2)
out = model(x)

input_names = ["input"]
output_names = ["end"]
torch.onnx.export(model, x, "NN1.onnx", input_names=input_names, output_names=output_names)

onnx_model = onnx.load("NN1.onnx")
from onnx2keras import onnx_to_keras
k_model = onnx_to_keras(onnx_model, ['input'])
