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

input_names = ["star"]
output_names = ["end"]
torch.onnx.export(model, x, "NN1.onnx", input_names=input_names, output_names=output_names)
onnx_model = onnx.load("NN1.onnx")
onnx_model.graph

from onnx2torch import convert
model_2 = convert(onnx_model)
model_2(x)

from onnx2keras import onnx_to_keras
k_model = onnx_to_keras(onnx_model, ['input'])
