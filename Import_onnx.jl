using ONNX

path = "NN1.onnx"
foo = ONNX.load(path, randn(2, 1))
