import onnx

m = onnx.load("battery_lstm_768.onnx")
for i in m.graph.input:
    shape = [d.dim_value for d in i.type.tensor_type.shape.dim]
    print(i.name, shape)
