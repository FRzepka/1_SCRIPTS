# Comparison report

- checkpoint: /home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/models/torch/1.5.0.0/outputs/checkpoints/soc_epoch0001_rmse0.02897.pt 

- checkpoint size: 276.95 KB (283,601 bytes)

- total parameters (from checkpoint): 22,657

- estimated storage: float32 88.50 KB (90,628 bytes), float16 44.25 KB, int8 22.13 KB


## ONNX: /home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/models/onnx/soc_1.5.0.0_streaming_int8.onnx

- file size: 84.26 KB (86,283 bytes)

- initializer elements: 22,917

- initializer bytes (inferred): 77.33 KB (79,182 bytes)

- dtypes breakdown: {'float32': 18627, 'uint8': 2, 'int8': 4160, 'int32': 128}


## ONNX: /home/florianr/MG_Farm/1_Scripts/DL_Models/LFP_LSTM_MLP/models/onnx/soc_1.5.0.0_streaming_fp32.onnx

- file size: 92.34 KB (94,557 bytes)

- initializer elements: 22,657

- initializer bytes (inferred): 88.50 KB (90,628 bytes)

- dtypes breakdown: {'float32': 22657}


## Comparison summary

- parameter count: 22,657

- estimated float32 raw: 88.50 KB (90,628 bytes)

- soc_1.5.0.0_streaming_int8.onnx initializer bytes: 77.33 KB (79,182 bytes)

- soc_1.5.0.0_streaming_fp32.onnx initializer bytes: 88.50 KB (90,628 bytes)
