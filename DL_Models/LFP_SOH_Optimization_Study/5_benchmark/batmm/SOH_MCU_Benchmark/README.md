## Preparation

- In a ./data folder (this can be changed in the ```config.py```), add all parquet files of the battery SOH measurements.
- Create two environments with the given YAMLs (due to conflicting tensorflow variants for the PC sided compression and the MCU sided inference, we need two)
- Download STM32 Cube IDE (project created with v2.1.1), STEdgeAI (v4.0) and Cube MX (v6.17).
- Move the template projects in the directory specified by ```CUBE_MX_DIR``` (editable) in ```config.py```

## Pruning and Quantizing

- Adjust your preferred settings in ```config.py```
- With stm_pc active, run ```kerasify.py```

## Running on the MCU

- With stm_mcu active, run ```benchmark.py```
- Make sure to have attached the USB Meter (first plug in the MCU, then the USB Meter so the ports get assigned correctly) (TODO: script that works w/o USB meter attached)
- Visualize with ```visualize.py```, it needs the resulting ```inference_results.csv``` and the ```tracker``` folder containing the measured CSVs.

## Scripts

| Script | Explanation |
| -------- | ------- |
| ```kerasify.py``` | Takes all base models, converts them to keras format (because only .keras files get quantized correctly), and prunes and quantizes them according to pruning degrees specified in ```config.py``` |
| ```benchmark.py``` | Iterates over every file in models/compressed_models (created by running the compression script) and benchmarks it on the MCU |
| ```config.py``` | Specify settings such as inter alia pruning degrees, datasets to use, STM CUBE paths or ports |
