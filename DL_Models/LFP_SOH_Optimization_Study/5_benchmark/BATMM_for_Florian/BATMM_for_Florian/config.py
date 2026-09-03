# ------ MCU -----
SERIAL_PORT = "/dev/ttyACM0"    # Port for MCU
METER_PORT  = "/dev/ttyACM1"    # Port for USB tracker
BAUD_RATE   = 115200


# ----- MODELS -----
MODELS_DIR = "./models"
COMPRESSED_MODELS_DIR = "./models/compressed_models"
SCALER_PATH = "./models/LSTM/scaler_robust.joblib"
MODEL_ARCHS =   [
    "cnn",
    "tcn",
    "lstm",
    "gru",
]
CHUNK_SIZES = {
    "cnn":  128,   # as of training yaml
    "tcn":  96,
    "lstm": 192,
    "gru":  160,
}


# ----- DATA -----
TRAIN_CELLS = [
    "df_FE_C01",
    "df_FE_C03",
    "df_FE_C05",
    "df_FE_C07",
    "df_FE_C09",
    "df_FE_C13",
    "df_FE_C17",
    "df_FE_C19",
    "df_FE_C25",
    "df_FE_C27",
]
VAL_CELLS = [
    "df_FE_C15",
    #"df_FE_C21"
    ]
TEST_CELLS = [
    "df_FE_C11",    # 2716
    "df_FE_C23",    # 4028
    "df_FE_C29"     # 1345
    ]

LIMIT_DATA_PER_DATASET = 0  # mainly for debugging, if you just want to take x samples from the datasets
DATA_DIR = "./data"


# ----- BENCHMARKING -----
RESULTS_DIR = "./results"
TRACK_EVERY = 50


# ----- PRUNING -----
PRUNING_RATIOS = {      # Pruning ratios to sweep and how many iterative rounds each gets
    #0.05: 1,
    #0.10: 1,
    0.20: 2,
    #0.30: 2,
    #0.40: 3,
    #0.50: 4,
    #0.60: 6,
    #0.70: 8,
    #0.80: 10,
    #0.85: 12,
    #0.90: 15,
    #0.95: 20,
    #0.99: 25,
}
PRUNE_SETTINGS = {
    "--pruning_mode":    "iterative",   # oneshot
    "--finetune_epochs": "10",
    "--finetune_lr":     "1e-4",
    "--distill_strategy": "base",
    "--distill_alpha":    "0.5",
}
EARLY_STOPPING = 5

# ----- CUBE -----
STEDGEAI_BIN = "/home/jonas/ST/STEdgeAI/4.0/Utilities/linux/stedgeai"
STEDGEAI_CORE_DIR = "/home/jonas/ST/STEdgeAI/4.0"
HEADLESS_BUILD_BIN = "/opt/st/stm32cubeide_2.1.1/headless-build.sh"
PROGRAMMER_CLI_BIN = "/opt/st/stm32cubeide_2.1.1/plugins/com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.linux64_2.2.400.202601091506/tools/bin/STM32_Programmer_CLI"
TEMP_FILES_DIR = "/home/jonas/STM32CubeMX/TEMP_FILES"
BASE_WORKSPACE_DIR = "/home/jonas/STM32CubeMX/workspaces"
CUBE_MX_DIR = "/home/jonas/STM32CubeMX"
