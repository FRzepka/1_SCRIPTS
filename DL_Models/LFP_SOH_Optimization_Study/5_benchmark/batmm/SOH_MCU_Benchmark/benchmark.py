import os
import shutil
import subprocess
import sys
import glob

from config import COMPRESSED_MODELS_DIR, CHUNK_SIZES, STEDGEAI_BIN, STEDGEAI_CORE_DIR, HEADLESS_BUILD_BIN, PROGRAMMER_CLI_BIN, TEMP_FILES_DIR, BASE_WORKSPACE_DIR, CUBE_MX_DIR

ai_runner_path = f"{STEDGEAI_CORE_DIR}/scripts/ai_runner"   # Environment variables that need to be set for stm_ai_runner before imports
if ai_runner_path not in sys.path:
    sys.path.insert(0, ai_runner_path)

from src.inference_conv import infer_conv
from src.inference_rnn import infer_rnn

env_config = os.environ.copy()
env_config["STEDGEAI_CORE_DIR"] = STEDGEAI_CORE_DIR
current_pythonpath = env_config.get("PYTHONPATH", "")
env_config["PYTHONPATH"] = f"{ai_runner_path}:{current_pythonpath}"


# ----- CONGIG -----
INFER_FN = {
    "cnn":  infer_conv,
    "tcn":  infer_conv,
    "lstm": infer_rnn,
    "gru":  infer_rnn,
}


def run_command(cmd, command_description, pass_env=None):
    """Helper function to run external CLI tools"""
    print(f"\n%%%%% {command_description}...")
    try:
        subprocess.run(cmd, env=pass_env, check=True)
        print(f"%%%%% {command_description} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"##### CRITICAL ERROR during: {command_description}", file=sys.stderr)
        raise


# Gather the models
model_files = sorted([
    f for f in os.listdir(COMPRESSED_MODELS_DIR)
    if os.path.isfile(os.path.join(COMPRESSED_MODELS_DIR, f))
    and f.endswith((".keras", ".tflite"))
])

if not model_files:
    print(f"##### No .keras or .tflite files found in {COMPRESSED_MODELS_DIR}. Exiting.")
    sys.exit(0)

print(f"%%%%% Found {len(model_files)} model(s) in {COMPRESSED_MODELS_DIR}:")
for f in model_files:
    print(f"%%%%%   {f}")


# ----- BENCHMARKING -----
successful_models = []
failed_models = []

for model_file in model_files:
    model_path = os.path.join(COMPRESSED_MODELS_DIR, model_file)
    model_name, ext = os.path.splitext(model_file)
    quantized = ext == ".tflite"

    # Determine architecture from filename prefix
    arch = next((a for a in INFER_FN if model_name.startswith(a)), None)
    if arch is None:
        print(f"\n##### Skipping {model_file}: filename doesn't start with a known architecture (cnn/tcn/lstm/gru)")
        failed_models.append((model_file, "Skipped: Unknown architecture prefix"))
        continue

    print(f"\n%%%%% {'='*64}")
    print(f"%%%%%   FILE: {model_file}  |  arch={arch}  |  quantized={quantized}")
    print(f"%%%%% {'='*64}")

    # Generate an isolated, model-specific workspace path
    MODEL_WORKSPACE = os.path.join(BASE_WORKSPACE_DIR, f"ws_{model_name}")

    try:
        TEMPLATE_PROJECT_DIR = f"{CUBE_MX_DIR}/keras_{arch}"
        PROJECT_NAME = f"{model_name}_benchmark"
        NEW_PROJECT_DIR = f"{CUBE_MX_DIR}/{model_name}_benchmark"

        # Clean up old workspace & metadata
        if os.path.exists(TEMP_FILES_DIR):
            shutil.rmtree(TEMP_FILES_DIR)
        os.makedirs(TEMP_FILES_DIR, exist_ok=True)

        if os.path.exists(MODEL_WORKSPACE):
            shutil.rmtree(MODEL_WORKSPACE)
        os.makedirs(MODEL_WORKSPACE, exist_ok=True)

        # Generate ST Edge AI code
        stedgeai_cmd = [
            STEDGEAI_BIN, "generate",
            "--model", model_path,
            "--target", "stm32h7",
            "--c-api", "st-ai",
            "--workspace", TEMP_FILES_DIR,
            "--output", TEMP_FILES_DIR,
            "--with-report"
        ]
        run_command(stedgeai_cmd, "Generating ST Edge AI C files", pass_env=env_config)

        # Duplicate the working template project layout
        print("\n%%%%% Copying template project...")
        if os.path.exists(NEW_PROJECT_DIR):
            shutil.rmtree(NEW_PROJECT_DIR)
        shutil.copytree(TEMPLATE_PROJECT_DIR, NEW_PROJECT_DIR)
        print(f"%%%%% Template copied to {NEW_PROJECT_DIR}")

        # Rename internal project name (project would otherwise have a conflict with the original it was copied from)
        print("\n%%%%% Renaming internal project ID...")
        project_meta_path = os.path.join(NEW_PROJECT_DIR, ".project")
        if os.path.exists(project_meta_path):
            with open(project_meta_path, "r") as f:
                content = f.read()
            updated_content = content.replace(f"<name>keras_{arch}</name>", f"<name>{PROJECT_NAME}</name>")
            with open(project_meta_path, "w") as f:
                f.write(updated_content)
            print(f"%%%%% Project ID successfully renamed to {PROJECT_NAME}")
        else:
            print("##### Warning: .project file not found in template destination")

        print("\n%%%%% Merging generated network files into new project...")
        target_app_dir = os.path.join(NEW_PROJECT_DIR, "AI", "App")
        os.makedirs(target_app_dir, exist_ok=True)
        generated_network_files = glob.glob(os.path.join(TEMP_FILES_DIR, "network*"))
        for file_path in generated_network_files:
            shutil.copy(file_path, target_app_dir)
        print(f"%%%%% Merged {len(generated_network_files)} core network layers into destination framework")

        # Compile firmware
        build_cmd = [
            HEADLESS_BUILD_BIN,
            "-data", MODEL_WORKSPACE,
            "-import", NEW_PROJECT_DIR,
            "-cleanBuild", f"{PROJECT_NAME}/Release"
        ]
        run_command(build_cmd, "Compiling firmware via STM32CubeIDE Headless Builder")

        # Flash to MCU
        flash_cmd = [
            PROGRAMMER_CLI_BIN,
            "-c", "port=SWD", "mode=UR",
            "-e", "all",               # Force full flash sector cleanup in beforehand
            "-d", os.path.join(NEW_PROJECT_DIR, "Release", f"{PROJECT_NAME}.elf"),
            "-v",
            "-rst"
        ]
        run_command(flash_cmd, "Resetting hardware and flashing to MCU")

        print("%%%%% Finished flashing!")

        # Run inference
        print(f"\n%%%%% Running inference benchmark (arch={arch}, quantized={quantized})...")
        INFER_FN[arch](model_path=model_path, quantized=quantized, chunk_size=CHUNK_SIZES[arch])

        # If code reaches this point, the entire pipeline succeeded!
        successful_models.append(model_file)

    except Exception as error:
        print(f"\n##### PIPELINE FAILED FOR MODEL: {model_file}")
        print(f"##### Reason: {error}")
        print("##### Skipping to the next model...\n")

        # Log the failure for the final report
        failed_models.append((model_file, str(error)))
        continue

    finally:
        # Clean up the temporary workspace directory
        if os.path.exists(MODEL_WORKSPACE):
            shutil.rmtree(MODEL_WORKSPACE)


print("%%%%% BENCHMARKING SUMMARY")
print(f"%%%%% {'='*64}")

print(f"\n%%%%% SUCCESSFUL MODELS ({len(successful_models)}):")
if successful_models:
    for m in successful_models:
        print(f"%%%%%   {m}")
else:
    print("%%%%%  None")

print(f"\n##### FAILED MODELS ({len(failed_models)}):")
if failed_models:
    for m, reason in failed_models:
        print(f"%%%%%   {m}")
        print(f"%%%%%      Reason: {reason}")
else:
    print("%%%%%  None")
print(f"\n%%%%% {'='*64}\n")
