import os
import subprocess
import sys
import glob
import time

# --- CONFIGURATION ---
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../STM32/workspace_1.17.0"))

PROJECTS = {
    "SOC_Base": "AI_Project_LSTM_SOC_base",
    "SOC_Pruned": "AI_Project_LSTM_SOC_pruned",
    "SOC_Quantized": "AI_Project_LSTM_SOC_quantized",
    "SOH_Base": "AI_Project_LSTM_SOH_base",
    "SOH_Pruned": "AI_Project_LSTM_SOH_pruned",
    "SOH_Quantized": "AI_Project_LSTM_SOH_quantized",
    # Legacy support (defaults to SOC if not specified)
    "Base": "AI_Project_LSTM_SOC_base",
    "Pruned": "AI_Project_LSTM_SOC_pruned",
    "Quantized": "AI_Project_LSTM_SOC_quantized"
}

# Common default paths for STM32CubeIDE tools (Update these if your version differs!)
ST_ROOT = r"C:\ST"
POSSIBLE_PROGRAMMER_PATHS = [
    r"C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe",
    r"C:\ST\STM32CubeIDE_1.17.0\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.win32_*\tools\bin\STM32_Programmer_CLI.exe"
]

# We need 'make.exe' and 'arm-none-eabi-gcc.exe'
# Usually found in STM32CubeIDE plugins
POSSIBLE_BUILD_TOOLS_PATHS = [
    r"C:\ST\STM32CubeIDE_1.17.0\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.13.3.rel1.win32_*\tools\bin",
    r"C:\ST\STM32CubeIDE_1.17.0\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.make.win32_*\tools\bin"
]

def find_tool(tool_name, search_paths):
    # Check system path first
    if shutil.which(tool_name):
        return tool_name
    
    # Search in specific paths with globbing
    for pattern in search_paths:
        matches = glob.glob(pattern)
        for path in matches:
            # If path is a directory, look for tool inside
            if os.path.isdir(path):
                tool_path = os.path.join(path, tool_name)
                if os.path.exists(tool_path):
                    return tool_path
            # If path is the file itself
            elif os.path.isfile(path) and os.path.basename(path).lower() == tool_name.lower():
                return path
    return None

import shutil

def add_build_tools_to_path():
    """Finds make and gcc and adds them to os.environ['PATH']"""
    
    # 1. Find make.exe
    make_path = None
    # Search specifically for make directories
    make_patterns = [
        r"C:\ST\STM32CubeIDE_1.17.0\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.make.win32_*\tools\bin"
    ]
    for pattern in make_patterns:
        matches = glob.glob(pattern)
        if matches:
            make_path = matches[0]
            break
            
    # 2. Find arm-none-eabi-gcc
    gcc_path = None
    gcc_patterns = [
        r"C:\ST\STM32CubeIDE_1.17.0\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.13.3.rel1.win32_*\tools\bin"
    ]
    for pattern in gcc_patterns:
        matches = glob.glob(pattern)
        if matches:
            gcc_path = matches[0]
            break

    if make_path and gcc_path:
        print(f"Found Build Tools:\n  Make: {make_path}\n  GCC:  {gcc_path}")
        os.environ["PATH"] += os.pathsep + make_path + os.pathsep + gcc_path
        return True
    else:
        print("ERROR: Could not find STM32 build tools (make/gcc) automatically.")
        print("Please ensure STM32CubeIDE 1.17.0 is installed in C:\\ST\\")
        return False

def get_programmer_path():
    for pattern in POSSIBLE_PROGRAMMER_PATHS:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None

def find_elf_file(project_dir, project_name):
    """
    Find the ELF file to flash.
    Prefer <project_name>.elf, but fall back to any *.elf in the Debug folder.
    This makes the script robust against STM32CubeIDE projects that kept the
    default 'AI_Project_LSTM.elf' name (e.g. SOH_* projects).
    """
    preferred = os.path.join(project_dir, f"{project_name}.elf")
    if os.path.exists(preferred):
        return preferred

    candidates = [
        f for f in glob.glob(os.path.join(project_dir, "*.elf"))
        if os.path.isfile(f)
    ]
    if not candidates:
        return None

    if len(candidates) > 1:
        candidates.sort(key=os.path.getmtime, reverse=True)

    return candidates[0]

def run_command(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command. Return code: {e.returncode}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python automate_stm32.py <ModelType> [action]")
        print("ModelType: Base, Pruned, Quantized")
        print("action: build, flash, all (default)")
        sys.exit(1)

    model_type = sys.argv[1]
    action = sys.argv[2] if len(sys.argv) > 2 else "all"

    if model_type not in PROJECTS:
        print(f"Unknown model type: {model_type}. Available: {', '.join(PROJECTS.keys())}")
        sys.exit(1)

    project_name = PROJECTS[model_type]
    project_dir = os.path.join(WORKSPACE_DIR, project_name, "Debug")

    if not os.path.exists(project_dir):
        print(f"Error: Project directory not found: {project_dir}")
        print("Make sure you have opened the project in STM32CubeIDE at least once to generate the makefiles.")
        sys.exit(1)

    # 1. Setup Environment
    if action in ["build", "all"]:
        if not add_build_tools_to_path():
            sys.exit(1)
        
        print(f"\n--- Building {model_type} ({project_name}) ---")
        # Clean
        if not run_command(["make", "clean"], cwd=project_dir):
            sys.exit(1)
        # Build
        if not run_command(["make", "-j16", "all"], cwd=project_dir):
            sys.exit(1)
        print("Build Successful!")

    # 2. Flash
    if action in ["flash", "all"]:
        programmer = get_programmer_path()
        if not programmer:
            print("ERROR: STM32_Programmer_CLI.exe not found.")
            sys.exit(1)
        
        print(f"\n--- Flashing {model_type} ---")
        elf_path = find_elf_file(project_dir, project_name)
        if not elf_path:
            print(f"Error: No .elf file found in {project_dir}")
            sys.exit(1)
        
        # Command: STM32_Programmer_CLI -c port=SWD -w <file.elf> -v -rst
        cmd = [programmer, "-c", "port=SWD", "-w", elf_path, "-v", "-rst"]
        if not run_command(cmd):
            print("Flashing Failed!")
            sys.exit(1)
        print("Flashing Successful!")

if __name__ == "__main__":
    main()
