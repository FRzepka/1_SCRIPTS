#!/bin/bash

#SBATCH -o C12431%j.%N.out  # Output-Datei im aktuellen Verzeichnis
#SBATCH -D .                          # Working Directory auf das aktuelle Verzeichnis setzen
#SBATCH -J C12431               # Job Name an den Script-Namen angepasst
#SBATCH --ntasks=1                    # Anzahl Prozesse P (CPU-Cores)
#SBATCH --cpus-per-task=8             # Anzahl CPU-Cores pro Prozess P
#SBATCH --mem=250G                     # 30GiB resident memory pro Node
#SBATCH --time=2-00:00:00             # Laufzeit auf 1 Woche setzen
#SBATCH --partition=gpu         # Auf GPU-Knoten in der gpu_short Partition rechnen
#SBATCH --mail-type=ALL               # Job-Status per Mail
#SBATCH --mail-user=vorname.nachname@tu-berlin.de

# Miniconda initialisieren
source /home/users/f/flo01010010/miniconda3/bin/activate
conda activate base

# Zum Skript-Verzeichnis wechseln
cd /home/users/f/flo01010010/HPC_projects/Scripts/BMS/Python/BMS_SOC_LSTM_stateful_1.2.4.31_Train_CPU

# Python-Skript ausführen
python BMS_SOC_LSTM_stateful_1.2.4_Train.py
