# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:40:27 2023

@author: Florian
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates 
import os, sys
import numpy as np
import pickle
from fastparquet import ParquetFile
from fastparquet import write

# Pfad zur GZIP-komprimierten CSV-Datei
#file_path = r"C:\Users\Florian\MGFarm\7_State_estimation\02_NMC\BX_SOH_Parquet\BX_SOH_C01\cycling\BX_SOH_C01_CYC_003.csv.parquet"
# Datei entpacken und mit pandas laden
#df = pd.read_parquet(file_path)
# DataFrame anzeigen
#print(df)

def readcsv(checkup_dir, cycle_dir):
    cu_df = pd.read_table(checkup_dir, delimiter = ';')
    cy_df = pd.read_table(cycle_dir, delimiter = ';')
    return cu_df, cy_df

def file_path(i, j):
    #i: folder
    #j: file    
    path_dir: str = r"C:\Users\Florian\MGFarm\7_State_estimation\02_NMC\BX_SOH_Parquet"
    folder_dir = os.listdir(path_dir)
    folder_name = folder_dir[i]
    try:
        cycle_dir: str = "{}\{}\{}".format(path_dir, folder_name, "cycling") 
        cy_files_name = os.listdir(cycle_dir)
        cycle_dir: str = "{}\{}".format(cycle_dir, cy_files_name[j])
    except:
        cycle_dir: str = "{}\{}\{}".format(path_dir, folder_name, "cycling") 
        cy_files_name = os.listdir(cycle_dir)
        cycle_dir: str = "{}\{}".format(cycle_dir, cy_files_name[j])
        print('keine Cycle Datei')
        pass
    return folder_dir, folder_name, cycle_dir, cy_files_name


def combine_df(): 
    # Für Start
    i = 0 # i: file
    j = 0 # j: folder
        
    # Start: folder, file = 0; Fehler: folder, file = ab Fehlerpunkt
    folder = 0
    file = 0
        
  
    # folder_dir: Anzahl der Ordner für Ende der Schleife
    folder_dir = file_path(i,j)[0]
    combined_df_list = [] # Liste der einzelnen DataFrames
    path_dir: str = r"C:\Users\Florian\MGFarm\7_State_estimation\02_NMC"
    output_dir = os.path.join(path_dir, "BX_SOH_Combined")
    os.makedirs(output_dir, exist_ok=True)  # Ordner für kombinierte DataFrames erstellen
    for i in range(folder, len(folder_dir)):
        # Folder i
        #print(i)
        folder_dir, folder_name, cycle_dir, cy_files_name = file_path(i, j)
        
        file_end = len(cy_files_name)
        df_list = []  # Liste für die einzelnen DataFrames
        #file_end = 25
        for j in range(file, file_end):
            # File j
            cycle_dir = file_path(i, j)[2]
            try:
                df = pd.read_parquet(cycle_dir)
                df_list.append(df)
                print(cycle_dir)
            except:
                print('Error')
                pass 
        combined_df = pd.concat(df_list)  # DataFrames in der Liste zusammenfügen        
        combined_df_list.append((folder_name, combined_df))
        print(folder_name)
        file = j = 0
        # Speichern des kombinierten DataFrames mit dem entsprechenden Namen aus folder_dir
        #combined_df_name = folder_name + ".parquet"
        #combined_df.to_parquet(combined_df_name)
        #print("Kombinierter DataFrame für", folder_name, "gespeichert")
        #file = j = 0
        
        # Speichern des kombinierten DataFrames in einem separaten Ordner mit dem entsprechenden Namen aus folder_dir
        combined_df_name = os.path.join(output_dir, folder_name + ".parquet")
        combined_df.to_parquet(combined_df_name)
        print("Kombinierter DataFrame für", folder_name, "gespeichert")
        file = j = 0


    return combined_df_list

'0. Erstellen der ersten Datei für Head der Dataframes und Funktionen'
#folder_dir, folder_name, cycle_dir, cy_files_name = file_path(0, 4)
combined_df_list = combine_df()



