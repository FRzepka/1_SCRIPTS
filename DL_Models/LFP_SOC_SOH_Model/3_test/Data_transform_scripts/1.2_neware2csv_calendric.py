import pickle
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
import datetime
from datetime import datetime
from tqdm import tqdm

# === DEFINITION DER PFADE ===
path = r'c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\4_Work_packages\3_State_estimation\01_LFP\00_Data\Versuch_18650\Calendric'
path_store = r'c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\4_Work_packages\3_State_estimation\01_LFP\00_Data\Versuch_18650_standart\Calendric'

# Stelle sicher, dass der Zielordner existiert; wenn nicht, erstelle ihn
if not os.path.exists(path_store):
    os.makedirs(path_store)

# === HILFSFUNKTIONEN ===

def get_cell_and_cycle_folders(path):
    """Gibt eine Liste von Zellen und Zyklen basierend auf der Ordnerstruktur zurück."""
    cell_folders = [folder for folder in os.listdir(path) if folder.startswith('MGFarm_18650_')]
    cells = [folder.split('_')[-1] for folder in cell_folders]
    return cells

def read_data_from_files(path_data):
    """Liest Daten aus Excel-Dateien und gibt DataFrames zurück."""
    print('Function: read_data_from_files')
    
    # Dateiliste für den aktuellen Pfad
    xls_dateien = [os.path.join(path_data, datei) for datei in os.listdir(path_data) if datei.endswith('.xls')]
    
    data_raw = pd.DataFrame()
    data_temp = pd.DataFrame()
    data_dms = pd.DataFrame()
    data_statis = pd.DataFrame()
        
    for file in tqdm(xls_dateien, desc="Files"):
        print(f"Verarbeite Datei: {file}")
        aux_data = pd.ExcelFile(file)
        for sheet in tqdm(aux_data.sheet_names, desc="Sheets"):
            if sheet.startswith('Detail_'):
                aux_soh = pd.read_excel(file, sheet_name=sheet)                
                data_raw = pd.concat([data_raw, aux_soh], ignore_index=True)
            elif sheet.startswith('DetailTemp'):
                aux_temp = pd.read_excel(file, sheet_name=sheet)                
                data_temp = pd.concat([data_temp, aux_temp], ignore_index=True) 
            elif sheet.startswith('DetailVol'):
                aux_dms = pd.read_excel(file, sheet_name=sheet)
                data_dms = pd.concat([data_dms, aux_dms], ignore_index=True) 
            elif sheet.startswith('Statis'):
                aux_statis = pd.read_excel(file, sheet_name=sheet)
                data_statis = pd.concat([data_statis, aux_statis], ignore_index=True) 
    
    return data_raw, data_temp, data_dms, data_statis

def process_data(data_raw, data_temp, data_dms, data_statis, columns):
    """Verarbeitet die rohen Daten und gibt bereinigte und kombinierte Daten zurück."""
    print('Function: process_data')
    
    # Daten zusammenführen, nur wenn die DataFrames nicht leer sind
    if not data_temp.empty:
        if 'Steptime[s]' not in data_raw.columns:
            print("Column 'Steptime[s]' missing in data_raw. Available columns:")
            print(data_raw.columns)
        if 'Relative Time(h:min:s.ms)' not in data_temp.columns:
            print("Column 'Relative Time(h:min:s.ms)' missing in data_temp. Available columns:")
            print(data_temp.columns)
        data_raw[data_temp.columns[4]] = data_temp[data_temp.columns[4]]
    
    if not data_dms.empty:
        data_raw[data_dms.columns[4]] = data_dms[data_dms.columns[4]]

    # Überflüssige Daten löschen
    data_temp = pd.DataFrame()
    data_dms = pd.DataFrame()

    # Einheiten konvertieren (wenn notwendig)
    if data_raw.columns[5][-3] == 'm':
        data_raw[data_raw.columns[5]] = data_raw[data_raw.columns[5]] / 1000
        data_raw[data_raw.columns[7]] = data_raw[data_raw.columns[7]] / 1000
        data_raw[data_raw.columns[8]] = data_raw[data_raw.columns[8]] / 1000
    data_raw.columns = columns

    # Raw Step ID hinzufügen
    for step in data_statis['Step']:
        step_id = data_statis.loc[data_statis['Step'] == step]
        data_raw.loc[data_raw['Step'] == step, 'Raw Step ID'] = int(step_id['Raw Step ID'])

    return data_raw


def save_data(data, path_store, campaign, cell, cycle):
    """Speichert den DataFrame als CSV-Datei."""
    print('save_data')
    
    df = pd.read_csv(r'c:\Users\Florian\TUB\3_Projekte\MG_Farm\4_Work_packages\0_Skripte\2. Neware_scripts\CSV_Vorlage.csv', sep=";")
    df = df.drop('Unnamed: 18', axis=1)
    
    df['Absolute_Time[yyyy-mm-dd hh:mm:ss]'] = data['Absolute Time']
    df['Step_Time[s]'] = pd.to_timedelta(data['Steptime[s]']).dt.total_seconds()
    df['Step_ID'] = data['Step']
    
    # Testzeit berechnen
    testtime = df['Step_Time[s]'].loc[df['Step_ID'] == 1]
    for i in range(2, df['Step_ID'].iloc[-1] + 1, 1):
        steptime = df['Step_Time[s]'].loc[df['Step_ID'] == i]
        steptime += testtime.iloc[-1]
        testtime = pd.concat([testtime, steptime])

    df['Testtime[s]'] = testtime
    df['Schedule_Step_ID'] = data['Raw Step ID'].astype('int64')
    df['Voltage[V]'] = data['Voltage[V]']
    df['Current[A]'] = data['Current[A]']
    df['Temperature[°C]'] = data['Temperature[°C]']
    df['Aux_U[V]'] = data['AuxU[V]']
    
    # Überprüfen und erstellen von Verzeichnissen, wenn sie nicht existieren
    save_path = os.path.join(path_store, campaign + '_' + cell)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Speichern des DataFrames basierend auf der Zelle und dem Zyklus
    df.to_csv(os.path.join(save_path, campaign + '_' + cell + '_' + cycle + '.csv'), sep=";", index=False)


# === HAUPTVERARBEITUNG ===
columns = ['Record Index','Status','JumpTo','Cycle','Step','Current[A]', 'Voltage[V]','CapaCity(Ah)',
           'Energy(Wh)','Steptime[s]','Absolute Time','Temperature[°C]', 'AuxU[V]'] 

# Identifizierung der Zellen und Zyklen
cells = get_cell_and_cycle_folders(path)

for cell in cells:
    cell_path = os.path.join(path, 'MGFarm_18650_' + cell)
    
    # Überprüfen, ob cell_path ein Verzeichnis ist
    if not os.path.isdir(cell_path):
        print(f"Skipping {cell_path} because it's not a directory.")
        continue

    cycles = [folder.split('_')[-1] for folder in os.listdir(cell_path) if os.path.isdir(os.path.join(cell_path, folder))]
    
    for cycle in cycles:
        # Überprüfen, ob die CSV-Datei für diese Zelle und diesen Zyklus bereits existiert
        csv_filename = f"MGFarm_18650_{cell}_{cycle}.csv"
        csv_path = os.path.join(path_store, "MGFarm_18650_" + cell, csv_filename)
        path_data = os.path.join(cell_path, 'MGFarm_18650_' + cell + '_' + cycle)
        
        if os.path.exists(csv_path):
            print(f"Skipping {csv_filename} because it already exists.")
            continue
        
        try:
            path_data = os.path.join(cell_path, 'MGFarm_18650_' + cell + '_' + cycle)
            
            # Daten aus Dateien lesen
            data_raw, data_temp, data_dms, data_statis = read_data_from_files(path_data)
            
            # Daten verarbeiten
            data_raw = process_data(data_raw, data_temp, data_dms, data_statis, columns)
            
            # Daten speichern
            save_data(data_raw, path_store, 'MGFarm_18650', cell, cycle)
            
            print('MGFarm_18650_' + cell + '_' + cycle + '...finished')

        except Exception as e:
            print('Error reading: '+'MGFarm_18650_'+cell+'_'+cycle)
            print("Error Details:", e)
