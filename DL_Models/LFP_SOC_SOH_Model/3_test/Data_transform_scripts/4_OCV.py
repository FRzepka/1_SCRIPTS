# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:01:00 2023

@author: Florian
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from collections import defaultdict
import dill

plt.close('all')

def read_and_prepare_data(file_path):
    df = pd.read_parquet(file_path)
    df['Step_ID_Diff'] = df['Schedule_Step_ID'].diff()
    df['Testtime[s]'] = df['Testtime[s]'].astype(int)
    df.set_index('Testtime[s]', drop=False, inplace=True)
    return df

def calculate_ocv_columns(df):
    df['OCV_dchg'] = ((df['Schedule_Step_ID'] == 13) & (df['Step_ID_Diff'] == -1)).shift(-1, fill_value=False) | ((df['Schedule_Step_ID'] == 13) & (df['Step_ID_Diff'] == 1)).shift(-1, fill_value=False) | ((df['Schedule_Step_ID'] == 18) & (df['Step_ID_Diff'] == 1)).shift(-1, fill_value=False)
    df['OCV_chg'] = ((df['Schedule_Step_ID'] == 18) & (df['Step_ID_Diff'] == -1)).shift(-1, fill_value=False) | ((df['Schedule_Step_ID'] == 18) & (df['Step_ID_Diff'] == 1)).shift(-1, fill_value=False) | ((df['Schedule_Step_ID'] == 23) & (df['Step_ID_Diff'] == 1)).shift(-1, fill_value=False)
    return df

def find_start_end_points(df, start_id, end_id):
    start_points = df[((df['Schedule_Step_ID'] == start_id) & (df['Step_ID_Diff'] == 1)).shift(-1, fill_value=False)]
    end_points = df[((df['Schedule_Step_ID'] == end_id) & (df['Step_ID_Diff'] == 1)).shift(-1, fill_value=False)]
    return start_points, end_points

def calculate_Q_OCV(df, start_points, end_points, column_name):
    for start, end in zip(start_points['Testtime[s]'], end_points['Testtime[s]']):
        df.loc[start:end, column_name] = (df.loc[start:end, 'Current[A]'] * df.loc[start:end, 'Testtime[s]'].diff().fillna(0)).cumsum() / 3600
        df.loc[start, column_name] = 0
    return df

import numpy as np

def create_segments(df_ocv, start_points_list, end_points_list, column_name):
    segments = []
    desired_soc_values = np.linspace(0, 1, 101)  # Erzeuge Werte von 0 bis 1 in 0.01 Schritten
    
    for i in range(len(start_points_list)):
        start = start_points_list[i]
        end = end_points_list[i]
        segment_end_index = df_ocv[df_ocv['Testtime[s]'] <= end].index.max()
        segment = df_ocv.loc[start:segment_end_index].copy()  # Erstellen einer expliziten Kopie
        segment['SOC_OCV'] = segment[column_name] / segment[column_name].iloc[-1]
        if 'dchg' in column_name:
            segment['SOC_OCV'] = 1 - segment['SOC_OCV']

        
        # Neuen DataFrame für die gewünschten SOC_OCV Werte erstellen
        new_segment = pd.DataFrame({'SOC_OCV': desired_soc_values})
        
        # Für jeden gewünschten SOC_OCV-Wert den nächsten Wert in den ursprünglichen Daten finden
        for desired_soc in desired_soc_values:
            closest_index = segment['SOC_OCV'].sub(desired_soc).abs().idxmin()
            for col in segment.columns:
                new_segment.loc[new_segment['SOC_OCV'] == desired_soc, col] = segment.loc[closest_index, col]
                
        segments.append(new_segment)
    
    return segments


def plot_segments(ax, segments, x_col, y_col, title, cell_name):
    mean_temp = np.mean([segment['Temperature[°C]'].mean() for segment in segments])
    title = f"{title} - Mittlere Temperatur: {mean_temp:.2f}°C"
    ax.set_title(title)

    for i, segment in enumerate(segments):
        mean_soh = segment['SOH'].mean()
        ax.plot(segment[x_col], segment[y_col], label=f'Check-up {i+1} - SOH: {mean_soh:.2f}%')
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()

def process_file(file_path):
    df = read_and_prepare_data(file_path)
    df = calculate_ocv_columns(df)

    start_points_dchg, end_points_dchg = find_start_end_points(df, 13, 18)
    start_points_chg, end_points_chg = find_start_end_points(df, 18, 23)

    df['Q_OCV_dchg'] = np.nan
    df['Q_OCV_chg'] = np.nan

    df = calculate_Q_OCV(df, start_points_dchg, end_points_dchg, 'Q_OCV_dchg')
    df = calculate_Q_OCV(df, start_points_chg, end_points_chg, 'Q_OCV_chg')

    cols = ['Testtime[s]', 'Voltage[V]', 'Current[A]', 'Temperature[°C]', 'Q_m', 'Q_sum', 'Capacity[Ah]', 'SOH', 'Q_c', 'Q_OCV_chg', 'Q_OCV_dchg', 'EFC', 'SOC_c', 'SOC_m']
    df_OCV_dchg = df[df['OCV_dchg']][cols].interpolate(method='linear', limit_direction='both', on='Testtime[s]')
    df_OCV_chg = df[df['OCV_chg']][cols].interpolate(method='linear', limit_direction='both', on='Testtime[s]')

    segments_OCV_chg = create_segments(df_OCV_chg, start_points_chg['Testtime[s]'].tolist(), end_points_chg['Testtime[s]'].tolist(), 'Q_OCV_chg')
    segments_OCV_dchg = create_segments(df_OCV_dchg, start_points_dchg['Testtime[s]'].tolist(), end_points_dchg['Testtime[s]'].tolist(), 'Q_OCV_dchg')

    return segments_OCV_chg, segments_OCV_dchg, df


def plot_data(df, segments_OCV_chg_list, segments_OCV_dchg_list, SOH_value, cell_name):
    fig = plt.figure(figsize=(15, 9))

    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    
    fig.suptitle(f'Zellendaten: {cell_name}', fontsize=16)  # Gesamtüberschrift mit Zellennamen

    ax0 = plt.subplot(gs[0, :])  # Oberer Plot, der die gesamte Breite einnimmt

    # Code für den ersten Plot (Voltage über Testtime)
    ax0.plot(df['Testtime[s]'], df['Voltage[V]'], color='b')
    ax0.scatter(df[df['OCV_dchg']]['Testtime[s]'], df[df['OCV_dchg']]['Voltage[V]'], color='g', label='OCV_dchg', s=100, zorder=5)
    ax0.scatter(df[df['OCV_chg']]['Testtime[s]'], df[df['OCV_chg']]['Voltage[V]'], color='r', label='OCV_chg', s=100, zorder=5)
    start_points_dchg, end_points_dchg = find_start_end_points(df, 13, 18)
    for start_point in start_points_dchg['Testtime[s]']:
        ax0.axvline(x=start_point, color='g', linestyle='--')
    for end_point in end_points_dchg['Testtime[s]']:
        ax0.axvline(x=end_point, color='g', linestyle='--')
    start_points_chg, end_points_chg = find_start_end_points(df, 18, 23)
    for start_point in start_points_chg['Testtime[s]']:
        ax0.axvline(x=start_point, color='orange', linestyle='--')
    for end_point in end_points_chg['Testtime[s]']:
        ax0.axvline(x=end_point, color='r', linestyle='--')
    ax0.set_ylabel('Voltage [V]')
    ax0.set_title('Voltage über Testtime')  # Titel des oberen Plots
    ax0.legend()

    ax1 = plt.subplot(gs[1, 0])  # Linker unterer Plot
    plot_segments(ax1, segments_OCV_chg_list, 'SOC_OCV', 'Voltage[V]', 'OCV-Charge', cell_name)

    ax2 = plt.subplot(gs[1, 1])  # Rechter unterer Plot
    plot_segments(ax2, segments_OCV_dchg_list, 'SOC_OCV', 'Voltage[V]', 'OCV-Discharge', cell_name)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Platz für die Gesamtüberschrift lassen
    plt.show()


class OCVData:
    def __init__(self):
        self.segments_OCV_chg = []
        self.segments_OCV_dchg = []
        self.count_chg = 0
        self.count_dchg = 0

    def add(self, segment_chg, segment_dchg):
        if segment_chg:
            if self.segments_OCV_chg:
                self.segments_OCV_chg = [(x + y) for x, y in zip(self.segments_OCV_chg, segment_chg)]
            else:
                self.segments_OCV_chg = segment_chg
            self.count_chg += 1
        
        if segment_dchg:
            if self.segments_OCV_dchg:
                self.segments_OCV_dchg = [(x + y) for x, y in zip(self.segments_OCV_dchg, segment_dchg)]
            else:
                self.segments_OCV_dchg = segment_dchg
            self.count_dchg += 1

    def finalize(self):
        if self.count_chg > 0:
            self.segments_OCV_chg = [x / self.count_chg for x in self.segments_OCV_chg]
        
        if self.count_dchg > 0:
            self.segments_OCV_dchg = [x / self.count_dchg for x in self.segments_OCV_dchg]


def merge_and_average(ocv_data_map):
    for soh, ocv_data in ocv_data_map.items():
        ocv_data.segments_OCV_chg = [pd.concat([segment for segment in ocv_data.segments_OCV_chg]).groupby('SOC_OCV').mean()]
        ocv_data.segments_OCV_dchg = [pd.concat([segment for segment in ocv_data.segments_OCV_dchg]).groupby('SOC_OCV').mean()]

def plot_for_given_soh(segments_OCV_chg_list, segments_OCV_dchg_list, SOH_value):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Nehmen Sie an, dass es nur jeweils eine Liste gibt (d.h., die Daten wurden bereits zusammengefasst und gemittelt)
    segment_chg = segments_OCV_chg_list[0]
    segment_dchg = segments_OCV_dchg_list[0]
    
    ax.plot(segment_chg['SOC_OCV'], segment_chg['Voltage[V]'], label='OCV-Charge', color='b')
    ax.plot(segment_dchg['SOC_OCV'], segment_dchg['Voltage[V]'], label='OCV-Discharge', color='r')
    
    # Füllen der Fläche zwischen den beiden OCV-Kurven mit hellem, durchsichtigem Rot
    ax.fill_between(segment_chg['SOC_OCV'], segment_chg['Voltage[V]'], segment_dchg['Voltage[V]'], color='red', alpha=0.2, label='Hysteresis')
    
    ax.set_title(f'OCV Data for SOH: {SOH_value:.2f}%')
    ax.set_xlabel('SOC_OCV')
    ax.set_ylabel('Voltage[V]')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def interpolate_ocv_data(soh, lower_soh, upper_soh, ocv_data_map):
    """Interpolates OCV data for a given SOH between two known SOH values."""
    
    # Get the OCV data for the lower and upper SOH values
    lower_data_chg = ocv_data_map[lower_soh].segments_OCV_chg[0]
    upper_data_chg = ocv_data_map[upper_soh].segments_OCV_chg[0]
    lower_data_dchg = ocv_data_map[lower_soh].segments_OCV_dchg[0]
    upper_data_dchg = ocv_data_map[upper_soh].segments_OCV_dchg[0]
    
    # Interpolate the OCV data for the given SOH
    interpolated_data_chg = lower_data_chg + (upper_data_chg - lower_data_chg) * (soh - lower_soh) / (upper_soh - lower_soh)
    interpolated_data_dchg = lower_data_dchg + (upper_data_dchg - lower_data_dchg) * (soh - lower_soh) / (upper_soh - lower_soh)
    
    return interpolated_data_chg, interpolated_data_dchg

def extrapolate_for_one(ocv_data_map):
    """Extrapolates the OCV data for SOH = 1.0."""
    
    # Get the last two known SOH values
    soh_values = sorted(list(ocv_data_map.keys()))
    second_last_soh = soh_values[-2]
    last_soh = soh_values[-1]
    
    # Use the interpolate_ocv_data function to extrapolate the OCV data for SOH = 1.0
    extrapolated_data_chg, extrapolated_data_dchg = interpolate_ocv_data(1.0, second_last_soh, last_soh, ocv_data_map)
    
    return extrapolated_data_chg, extrapolated_data_dchg

# Hauptverzeichnis
main_dir = r'c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\4_Work_packages\3_State_estimation\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_Dataframes'


ocv_data_map = defaultdict(OCVData)  # Initialisierung des defaultdict

for subdir, _, _ in tqdm(os.walk(main_dir), desc='Reading Files'):
    file_path = os.path.join(subdir, "df_cu.parquet")
    if os.path.exists(file_path):
        segments_OCV_chg, segments_OCV_dchg, df = process_file(file_path)
        cell_name = os.path.basename(subdir)  # Holt den Namen des Subfolders
        
        # Hier wird der gemittelte SOH-Wert für die Segmente berechnet und im Dictionary gespeichert
        for chg_segment, dchg_segment in zip(segments_OCV_chg, segments_OCV_dchg):
            mean_soh = chg_segment['SOH'].mean()  # Oder dchg_segment, sollte dasselbe sein
            ocv_data_map[mean_soh].add([chg_segment], [dchg_segment])

        plot_data(df, segments_OCV_chg, segments_OCV_dchg, df['SOH'].iloc[0], cell_name)

# Finalisieren der Daten in ocv_data_map
for soh, data in ocv_data_map.items():
    data.finalize()

rounded_ocv_data_map = defaultdict(OCVData)

# Daten zu rounded_ocv_data_map hinzufügen
for soh, data in ocv_data_map.items():
    rounded_soh = round(soh, 2)
    rounded_ocv_data_map[rounded_soh].add(data.segments_OCV_chg, data.segments_OCV_dchg)

# Finalisieren der Daten in rounded_ocv_data_map
for soh, data in rounded_ocv_data_map.items():
    data.finalize()

# Interpolate missing OCV data between 0.62 and 0.99
for soh in np.linspace(0.62, 0.99, 38):  # 38 values between 0.62 and 0.99 (inclusive) with a step of 0.01
    if soh not in rounded_ocv_data_map:
        # Find the two neighboring SOH values in rounded_ocv_data_map
        lower_soh = max([key for key in rounded_ocv_data_map.keys() if key < soh])
        upper_soh = min([key for key in rounded_ocv_data_map.keys() if key > soh])
        
        interpolated_data_chg, interpolated_data_dchg = interpolate_ocv_data(soh, lower_soh, upper_soh, rounded_ocv_data_map)
        
        # Create a new OCVData object and add the interpolated data
        new_ocv_data = OCVData()
        new_ocv_data.add([interpolated_data_chg], [interpolated_data_dchg])
        new_ocv_data.finalize()
        
        rounded_ocv_data_map[soh] = new_ocv_data

# Extrapolate the OCV data for SOH = 1.0
extrapolated_data_chg, extrapolated_data_dchg = extrapolate_for_one(rounded_ocv_data_map)
new_ocv_data = OCVData()
new_ocv_data.add([extrapolated_data_chg], [extrapolated_data_dchg])
new_ocv_data.finalize()
rounded_ocv_data_map[1.0] = new_ocv_data

desired_soh = round(0.71, 2)  # Round the SOH value to match the keys in the dictionary
ocv_data = rounded_ocv_data_map.get(desired_soh)

if ocv_data:
    plot_for_given_soh(ocv_data.segments_OCV_chg, ocv_data.segments_OCV_dchg, desired_soh * 100)  # Multiplizieren mit 100, um den SOH in Prozent anzugeben
else:
    print(f"No OCV data found for SOH: {desired_soh}")

save_path = r'c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\4_Work_packages\3_State_estimation\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_Dataframes\Parameters\OCV_data.dill'
with open(save_path, 'wb') as file:
    dill.dump(rounded_ocv_data_map, file)

print(f"Data saved to: {save_path}")


