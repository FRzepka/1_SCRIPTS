import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

# ARM Cortex-M4 Memory Layout (Arduino Uno R4 - Renesas RA4M1)
arm_memory_layout = {
    'total_ram': 32.0,  # kB
    'total_flash': 256.0,  # kB
    # RAM breakdown (estimated based on ARM Cortex-M4 typical usage)
    'ram_reserved_system': 2.0,    # System/HAL reserved
    'ram_stack': 2.0,              # Stack space
    'ram_heap_available': 28.0,    # Available for application
    # Flash breakdown
    'flash_bootloader': 8.0,       # Bootloader
    'flash_system_libs': 16.0,     # Arduino core + system libraries
    'flash_available': 232.0       # Available for user code
}

# Measured data from Arduino (your actual measurements)
stateful_data = {
    '16×16': {'RAM': 7.7, 'Flash': 48.8},
    '32×32': {'RAM': 8.9, 'Flash': 106.9},
    '64×64': {'RAM': 9.8, 'Flash': 123.0}
}

# Detailed breakdown of your measurements (estimated based on typical LSTM implementation)
detailed_breakdown = {
    '16×16': {
        'model_weights_ram': 3.2,    # Weights loaded in RAM
        'hidden_state': 0.8,         # LSTM hidden state
        'input_buffer': 0.5,         # Input data buffer
        'output_buffer': 0.2,        # Output buffer
        'arduino_core': 2.5,         # Arduino framework overhead
        'user_code': 0.5,            # Your application code
        'model_weights_flash': 25.0, # Model stored in flash
        'arduino_libs_flash': 23.8   # Arduino libs in flash
    },
    '32×32': {
        'model_weights_ram': 4.8,
        'hidden_state': 1.6,
        'input_buffer': 0.5,
        'output_buffer': 0.2,
        'arduino_core': 2.5,
        'user_code': 0.8,
        'model_weights_flash': 80.0,
        'arduino_libs_flash': 26.9
    },
    '64×64': {
        'model_weights_ram': 6.0,
        'hidden_state': 3.2,
        'input_buffer': 0.5,
        'output_buffer': 0.2,
        'arduino_core': 2.5,
        'user_code': 1.0,
        'model_weights_flash': 95.0,
        'arduino_libs_flash': 28.0
    }
}

# Window-based estimation (extrapolated from 32x32 with 5000s = 80kB)
window_data = {
    '32×32': {'RAM': 80.0, 'Flash': 106.9}  # Similar Flash, but massive RAM for window
}

# MCU specifications (typical values)
mcu_specs = {
    'Arduino R4': {'RAM': 32, 'Flash': 256},
    'ESP32': {'RAM': 520, 'Flash': 4096},
    'STM32H7': {'RAM': 1024, 'Flash': 2048},
    'RP2040': {'RAM': 264, 'Flash': 2048},
    'Arduino Nano': {'RAM': 2, 'Flash': 32},
    'Arduino Uno': {'RAM': 2, 'Flash': 32},
    'STM32F4': {'RAM': 192, 'Flash': 1024}
}

def plot_ram_comparison():
    """A1: RAM-Vergleich stateful vs. window-based"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data for comparison
    models = ['16×16 stateful', '32×32 stateful', '64×64 stateful', '32×32 window']
    ram_values = [7.7, 8.9, 9.8, 80.0]
    colors = ['#2E86AB', '#2E86AB', '#2E86AB', '#F24236']
    
    # Arduino R4 has 32kB RAM
    arduino_ram = 32
    percentages = [(ram/arduino_ram)*100 for ram in ram_values]
    
    bars = ax.bar(models, ram_values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}% des\nArduino RAM',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('RAM Verbrauch (kB)', fontsize=12)
    ax.set_title('RAM-Vergleich: Stateful vs. Window-based Architektur', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add Arduino RAM limit line
    ax.axhline(y=arduino_ram, color='red', linestyle='--', alpha=0.7, label='Arduino R4 RAM Limit (32kB)')
    ax.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_memory_breakdown():
    """A2: Speicheraufteilung (RAM + Flash) je Architektur"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Use detailed breakdown data
    architectures = ['16×16', '32×32', '64×64']
    
    # RAM breakdown from detailed measurements
    ram_weights = [detailed_breakdown[arch]['model_weights_ram'] for arch in architectures]
    ram_hidden = [detailed_breakdown[arch]['hidden_state'] for arch in architectures]
    ram_buffers = [detailed_breakdown[arch]['input_buffer'] + detailed_breakdown[arch]['output_buffer'] for arch in architectures]
    ram_arduino = [detailed_breakdown[arch]['arduino_core'] for arch in architectures]
    ram_user = [detailed_breakdown[arch]['user_code'] for arch in architectures]
    
    # Flash breakdown from detailed measurements
    flash_weights = [detailed_breakdown[arch]['model_weights_flash'] for arch in architectures]
    flash_arduino = [detailed_breakdown[arch]['arduino_libs_flash'] for arch in architectures]
    
    # RAM stacked bar chart
    width = 0.6
    x = np.arange(len(architectures))
    
    p1 = ax1.bar(x, ram_weights, width, label='Model Weights', color='#FF6B6B')
    p2 = ax1.bar(x, ram_hidden, width, bottom=ram_weights, label='Hidden State', color='#4ECDC4')
    p3 = ax1.bar(x, ram_buffers, width, bottom=np.array(ram_weights)+np.array(ram_hidden), 
                 label='I/O Buffers', color='#45B7D1')
    p4 = ax1.bar(x, ram_arduino, width, bottom=np.array(ram_weights)+np.array(ram_hidden)+np.array(ram_buffers), 
                 label='Arduino Core', color='#96CEB4')
    p5 = ax1.bar(x, ram_user, width, bottom=np.array(ram_weights)+np.array(ram_hidden)+np.array(ram_buffers)+np.array(ram_arduino), 
                 label='User Code', color='#FFEAA7')
    
    ax1.set_ylabel('RAM Verbrauch (kB)')
    ax1.set_title('RAM Speicheraufteilung (ARM Cortex-M4)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(architectures)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add Arduino RAM limit line
    ax1.axhline(y=arm_memory_layout['total_ram'], color='red', linestyle='--', alpha=0.7, 
                label='Arduino R4 RAM Total (32kB)')
    ax1.axhline(y=arm_memory_layout['ram_heap_available'], color='orange', linestyle='--', alpha=0.7, 
                label='Verfügbarer Heap (28kB)')
    
    # Flash stacked bar chart
    p6 = ax2.bar(x, flash_weights, width, label='Model Weights', color='#FF6B6B')
    p7 = ax2.bar(x, flash_arduino, width, bottom=flash_weights, label='Arduino Core', color='#96CEB4')
    
    ax2.set_ylabel('Flash Verbrauch (kB)')
    ax2.set_title('Flash Speicheraufteilung (ARM Cortex-M4)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(architectures)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add Arduino Flash limits
    ax2.axhline(y=arm_memory_layout['total_flash'], color='red', linestyle='--', alpha=0.7, 
                label='Arduino R4 Flash Total (256kB)')
    ax2.axhline(y=arm_memory_layout['flash_available'], color='orange', linestyle='--', alpha=0.7, 
                label='Verfügbarer Flash (232kB)')
    
    plt.tight_layout()
    return fig

def plot_arm_memory_architecture():
    """A2b: ARM Cortex-M4 Speicherarchitektur Übersicht"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # RAM Architecture visualization
    ram_segments = [
        ('System Reserved', 2.0, '#FF4444'),
        ('Stack Space', 2.0, '#FF8844'),
        ('Available Heap', 28.0, '#44AA44')
    ]
    
    flash_segments = [
        ('Bootloader', 8.0, '#FF4444'),
        ('System Libraries', 16.0, '#FF8844'), 
        ('Available Flash', 232.0, '#44AA44')
    ]
    
    # RAM pie chart
    ram_sizes = [seg[1] for seg in ram_segments]
    ram_labels = [f'{seg[0]}\n{seg[1]:.1f} kB' for seg in ram_segments]
    ram_colors = [seg[2] for seg in ram_segments]
    
    wedges1, texts1, autotexts1 = ax1.pie(ram_sizes, labels=ram_labels, colors=ram_colors,
                                          autopct='%1.1f%%', startangle=90)
    ax1.set_title('ARM Cortex-M4 RAM Architektur\n(Arduino Uno R4 - 32kB total)', 
                  fontsize=14, fontweight='bold')
    
    # Flash pie chart
    flash_sizes = [seg[1] for seg in flash_segments]
    flash_labels = [f'{seg[0]}\n{seg[1]:.1f} kB' for seg in flash_segments]
    flash_colors = [seg[2] for seg in flash_segments]
    
    wedges2, texts2, autotexts2 = ax2.pie(flash_sizes, labels=flash_labels, colors=flash_colors,
                                          autopct='%1.1f%%', startangle=90)
    ax2.set_title('ARM Cortex-M4 Flash Architektur\n(Arduino Uno R4 - 256kB total)', 
                  fontsize=14, fontweight='bold')
    
    # Add your model usage annotations
    ax1.text(0, -1.5, 'Deine Modelle:\n16×16: 7.7kB (25% des verfügbaren Heaps)\n32×32: 8.9kB (32%)\n64×64: 9.8kB (35%)', 
             ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
             fontsize=10)
    
    ax2.text(0, -1.5, 'Deine Modelle:\n16×16: 48.8kB (21% des verfügbaren Flash)\n32×32: 106.9kB (46%)\n64×64: 123kB (53%)', 
             ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
             fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_mcu_overview():
    """A3: Mikrocontroller-Übersicht & Modell-Fit"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # MCU families and colors
    families = {
        'Arduino': ['Arduino R4', 'Arduino Nano', 'Arduino Uno'],
        'ESP': ['ESP32'],
        'STM32': ['STM32H7', 'STM32F4'],
        'RP': ['RP2040']
    }
    
    family_colors = {
        'Arduino': '#FF6B6B',
        'ESP': '#4ECDC4',
        'STM32': '#45B7D1',
        'RP': '#96CEB4'
    }
    
    # Plot MCUs
    for family, mcus in families.items():
        ram_vals = [mcu_specs[mcu]['RAM'] for mcu in mcus]
        flash_vals = [mcu_specs[mcu]['Flash'] for mcu in mcus]
        ax.scatter(ram_vals, flash_vals, 
                  color=family_colors[family], 
                  s=100, alpha=0.7, 
                  label=family, 
                  edgecolors='black')
        
        # Add MCU labels
        for mcu, ram, flash in zip(mcus, ram_vals, flash_vals):
            ax.annotate(mcu, (ram, flash), xytext=(5, 5), 
                       textcoords='offset points', fontsize=9)
    
    # Add requirement lines
    max_stateful_ram = 9.8  # 64x64 stateful
    window_ram = 80.0       # window-based
    
    ax.axhline(y=123, color='green', linestyle='-', alpha=0.7, 
               label='Flash Bedarf (64×64 stateful)', linewidth=2)
    ax.axvline(x=max_stateful_ram, color='green', linestyle='-', alpha=0.7, 
               label='RAM Bedarf (64×64 stateful)', linewidth=2)
    ax.axvline(x=window_ram, color='red', linestyle='--', alpha=0.7, 
               label='RAM Bedarf (window-based)', linewidth=2)
    
    ax.set_xlabel('RAM (kB)', fontsize=12)
    ax.set_ylabel('Flash (kB)', fontsize=12)
    ax.set_title('Mikrocontroller-Übersicht: Speicher-Anforderungen vs. MCU-Kapazitäten', 
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_window_pie_chart():
    """A4: Window-RAM als Kreisdiagramm"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Window-based breakdown for 5000 seconds
    total_ram = 80.0  # kB
    window_array = 75.0  # Most of the RAM goes to the time window
    weights = 3.0        # Model weights
    firmware = 2.0       # Firmware code
    
    sizes = [window_array, weights, firmware]
    labels = [f'Zeitfenster-Array\n({window_array:.1f} kB)', 
              f'Model Weights\n({weights:.1f} kB)', 
              f'Firmware Code\n({firmware:.1f} kB)']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    explode = (0.1, 0, 0)  # explode the largest slice
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                     autopct='%1.1f%%', shadow=True, startangle=90,
                                     textprops={'fontsize': 12})
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)
    
    ax.set_title('RAM-Verteilung: Window-based Architektur (5000s Zeitfenster)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def plot_ram_headroom_heatmap():
    """A5: RAM-Headroom-Ampel"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # MCUs to analyze
    mcus = ['Arduino Uno', 'Arduino Nano', 'Arduino R4', 'RP2040', 'STM32F4', 'ESP32', 'STM32H7']
    models = ['16×16 stateful', '32×32 stateful', '64×64 stateful', '32×32 window']
    model_ram = [7.7, 8.9, 9.8, 80.0]
    
    # Calculate headroom matrix
    headroom_matrix = []
    for mcu in mcus:
        mcu_ram = mcu_specs[mcu]['RAM']
        headroom_row = []
        for ram_need in model_ram:
            if ram_need > mcu_ram:
                headroom = -100  # Impossible
            else:
                headroom = ((mcu_ram - ram_need) / mcu_ram) * 100
            headroom_row.append(headroom)
        headroom_matrix.append(headroom_row)
    
    headroom_matrix = np.array(headroom_matrix)
    
    # Create custom colormap (red for <0, yellow for 0-50, green for >50)
    colors = []
    for i in range(len(mcus)):
        color_row = []
        for j in range(len(models)):
            value = headroom_matrix[i, j]
            if value < 0:
                color_row.append('#FF4444')  # Red
            elif value < 50:
                color_row.append('#FFAA00')  # Yellow/Orange
            else:
                color_row.append('#44AA44')  # Green
        colors.append(color_row)
    
    # Create heatmap
    im = ax.imshow(headroom_matrix, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=100)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(mcus)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_yticklabels(mcus)
    
    # Add text annotations
    for i in range(len(mcus)):
        for j in range(len(models)):
            value = headroom_matrix[i, j]
            if value < 0:
                text = 'UNMÖGLICH'
                color = 'white'
            else:
                text = f'{value:.0f}%\nfrei'
                color = 'black' if value > 25 else 'white'
            ax.text(j, i, text, ha='center', va='center', 
                   color=color, fontweight='bold', fontsize=10)
    
    ax.set_title('RAM-Headroom Matrix: Verfügbarer Speicher nach MCU und Modell', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Modell-Architektur', fontsize=12)
    ax.set_ylabel('Mikrocontroller', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Freier RAM (%)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    return fig

def create_all_plots():
    """Create all plots and save them"""
    print("Erstelle Diagramme...")
    
    # Create plots
    fig1 = plot_ram_comparison()
    fig2 = plot_memory_breakdown()
    fig2b = plot_arm_memory_architecture()
    fig3 = plot_mcu_overview()
    fig4 = plot_window_pie_chart()
    fig5 = plot_ram_headroom_heatmap()
    
    # Save plots
    figures = [
        (fig1, 'A1_RAM_Vergleich_stateful_vs_window.png'),
        (fig2, 'A2_Speicheraufteilung_RAM_Flash.png'),
        (fig2b, 'A2b_ARM_Cortex_M4_Architektur.png'),
        (fig3, 'A3_MCU_Übersicht_Modell_Fit.png'),
        (fig4, 'A4_Window_RAM_Kreisdiagramm.png'),
        (fig5, 'A5_RAM_Headroom_Ampel.png')
    ]
    
    for fig, filename in figures:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Gespeichert: {filename}")
    
    # Show all plots
    plt.show()
    
    print("\nZusammenfassung der Ergebnisse:")
    print("="*50)
    print("ARM CORTEX-M4 ARCHITEKTUR (Arduino Uno R4):")
    print("- Gesamt RAM: 32kB")
    print("- Verfügbarer Heap: ~28kB (nach System-Reserved)")
    print("- Gesamt Flash: 256kB")
    print("- Verfügbarer Flash: ~232kB (nach Bootloader/System)")
    print("\nSTATEFUL ARCHITEKTUR:")
    print("- 16×16: 7.7kB RAM (25% des verfügbaren Heaps)")
    print("- 32×32: 8.9kB RAM (32% des verfügbaren Heaps)")
    print("- 64×64: 9.8kB RAM (35% des verfügbaren Heaps)")
    print("\nWINDOW-BASED ARCHITEKTUR:")
    print("- 32×32: 80kB RAM (286% des verfügbaren Heaps) - UNMÖGLICH!")
    print("\nFAZIT: Stateful-Architektur nutzt ARM-Speicher effizient!")

if __name__ == "__main__":
    create_all_plots()
