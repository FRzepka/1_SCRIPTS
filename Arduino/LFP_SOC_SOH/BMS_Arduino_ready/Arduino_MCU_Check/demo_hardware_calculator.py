"""
Demo Script für Arduino Hardware Calculator
===========================================

Demonstriert die Funktionalität des Hardware Calculators mit einem synthetischen LSTM Modell,
da das echte Modell benutzerdefinierte Klassen enthält.
"""

import sys
import os
import torch
import torch.nn as nn

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arduino_hardware_calculator import ArduinoHardwareCalculator

def create_demo_lstm_model():
    """Create a demo LSTM model similar to your architecture"""
    
    class DemoLSTMModel(nn.Module):
        def __init__(self, input_size=4, hidden_size=128, num_layers=2, output_size=1):
            super(DemoLSTMModel, self).__init__()
            
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
            
            self.fc1 = nn.Linear(hidden_size, 64)
            self.fc2 = nn.Linear(64, output_size)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            lstm_out, (hn, cn) = self.lstm(x)
            out = self.fc1(lstm_out[:, -1, :])  # Use last output
            out = self.relu(out)
            out = self.fc2(out)
            out = self.sigmoid(out)
            return out
    
    # Create model
    model = DemoLSTMModel()
    
    # Save model state dict
    demo_model_path = "demo_lstm_model.pth"
    torch.save(model.state_dict(), demo_model_path)
    
    print(f"✅ Created demo LSTM model: {demo_model_path}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Demo Model Info:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Model Size: {total_params * 4 / 1024:.2f} KB")
    
    return demo_model_path

def demo_hardware_analysis():
    """Run hardware analysis on demo model"""
    
    print("🚀 Arduino Hardware Calculator Demo")
    print("=" * 60)
    
    # Create demo model
    demo_model_path = create_demo_lstm_model()
    
    # Initialize calculator
    calculator = ArduinoHardwareCalculator()
    
    try:
        # Run complete analysis
        print(f"\n🔍 Analyzing: {demo_model_path}")
        report = calculator.run_complete_analysis(demo_model_path)
        
        # Print results
        print("\n" + "=" * 60)
        print("📊 HARDWARE REQUIREMENTS SUMMARY")
        print("=" * 60)
        
        flash_req = report['flash_requirements']
        ram_req = report['ram_requirements']
        
        print(f"\n💾 Flash Memory Requirements:")
        print(f"   Model Size:     {flash_req['model_size_kb']:.1f} KB")
        print(f"   C-Code Overhead: {flash_req['overhead_kb']:.1f} KB")
        print(f"   Total Required:  {flash_req['total_flash_kb']:.1f} KB")
        
        print(f"\n🧠 RAM Memory Requirements:")
        print(f"   LSTM States:     {ram_req['lstm_state_bytes']} bytes ({ram_req['lstm_state_bytes']/1024:.1f} KB)")
        print(f"   Working Memory:  {ram_req['working_memory_bytes']} bytes ({ram_req['working_memory_bytes']/1024:.1f} KB)")
        print(f"   System Overhead: {ram_req['overhead_bytes']} bytes ({ram_req['overhead_bytes']/1024:.1f} KB)")
        print(f"   Total Required:  {ram_req['total_ram_bytes']} bytes ({ram_req['total_ram_kb']:.1f} KB)")
        
        print(f"\n⚡ Performance Estimates:")
        for board_name, perf in report['performance_estimates'].items():
            if board_name in ['uno', 'due', 'esp32']:  # Show key boards only
                print(f"   {board_name.upper():8s}: {perf['inference_time_ms']:.1f} ms ({perf['inferences_per_second']:.1f} inf/sec)")
        
        print(f"\n🎯 Compatible Arduino Boards:")
        compatible_boards = []
        for board_name, board_info in report['board_compatibility'].items():
            if board_info['compatible']:
                compatible_boards.append(f"{board_name.upper()} (Flash: {board_info['flash_usage_percent']:.0f}%, RAM: {board_info['ram_usage_percent']:.0f}%)")
        
        if compatible_boards:
            for board in compatible_boards:
                print(f"   ✅ {board}")
        else:
            print("   ❌ No fully compatible boards found!")
        
        print(f"\n⚠️  Problematic Boards:")
        for board_name, board_info in report['board_compatibility'].items():
            if not board_info['compatible']:
                issues = []
                if not board_info['flash_ok']:
                    issues.append(f"Flash overflow ({board_info['flash_usage_percent']:.0f}%)")
                if not board_info['ram_ok']:
                    issues.append(f"RAM overflow ({board_info['ram_usage_percent']:.0f}%)")
                print(f"   ❌ {board_name.upper()}: {', '.join(issues)}")
        
        # Show optimization suggestions for problematic boards
        if report['optimization_suggestions']:
            print(f"\n💡 Optimization Suggestions:")
            for board_name, suggestions in report['optimization_suggestions'].items():
                if suggestions:  # Only show if there are suggestions
                    print(f"\n   {board_name.upper()}:")
                    for suggestion in suggestions:
                        print(f"     {suggestion}")
        
        print(f"\n💾 Full report saved to: arduino_hardware_analysis.json")
        
        # Cleanup demo file
        if os.path.exists(demo_model_path):
            os.remove(demo_model_path)
            print(f"🧹 Cleaned up: {demo_model_path}")
            
        return report
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def show_board_comparison():
    """Show detailed Arduino board comparison"""
    
    calculator = ArduinoHardwareCalculator()
    
    print("\n" + "=" * 90)
    print("📋 DETAILED ARDUINO BOARD COMPARISON")
    print("=" * 90)
    
    print(f"{'Board':<12} | {'Flash':<12} | {'RAM':<15} | {'CPU':<12} | {'Architecture':<12} | {'Use Case':<20}")
    print("-" * 90)
    
    use_cases = {
        'uno': 'Simple prototyping',
        'nano': 'Compact projects', 
        'leonardo': 'USB/HID projects',
        'mega2560': 'Complex projects',
        'due': 'High performance',
        'esp32': 'WiFi/BT projects',
        'teensy40': 'Audio/DSP projects'
    }
    
    for board_name, specs in calculator.arduino_boards.items():
        flash_str = f"{specs['flash_kb']} KB"
        ram_str = f"{specs['ram_bytes']} bytes ({specs['ram_bytes']/1024:.0f}KB)" if specs['ram_bytes'] >= 1024 else f"{specs['ram_bytes']} bytes"
        cpu_str = f"{specs['cpu_mhz']} MHz"
        use_case = use_cases.get(board_name, 'General purpose')
        
        print(f"{board_name:<12} | {flash_str:<12} | {ram_str:<15} | {cpu_str:<12} | {specs['architecture']:<12} | {use_case:<20}")

if __name__ == "__main__":
    demo_hardware_analysis()
    show_board_comparison()
