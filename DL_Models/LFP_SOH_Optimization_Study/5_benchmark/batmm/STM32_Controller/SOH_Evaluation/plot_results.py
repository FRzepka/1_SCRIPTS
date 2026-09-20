import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python plot_results.py <data.csv>")
    sys.exit(1)

    csv_file = sys.argv[1]
    preds = pd.read_csv(csv_file)

plt.figure(figsize=(10,5))
plt.plot(preds["EFC"], preds["predicted_SOH"], marker='.', linewidth=1)
plt.xlabel("Equivalent Full Cycles (EFC)")
plt.ylabel("Predicted SOH")
plt.title("Predicted SOH vs. Cycle Count")
plt.grid(True)
plt.tight_layout()
plt.show()

