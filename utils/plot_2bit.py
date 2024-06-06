import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from common import plot_quantization_techniques

# Set the style
sns.set_style("whitegrid")
sns.set_theme(context="notebook", style="whitegrid")


# Data for 1-xxB models
models_1 = ["1-7B", "1-13B", "1-30B", "1-65B"]
data_1 = {
    # "RTN": [1.9e3, 781.20, 68.04, 15.08],
    "GPTQ": [44.01, 15.60, 10.92, 9.51],
    # "AWQ": [2.6e5, 2.8e5, 2.4e5, 7.4e4],
    "OmniQuant": [9.72, 7.93, 7.12, 5.95],
    "QuIP": [29.74, 12.48, 11.57, 7.83],
    "PB-LLM": [24.61, 17.73, 12.65, 7.85],
}

# Data for 2-xxB models
models_2 = ["2-7B", "2-13B", "2-70B"]
data_2 = {
    # "RTN": [4.2e3, 122.08, 27.27],
    "GPTQ": [36.77, 28.14, np.nan],
    # "AWQ": [2.2e5, 1.2e5, np.nan],
    "OmniQuant": [11.06, 8.26, 6.55],
    "BitDistiller": [8.08, 6.78, np.nan],
    "LLM-QAT": [9.30, 7.80, np.nan],
    "PB-LLM": [25.37, 49.81, np.nan],
    "SliM-LLM": [16.01, 9.41, 6.28],
    "QuIP": [39.73, 13.48, 6.64],
}

# Create a figure and axes for side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot quantization techniques for 1-xxB models
plot_quantization_techniques(
    ax1, models_1, data_1, "Quantization using 2-bit (LLama-1)"
)

# Plot quantization techniques for 2-xxB models
plot_quantization_techniques(
    ax2, models_2, data_2, "Quantization using 2-bit (LLama-2)"
)

# Adjust spacing and display the plots
plt.tight_layout()
plt.savefig("./asset/quantization_2bit.png", dpi=300)
