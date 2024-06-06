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
    "RTN": [7.01, 5.88, 4.87, 4.24],
    "GPTQ": [6.55, 5.62, 4.80, 4.17],
    "AWQ": [6.46, 5.51, 4.63, 3.99],
    "OmniQuant": [6.15, 5.44, 4.56, 3.94],
    "SliM-LLM": [6.40, 5.48, 4.61, 3.99],
}

# Data for 2-xxB models
models_2 = ["2-7B", "2-13B", "2-70B"]
data_2 = {
    "RTN": [6.66, 5.51, 3.97],
    "GPTQ": [6.29, 5.42, 3.85],
    "AWQ": [6.24, 5.32, np.nan],
    "OmniQuant": [6.03, 5.28, 3.78],
    "BitDistiller": [5.97, 5.20, np.nan],  # No data in the table
    "LLM-QAT": [6.02, 5.32, np.nan],  # No data in the table
    "SliM-LLM": [6.24, 5.26, 3.67],
}

# Create a figure and axes for side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot quantization techniques for 1-xxB models
plot_quantization_techniques(
    ax1, models_1, data_1, "Quantization using 3-bit (LLama-1)", log=False
)

# Plot quantization techniques for 2-xxB models
plot_quantization_techniques(
    ax2,
    models_2,
    data_2,
    "Quantization using 3-bit (LLama-2)",
    log=False,
)

# Adjust spacing and display the plots
plt.tight_layout()
plt.savefig("./asset/quantization_3bit.png", dpi=300)
