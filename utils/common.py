def plot_quantization_techniques(ax, models, data, title, log=True):
    # Plot lines for each quantization technique
    for technique, values in data.items():
        ax.plot(models, values, marker="o", label=technique)

    # Set labels and title
    ax.set_xlabel("Model")
    ax.set_ylabel("PPL (log-scale)")
    ax.set_title(title)

    # Set x-tick labels
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45)

    # Add a legend
    ax.legend()

    # Log-scale
    if log:
        ax.set_yscale("log")
