import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("timerange_result.csv")
print(df)

# Step 1: Define Subject Order (if needed, match previous sorting)
subject_order = df["subject"].unique()
type_order = ["ACO", "Random", "Fixed time"]

# Step 2: Define Colors for Each Type (Research Paper Style)
type_colors = {"ACO": "#377eb8", "Random": "#ff7f00", "Fixed time": "#4daf4a"}

# Set bar width and spacing
bar_width = 0.25
subject_spacing = 1.5  # More spacing between subjects
y_positions = []

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

freq = 1
freq_band="8-13"
if freq!=0:
    freq_band="13-30"

# Loop through subjects
current_y = 0  # Start y position
for subject in subject_order:
    subset = df[df["subject"] == subject]
    
    for j, type_ in enumerate(type_order):
        type_data = subset[subset["type"] == type_]
        if not type_data.empty:
            start_time = type_data["start_time"].values[freq]
            duration = type_data["duration"].values[freq]
            ax.barh(
                current_y + j * bar_width, duration,
                left=start_time, color=type_colors[type_], label=type_ if current_y == 0 else "",
                height=bar_width, alpha=0.9
            )

    # Store y-position for subject label and move to the next subject
    y_positions.append(current_y + bar_width)
    current_y += subject_spacing  # More space between subjects

# Set y-ticks to align with subject groups
ax.set_ylabel("Subjects", fontsize=12, labelpad=15)
ax.set_yticks(y_positions)
ax.set_yticklabels(subject_order, fontsize=10)

# Formatting
ax.set_xlabel("Time (ms)", fontsize=12, labelpad=15)
ax.legend(loc="upper right", fontsize=10, frameon=False)
ax.grid(axis="x", linestyle="--", alpha=0.6)
plt.title(f"Start Time and Duration for Each Subject & Type for Range {freq_band} Hz", fontsize=14)
fig.savefig(f'result2_timesegmentrange_{freq_band}.png', dpi=300)
plt.show()