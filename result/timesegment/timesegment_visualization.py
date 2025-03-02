import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches  # For custom legend

# Load the dataset
df = pd.read_csv("timerange_result.csv")

# Set the desired frequency to filter (Change this to 1 when needed)
desired_freq = 1  

# Filter data by the selected frequency
df = df[df["frequency"] == desired_freq]

# Define Subject Order (Keep the original subject order)
subject_order = df["subject"].unique()
type_order = ["ACO", "Random", "Fixed time"]

# Define Colors for Each Type
type_colors = {"ACO": "#a5a5a5", "Random": "#ed7d31", "Fixed time": "#4472c4"}

# Set bar width and spacing
bar_width = 0.75  
subject_spacing = 4  # More space between subjects
y_positions = []

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Frequency band label for title
freq_band = "8-13" if desired_freq == 1 else "13-30"

current_y = 0  
for subject in subject_order:
    subset = df[df["subject"] == subject]  # Filter current subject's data

    # Compute middle position for bars and labels
    total_bars = len(type_order)  # Number of bars per subject
    middle_y = current_y + (total_bars * bar_width) / 2  # Center of the group

    for j, type_ in enumerate(type_order):
        type_data = subset[subset["type"] == type_]  # Filter by type
        
        if not type_data.empty:
            start_time = float(type_data["start_time"].values[0])  
            duration = float(type_data["duration"].values[0])  

            y_position = middle_y + (j - 1) * bar_width  # Centered bars

            ax.barh(y_position, duration, left=start_time, 
                    color=type_colors[type_], height=bar_width, alpha=0.9)

    # Save subject label position (Centered)
    y_positions.append(middle_y)

    # Add a horizontal grid line **below** the group
    ax.axhline(y=current_y + subject_spacing - (bar_width / 2), color="gray", linestyle="--", alpha=0.5)

    # Move to next subject position **AFTER bars are placed**
    current_y += subject_spacing  

# Set y-ticks to align with subject labels
ax.set_ylabel("Subjects", fontsize=12, labelpad=15)
ax.set_yticks(y_positions)
ax.set_yticklabels(subject_order, fontsize=10)
ax.set_xlim(left=0, right=4000)  # Adjust x-axis limits if needed

# Create Custom Legend with Background
legend_patches = [mpatches.Patch(color=type_colors[t], label=t) for t in type_order]
ax.legend(handles=legend_patches, loc="upper right", fontsize=10, frameon=True, facecolor="white")

# Formatting
ax.set_xlabel("Time (ms)", fontsize=12, labelpad=15)
ax.grid(axis="x", linestyle="--", alpha=0.6)  # Keep vertical grid for time
plt.title(f"Start Time and Duration for Each Subject & Type for Range {freq_band} Hz", fontsize=14)
fig.savefig(f'result2_timesegmentrange_{freq_band}.png', dpi=300)
plt.show()
