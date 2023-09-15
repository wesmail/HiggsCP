import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV files into DataFrames
df_ttbb = pd.read_csv('../kinematics_polarisation_ttbb.csv')
df_ttH = pd.read_csv('../kinematics_polarisation_ttH.csv')
df_ttA = pd.read_csv('../kinematics_polarisation_ttA.csv')

# Get the column names (assuming all files have the same columns)
columns = df_ttbb.columns

# Function to plot a canvas of histograms
def plot_canvas(start, end, columns, data_list, labels):
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.flatten()
    for i in range(start, end):
        ax = axes[i-start]
        for data, label in zip(data_list, labels):
            ax.hist(data[columns[i]], bins=20, alpha=0.3, label=label, density=True)
        ax.set_xlabel(columns[i])
        ax.set_ylabel('Counts')
        ax.legend()
    fig.tight_layout()

# Data and labels
data_list = [df_ttbb, df_ttH, df_ttA]
labels = ['ttbb', 'ttH', 'ttA']

# Plot the first 12 columns
plot_canvas(0, 12, columns, data_list, labels)

# Plot the next 12 columns
plot_canvas(12, 24, columns, data_list, labels)

# Plot the remaining 11 columns
fig, axes = plt.subplots(3, 4, figsize=(15, 10))
axes = axes.flatten()
for i in range(24, 35):
    ax = axes[i-24]
    for data, label in zip(data_list, labels):
        ax.hist(data[columns[i]], bins=20, alpha=0.3, label=label, density=True)
    ax.set_xlabel(columns[i])
    ax.set_ylabel('Density')
    ax.legend()

# Remove the last unused subplot
axes[-1].axis('off')
fig.tight_layout()

# Show all the plots
plt.show()
