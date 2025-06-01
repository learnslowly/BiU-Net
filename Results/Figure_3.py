import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Load the CSV file into a DataFrame
file_path = 'LOS_demographic.csv'
phenotype = pd.read_csv(file_path)

phenotype.isna().sum()
# Check for duplicated 'Los_ID' in the DataFrame
duplicates = phenotype['Los_ID'].duplicated(keep=False)
duplicated = phenotype.loc[duplicates, 'Los_ID'].values
duplicated
phenotype = phenotype.drop_duplicates(subset='Los_ID', keep='first')
phenotype = phenotype.set_index('Los_ID')
phenotype

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({
    'figure.dpi': 300,  # Increase DPI for high-resolution plots
    'savefig.dpi': 300,  # High resolution for saved figures    
    'font.size': 18,          # Default text size
    'axes.titlesize': 18,     # Axes title font size
    'axes.labelsize': 16,     # Axes labels font size
    'xtick.labelsize': 16,    # X tick labels font size
    'ytick.labelsize': 16,    # Y tick labels font size
    'legend.fontsize': 16,    # Legend font size
    'figure.titlesize': 22,   # Figure title font size
    'axes.grid': True,        # Enable grid
    'grid.linestyle': '-',    # Set grid line style
    'grid.alpha': 0.5,        # Set grid transparency
    'grid.color': 'gray',     # Set grid color
    'grid.linewidth': 0.5,    # Set grid line width
})
def analyze_phenotype(phenotype):
    print("Population Distribution Statistics:")
    
    # Handle missing data in key columns
    phenotype = phenotype.dropna(subset=["Age", "Gender", "Race"])
    
    # Combine smaller racial categories into "Other"
    small_races = ["Asian", "Hispanic", "Other", "American Indian/Native American"]
    phenotype['Race'] = phenotype['Race'].replace(small_races, "other")
    
    # Age statistics
    print("\nAge Statistics:")
    print(phenotype['Age'].describe())
    
    # Race and Gender combination statistics
    print("\nRace and Gender Distribution:")
    combined_dist = phenotype.groupby(['Race', 'Gender']).size().unstack(fill_value=0)
    print(combined_dist)
    
    # Calculate proportions for the height of bars (race distribution)
    race_total = combined_dist.sum(axis=1)
    race_proportions = race_total / race_total.sum() * 100  # Proportions of each race
    
    # Calculate the split for genders within each race
    gender_proportions_within_race = combined_dist.div(race_total, axis=0) * 100  # Percent split within each race
    
    # Visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Age histogram
    sns.histplot(phenotype['Age'], ax=ax1, color=sns.color_palette("Paired")[0], edgecolor='black', alpha=0.75)
    ax1.set_title('Age Distribution')
    ax1.set_xlabel('Age', fontweight='bold')
    ax1.set_ylabel('Count')
    
    # Stacked bar chart for Race and Gender
    bottom = np.zeros(len(race_proportions))  # Track the bottom of each bar for stacking
    race_labels = race_proportions.index
    colors = sns.color_palette('Paired', n_colors=len(gender_proportions_within_race.columns))

    for i, gender in enumerate(gender_proportions_within_race.columns):
        proportions = gender_proportions_within_race[gender] / 100 * race_proportions  # Scale by race proportion
        ax2.bar(
            race_labels, 
            proportions, 
            bottom=bottom, 
            label=gender, 
            color=colors[i]
        )
        bottom += proportions  # Update bottom for stacking
    
    ax2.set_title('Race and Sex Distribution')
    ax2.set_xlabel('Race', fontweight='bold')
    ax2.set_ylabel('Percentage')
    ax2.legend(title='Sex', loc='upper right')
    
    plt.tight_layout()
    plt.savefig("../Figures/Figure_3.pdf")
    plt.show()
    
# Call the function
analyze_phenotype(phenotype)