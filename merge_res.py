import os
import glob
import re
import pandas as pd
import sys
from config.modelconfig import ModelConfig

def combine_csv_files(config, combined_filename, rand_state):
    # Identify all the CSV files matching the pattern
    csv_files = glob.glob(f"{config.analysisDir}/{config.runId}_rand{rand_state}_*.csv")
    csv_files_sorted = sorted(csv_files, key=lambda x: float(re.search(r'.*?missing(\d+)%.*?', x).group(1)))
    
    if not csv_files_sorted:
        print(f"No CSV files found for random state {rand_state}")
        return
        
    combined_df = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files_sorted])
    combined_df.to_csv(combined_filename, index=False, sep=',')
    print(f"Combined CSV saved to {combined_filename}")

# Get the config file path from command line
config_file = sys.argv[1]
config = ModelConfig.from_yaml(config_file)

# Random states used in the benchmark
random_states = [0, 42, 1024]

for rand_state in random_states:
    output_file = os.path.join(config.analysisDir, f"{config.runId}_{rand_state}.csv")
    combine_csv_files(config, output_file, rand_state)
    print(f"Combined results for random state {rand_state}")
