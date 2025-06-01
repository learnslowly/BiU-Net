#!/usr/bin/env python3
import os
import yaml
import argparse
import re
from pathlib import Path
import itertools

def parse_args():
    parser = argparse.ArgumentParser(description='Generate config and job files for grid search')
    parser.add_argument('--template', type=str, default="configs/HLA_chr6_ALL_seg128_overlap64_0.yaml", help='Template YAML config file')
    parser.add_argument('--job-prefix', type=str, default='train', help='Prefix for job files')
    parser.add_argument('--dataset', type=str, default="hla", help='Dataset name for job naming')
    return parser.parse_args()

def load_yaml_template(template_path):
    """Load YAML template and preserve the original text"""
    with open(template_path, 'r') as f:
        template_text = f.read()

    # Load as YAML to get the data structure
    with open(template_path, 'r') as f:
        template_data = yaml.safe_load(f)

    return template_data, template_text

def generate_config_with_preserved_format(template_text, param_values, run_id):
    """Generate a new config while preserving the original format"""
    new_text = template_text

    # Remove the grid_search section if present
    grid_search_pattern = re.compile(r'# Grid search configuration.*?(?=\n\n|\n*$)', re.DOTALL)
    new_text = re.sub(grid_search_pattern, '', new_text)

    # Also remove slurm_settings section if present
    slurm_pattern = re.compile(r'# SLURM settings.*?(?=\n\n|\n*$)', re.DOTALL)
    new_text = re.sub(slurm_pattern, '', new_text)

    # Replace runId
    run_id_pattern = re.compile(r'runId:.*?\n')
    new_text = re.sub(run_id_pattern, f'runId: {run_id}\n', new_text)

    # Replace each parameter
    for param_name, param_value in param_values.items():
        # Create a pattern that matches the parameter and its value, handling various formats
        param_pattern = re.compile(f'{param_name}:.*?\n')

        # Format the value appropriately
        if isinstance(param_value, str):
            formatted_value = f"{param_name}: {param_value}\n"
        elif isinstance(param_value, float):
            # Keep the exact format for floats
            formatted_value = f"{param_name}: {param_value}\n"
        elif isinstance(param_value, int):
            formatted_value = f"{param_name}: {param_value}\n"
        elif isinstance(param_value, list):
            formatted_value = f"{param_name}: {param_value}\n"
        else:
            formatted_value = f"{param_name}: {param_value}\n"

        # Replace in the text
        new_text = re.sub(param_pattern, formatted_value, new_text)

    return new_text

def generate_batch_script(config_path, job_num, dataset, slurm_settings):
    """Generate batch script with SLURM settings from template"""
    # Default SLURM settings
    default_settings = {
        'partition': 'your_hpc_nodes_partition',
        'account': 'your_hpc_account',
        'nodes': 3,
        'ntasks': 6,
        'ntasks-per-node': 2,
        'gpus-per-task': 1,
        'gres': 'gpu:2',
        'time': '2-23:00:00'
    }

    # Override defaults with settings from YAML
    for key, value in slurm_settings.items():
        default_settings[key] = value

    # Generate SLURM directives
    slurm_directives = [f"#SBATCH --output=job/{dataset}_{job_num}.log",
                        "##SBATCH --open-mode=append"]

    # Add all settings as SLURM directives
    for key, value in default_settings.items():
        slurm_directives.append(f"#SBATCH --{key}={value}")

    # Generate the full batch script
    batch_template = f"""#!/bin/bash
{chr(10).join(slurm_directives)}
#module load cuda/12.2.1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/u19

# GPU Assignment
# Export master address and port for distributed communication
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12345
starttime=`date +'%Y-%m-%d %H:%M:%S'`
echo "Node: $SLURM_NODELIST"
echo "Local Rank: $SLURM_LOCALID"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
srun --wait 0 python train.py --configFile {config_path}
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s)
end_seconds=$(date --date="$endtime" +%s)
total_seconds=$((end_seconds - start_seconds))
hours=$((total_seconds / 3600))
minutes=$(((total_seconds % 3600) / 60))
echo "Running time: ${{hours}}h ${{minutes}}m"
"""
    return batch_template

def main():
    args = parse_args()

    # Load template config and preserve text
    template_config, template_text = load_yaml_template(args.template)
    template_path = Path(args.template)
    template_basename = template_path.stem

    # Check for grid_search section in YAML
    grid_search_params = {}
    run_id_prefix = "RUN"  # Default value
    slurm_settings = {}    # Default empty SLURM settings

    # If grid_search section exists in the template, extract parameters
    if 'grid_search' in template_config:
        grid_section = template_config['grid_search']
        if 'params' in grid_section:
            grid_search_params = grid_section['params']
        if 'run_id_prefix' in grid_section:
            run_id_prefix = grid_section['run_id_prefix']

    # If slurm_settings section exists, extract settings
    if 'slurm_settings' in template_config:
        slurm_settings = template_config['slurm_settings']

    # Check if we have any parameters to search
    if not grid_search_params:
        print("No grid search parameters found in the template YAML. Please add a grid_search section.")
        return

    # Convert string values to appropriate types
    for param_name, values in grid_search_params.items():
        for i, value in enumerate(values):
            if isinstance(value, str):
                try:
                    # Try to convert to float or int if it looks like a number
                    if '.' in value:
                        values[i] = float(value)
                    else:
                        values[i] = int(value)
                except (ValueError, TypeError):
                    # Keep as string if not a number
                    pass

    # Generate all parameter combinations
    param_names = list(grid_search_params.keys())
    param_values = [grid_search_params[name] for name in param_names]
    param_combinations = list(itertools.product(*param_values))

    configs_generated = []
    batch_files_generated = []

    # Generate configs and job files
    for i, combination in enumerate(param_combinations, 1):
        # Create parameter dictionary
        param_dict = {}
        param_desc = []

        for param_name, param_value in zip(param_names, combination):
            param_dict[param_name] = param_value
            param_desc.append(f"{param_name} = {param_value}")

        # Set run ID
        run_id = f"{run_id_prefix}_{i:04d}"

        # Generate new config text with preserved format
        new_config_text = generate_config_with_preserved_format(template_text, param_dict, run_id)

        # Generate file names
        base_without_number = template_basename.rsplit('_', 1)[0]
        config_file = f"configs/{base_without_number}_{i}.yaml"
        job_file = f"job/{args.job_prefix}_{args.dataset}_{i}.batch"
        job_num = f"{args.job_prefix}_{args.dataset}_{i}"

        # Save config file
        with open(config_file, 'w') as f:
            f.write(new_config_text)
        configs_generated.append(config_file)

        # Generate and save batch script
        batch_script = generate_batch_script(config_file, job_num, args.dataset, slurm_settings)
        with open(job_file, 'w') as f:
            f.write(batch_script)
        batch_files_generated.append(job_file)

        print(f"Generated config: {config_file}")
        print(f"Generated batch job: {job_file}")
        print(f"Parameters: {', '.join(param_desc)}")
        print("-" * 50)

    print(f"\nCreated {len(configs_generated)} configuration files and {len(batch_files_generated)} batch job files.")
    print("\nUsage example:")
    print(f"sbatch {batch_files_generated[0]}")

if __name__ == "__main__":
    main()
