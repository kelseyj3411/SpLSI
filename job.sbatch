#!/bin/bash
#SBATCH --job-name=splsi_sim
#SBATCH --output=splsi_%A_%a.out
#SBATCH --error=splsi_%A_%a.err
#SBATCH --account=pi-cdonnat
#SBATCH --array=1-42
#SBATCH --time=12:00:00
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=danielxli@uchicago.edu  # Where to send email

module load python/anaconda-2022.05
source activate /home/danielxli/splsi_env
cd SpLSI
python3 run_sim.py "$SLURM_ARRAY_TASK_ID" 
