#!/bin/bash
#SBATCH --job-name=splsi_test
#SBATCH --account=pi-cdonnat
#SBATCH --time=2:00:00
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=ALL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=danielxli@uchicago.edu  # Where to send email

module load python/anaconda-2022.05
source activate /home/danielxli/splsi_env
cd SpLSI
python3 run_sim.py 4
