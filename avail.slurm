#! /bin/sh

#SBATCH --job-name=avail_TML
#SBATCH --output=./output/e2_%n_%j_%t.out # tedirect stdout
#SBATCH --error=./output/e2_%n_%j_%t.err # redirect stderr
#SBATCH --partition=studentbatch # (see resources section)
#SBATCH --time=2880 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=8100 # CPU memory (MB)
#SBATCH --cpus-per-task=2 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total

/home/yandex/DL20222023a/baryaacovi/anaconda3/bin/python3 main.py --pgd-samples-selection smallest_model_loss