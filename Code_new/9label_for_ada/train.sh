#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p long
#SBATCH -n 2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END

module load u18/matlab/R2024a

mkdir /scratch/brahad.kokad

scp -r /home2/brahad.kokad/9label_for_ada /scratch/brahad.kokad

cd /scratch/brahad.kokad/9label_for_ada

# tar -xvf for_compute.tar
# echo "untarred for_compute.tar"

# rm for_compute.tar

# mkdir training_results

echo "running matlab script"
matlab -nodesktop -nosplash -singleCompThread -r train_deepnet_bayesopt

scp -r /scratch/brahad.kokad/9label_for_ada/training_results /home2/brahad.kokad/9label_for_ada/
echo "copied results to home done"

# rm -rf /scratch/brahad.kokad/new_for_ada
# echo "cleaned up scratch"
