#!/bin/bash
#SBATCH -A brahad.kokad
#SBATCH --qos=medium
#SBATCH -p long
#SBATCH -n 2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END

module load u18/matlab/R2024a

echo "copying smart_chair.tar directory to scratch"
rsync -r /home/brahad.kokad/smart_chair.tar /ssd_scratch/brahad.kokad/smart_chair.tar
echo "copying smart_chair.tar directory to scratch done"

tar -xvf /ssd_scratch/brahad.kokad/smart_chair.tar
echo "untarred smart_chair.tar"

rm /ssd_scratch/brahad.kokad/smart_chair.tar

cd /scratch/brahad.kokad/smart_chair

echo "running matlab script"
matlab -nodesktop -nosplash -singleCompThread -r train_deepnet_bayesopt.m

rsync -zaP /ssd_scratch/brahad.kokad/smart_chair/results /home/brahad.kokad/results/
echo "copied results to home done"

rm -rf /ssd_scratch/brahad.kokad/smart_chair
echo "cleaned up scratch"