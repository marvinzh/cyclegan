#!/bin/bash

#$ -cwd                      ## Execute a job in current directory
#$ -l q_node=1               ## Use number of node
#$ -l h_rt=24:00:00          ## Running job time

export PATH=/home/1/17M14253/anaconda2/bin:$PATH

. /etc/profile.d/modules.sh  ## Initialize module commands
module load cuda/9.0.176
module load intel
module load cudnn/7.1
module load nccl/2.2.13
module load openmpi/2.1.2-pgi2018

source activate tfpy36

python $1

