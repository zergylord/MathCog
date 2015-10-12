#!/bin/bash 
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling  
#################
#set a job name  
#SBATCH --job-name=IMB4
#################  
#a file for job output, you can check job progress
#SBATCH --output=IMB4.out
#################
# a file for errors from the job
#SBATCH --error=IMB4.err
#################
#time you think you need; default is one hour
#in minutes in this case
#SBATCH --time=2:00
#################
#quality of service; think of it as job priority
#SBATCH --qos=normal
#################
#number of nodes you are requesting
#SBATCH --nodes=1
#################
#memory per node; default is 4000 MB per CPU
#SBATCH --mem=4000
#you could use --mem-per-cpu; they mean what we are calling cores
#################
#tasks to run per node; a "task" is usually mapped to a MPI processes.
# for local parallelism (OpenMP or threads), use "--ntasks-per-node=1
# --cpus-per-tasks=16" instead
#SBATCH --ntasks-per-node=1
#################

#now run normal batch commands
#module load openmpi/1.6.5/intel13sp1up1

#run Intel MPI Benchmarks with mpirun
#srun /usr/mpi/intel-13/openmpi-1.6.5-1/tests/IMB-3.2.4/IMB-MPI1
srun th train.lua
