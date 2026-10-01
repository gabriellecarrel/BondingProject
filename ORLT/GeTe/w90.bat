#!/bin/bash

#SBATCH --job-name=STATIC
#SBATCH --nodes 1
#SBATCH --output=slurm.%N.%j.o
#SBATCH --error=slurm.%N.%j.e
#SBATCH --ntasks-per-node=12
#SBATCH --mem-per-cpu=14G 
#SBATCH --time=3:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=gc2596@princeton.edu

###ENVIRONMENT###

module purge
module load intel-oneapi/2024.2 
module load intel-mkl/2024.2 
module load openmpi/oneapi-2024.2/4.1.6 
module load hdf5/oneapi-2024.2/openmpi-4.1.6/1.14.4 
                       
 export JOB=$1
 export WorkDir=/scratch/gpfs/SCHOOP/$LOGNAME/$JOB
 mkdir -p $WorkDir
 export HomeDir=$PWD
 cp -rp $HomeDir/* $WorkDir
 cd $WorkDir
 echo $WorkDir
 echo 'start time'
 time
 srun /projects/SCHOOP/VASP_Executables/wannier90-3.1.0/wannier90.x wannier90.win  > "$HomeDir"/"$JOB".out 2> "$HomeDir"/"$JOB".err
 echo 'end time'
 time

cp $WorkDir/* $HomeDir/
#mkdir /projects/SCHOOP/$LOGNAME/$JOB/ #comment out if running from projects
#cp $WorkDir/* /projects/SCHOOP/$LOGNAME/$JOB/ #comment out if running from projects 
cd $HomeDir
rm -r $WorkDir


