#!/bin/bash

LOCAL_RANK=${MPI_LOCALRANKID} # mpirun Intel MPI
if [ -z "${LOCAL_RANK}" ]; then LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}; fi # mpirun OpenMPI
if [ -z "${LOCAL_RANK}" ]; then LOCAL_RANK=${SLURM_LOCALID}; fi  # srun 

export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}

"$@"