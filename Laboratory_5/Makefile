SHELL := /bin/bash
# отключаем использование всех доп.параметров переменных среды
export UCX_WARN_UNUSED_ENV_VARS=n
# задаем весь транспорт взаимодействия, общей памяти и ускоренных команд, иначе не получится коммуницировать напрямую с различными узлами памяти в кластере
export UCX_TLS=all


CUDA_HOME ?= /usr/local/cuda
LD_FLAGS = -L$(CUDA_HOME)/lib64 -lcudart -lnvToolsExt

b_mpi: 
	mpicxx -I$(CUDA_HOME)/include -std=c++14 Source.cpp kernels.o $(LD_FLAGS) -o run.out


b_kernels:
	nvcc -lineinfo -gencode arch=compute_70,code=sm_70 -std=c++14 Source.cu -o kernels.o -c

run:
	mpirun -mca pml ucx -np 1 ./run.out 16 0.000001 1000000

all:
	make b_kernels
	make b_mpi

profile:
	mpirun -mca pml ucx -np 4 nsys profile --trace=mpi,cuda,nvtx -o profile.%q{OMPI_COMM_WORLD_RANK} ./run.out 16 0.000001 1000000