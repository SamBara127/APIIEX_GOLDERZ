CUDA_HOME ?= /usr/local/cuda
LIB_FLAGS = -L$(CUDA_HOME)/lib64 -lm -lcudart -lcudnn -lcublas -lnvToolsExt
INC_FLAGS = -I$(CUDA_HOME)/include 
OUT_DIR = out_files/

TENSOR_SIZE = 1024
AMOUNT = 14
EPOCHES = 1000000
LESS = 1

b_dataset_data:
	@nvcc -lineinfo -gencode arch=compute_70,code=sm_70 dataset_data.cu -o $(OUT_DIR)dataset_data.out -c

b_neyral_net:
	@nvcc -lineinfo -gencode arch=compute_70,code=sm_70 neyral_net.cu -o $(OUT_DIR)neyral_net.out -c

b_main:
	@pgc++ $(INC_FLAGS) main.cpp $(OUT_DIR)dataset_data.out $(OUT_DIR)neyral_net.out $(LIB_FLAGS) -cudalib=cublas -o run.out

run:
	@echo START_PROGRAM....
	@./run.out $(TENSOR_SIZE) $(AMOUNT) $(EPOCHES) $(LESS)
	@echo PROGRAM HAS ENDED SUCCESSFULLY

build:
	@echo START_BUILDING_SOLUTION....
	@echo ---------------------- BIULD CLASS_FILE "b_dataset_data.cu"
	@make b_dataset_data
	@echo ---------------------- BIULD CLASS_FILE "b_neyral_net.cu"
	@make b_neyral_net
	@echo ---------------------- BIULD MAIN_FILE "main.cpp"
	@make b_main
	@echo ----------------------BUILD HAS COMLPETED!--------------------