b_cpu:
	pgcc -o Compile_CPU Source.c

bm_cpu:
	pgcc -ta=multicore -o Compile_CPU Source.c

b_gpu:
	nvcc -arch=sm_70 Source.cu -lm -o cuda

run_gpu:
	./cuda 128 0.000001 1000000 0

profile:
	nsys profile -o testGPU -t cuda ./cuda 128 0.000001 1000000 0

test:
	nvcc -arch=sm_70 test.cu -lm -o testik

cpu:
	pgcc -o testik test.c