b_cpu:
	pgcc -o Compile_CPU Source.c

bm_cpu:
	pgcc -ta=multicore -o Compile_CPU Source.c

b_gpu:
	pgc++ -acc -Minfo=accel  -cudalib=cublas -o blas Source.c

run_cpu:
	./Compile_CPU 128 0.000001 1000000

run_gpu:
	./blas 128 0.000001 1000000

profile:
	nsys profile -o testGPU -t openacc,cuda ./blas 128 0.000001 1000000