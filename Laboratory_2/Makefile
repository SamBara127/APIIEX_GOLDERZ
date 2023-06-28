b_cpu:
	pgcc -o Compile_CPU Source.c

bm_cpu:
	pgcc -ta=multicore -o Compile_CPU Source.c

b_gpu:
	pgcc -acc -Minfo=accel -o Compile_GPU Source.c

run_cpu:
	./Compile_CPU 256 0.000001 1000000

run_gpu:
	./Compile_GPU 64 0.000001 1000000

profile:
	