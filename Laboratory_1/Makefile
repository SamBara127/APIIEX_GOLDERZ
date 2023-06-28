profile_on:
	export PGI_ACC_TIME=1

profile_off:
	export PGI_ACC_TIME=0

b_cpu:
	pgcc -o Compile_CPU Source.c

b_gpu:
	pgcc -acc -Minfo=accel -o Compile_GPU Source.c

run_cpu:
	./Compile_CPU 

run_gpu:
	./Compile_GPU 