CC = cc

default: libkernels.a

libkernels.a: kernels.o
	ar rcs $@ $^
    
kernels.o: kernels.c kernels.h
	#$(CC) -xHOST -fPIC -qopenmp -Ofast -qopt-report=5 -complex-limited-range -c $<
	$(CC) -fPIC -fopenmp -O3 -c $<

clean:
	rm *.o *.a
