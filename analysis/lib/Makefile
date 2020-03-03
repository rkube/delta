CC = icc

default: libkernels.a

libkernels.a: kernels.o
	ar rcs $@ $^
    
kernels.o: kernels.c kernels.h
	$(CC) -fPIC -qopenmp -O3 -qopt-report=5 -complex-limited-range -c $<

clean:
	rm *.o *.a
