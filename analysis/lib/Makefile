CC = icc

default: libkernels.a

libkernels.a: kernels.o
	ar rcs $@ $^
    
kernels.o: kernels.c kernels.h
	$(CC) -fPIC -qopenmp -O3 -c $<

clean:
	rm *.o *.a
