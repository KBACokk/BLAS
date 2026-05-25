.PHONY: all build test_interface test_performance test

all: build test

build:
	g++ -O3 main.cpp -o benchmark -lopenblas

test_interface:
	pytest test_blas.py test_trsm_interface.py -v

test_performance:
	./benchmark

test: test_interface test_performance

run_bench:
	./benchmark
