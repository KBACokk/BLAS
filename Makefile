all: build test_python

build:
	g++ -O3 main.cpp -o benchmark -lopenblas

run_bench:
	./benchmark

test_python:
	pytest test_blas.py