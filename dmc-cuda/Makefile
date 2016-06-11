NVCC=nvcc -gencode arch=compute_52,code=sm_52 -std=c++11 -I ../libs/moderngpu/src -I . --expt-extended-lambda -Xptxas=-v -lineinfo

SAMPLERS=samplers/*.hxx hamiltonians/quantum-system.hxx util/*.hxx

qdo-diatom: apps/qdo-diatom.cu hamiltonians/qdo-diatom.hxx $(SAMPLERS)
	$(NVCC) apps/qdo-diatom.cu -o qdo-diatom
