NVCC=nvcc -gencode arch=compute_52,code=sm_52 -std=c++11 -I ../libs/moderngpu/src -I . --expt-extended-lambda -lineinfo

SAMPLERS=samplers/*.hxx hamiltonians/quantum-system.hxx util/*.hxx

qdo-diatom: apps/qdo-diatom.cu hamiltonians/qdo-diatom.hxx $(SAMPLERS)
	$(NVCC) apps/qdo-diatom.cu -o qdo-diatom

qdo-triatom: apps/qdo-triatom.cu hamiltonians/qdo-diatom.hxx hamiltonians/qdo-triatom.hxx $(SAMPLERS)
	$(NVCC) apps/qdo-triatom.cu -o qdo-triatom
