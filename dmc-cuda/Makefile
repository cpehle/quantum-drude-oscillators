NVCC=nvcc -gencode arch=compute_52,code=sm_52 -std=c++11 -I ../libs/moderngpu/src -I . --expt-extended-lambda -Xptxas=-v -lineinfo

qdo-diatom: apps/qdo-diatom.cu hamiltonians/qdo-diatom.hxx hamiltonians/quantum-system.hxx samplers/traditional-dmc.hxx util/*.hxx
	$(NVCC) apps/qdo-diatom.cu -o qdo-diatom
