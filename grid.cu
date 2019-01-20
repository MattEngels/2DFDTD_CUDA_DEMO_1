#include "grid_2d.h"
#include "parameters.h"
#include "macros.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void initialize_grid_arrays(Grid *g, int M, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int size = M*N;
	if (i < size) {
		g->ez[i] = 0.0;
		g->hx[i] = 0.0;
		g->hy[i] = 0.0;
	}
}

void checkErrorAfterKernelLaunch(const char *location) {
	cudaError_t cudaStatus;
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed in %s: %s\n", location, cudaGetErrorString(cudaStatus));
	}
}

void gridInit(Grid *g) {
	g->sizeX = M;
	g->sizeY = N;
	g->time = 0;
	g->maxTime = maxTime;
	g->type = tmZGrid;
	g->cdtds = Sc;
	int m, n;
	dim3 BLK,THD;
	THD.x = 512;
	BLK.x = floor((THD.x - 1 + (M*N)) / THD.x);



	cudaCalloc((void**)&g->hx, sizeof(double), M*N);
	cudaCalloc((void**)&g->hy, sizeof(double), M*N);
	cudaCalloc((void**)&g->ez, sizeof(double), M*N);

	// Call initialize kernel
	initialize_grid_arrays<<<BLK,THD>>>(g, M, N);
	checkErrorAfterKernelLaunch("gridInit");


}
