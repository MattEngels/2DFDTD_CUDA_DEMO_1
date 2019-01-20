#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "parameters.h"
#include "grid_2d.h"
#include <stdio.h>

__global__ void HxUpdate_Kernel(double *dHx, double *dEz, int DIM)
{
	// Map from threadIdx/blockIdx to cell position
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = row * blockDim.x *gridDim.x + col;

	int total = DIM*(DIM - 1);
	int top = offset + blockDim.x * gridDim.x;
	double Chxe = 0.0018769575507639;
	if ((row == DIM - 1)) top -= DIM;

	if (offset < total)
		dHx[offset] = 1.0 * dHx[offset] - Chxe * (dEz[top] - dEz[offset]);
	else
		dHx[offset] = 0.0;
}

__global__ void HyUpdate_Kernel(double *dHy, double *dEz, int DIM, int type)
{
	// Map from threadIdx/blockIdx to cell position
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = row * blockDim.x *gridDim.x + col;

	int total = DIM*(DIM - 1);
	int right = offset + 1;
	double Chye = 0.0018769575507639;
	if ((col == DIM - 1) || (col == DIM -2)) right--;

	if (type == 1) {
		if (offset < DIM - 1)
			dHy[offset] = 1.0 * dHy[offset] + Chye * (dEz[right] - dEz[offset]);
		else
			dHy[offset] = 0.0;
	}
	else {
		if (offset < total)
			dHy[offset] = 1.0 * dHy[offset] + Chye * (dEz[right] - dEz[offset]);
		else
			dHy[offset] = 0.0;
	}
}

__global__ void HxHyUpdate_Kernel(Grid *g, int M, int N)
{
	__shared__ double Che;
	int size_Hx = M * (N - 1);
	int size_Hy = (M - 1) * N;


	// Map from threadIdx/blockIdx to cell position
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = row * blockDim.x *gridDim.x + col;

	if (threadIdx.x == 0)
		Che = 0.0018769575507639;
	__syncthreads();

	int top = offset + blockDim.x * gridDim.x;
	int	right = offset + 1;

	// Calculate Hx
	if ((row == M - 1))
		top -= M;
	if (offset < size_Hx)
		g->hx[offset] = 1.0 * g->hx[offset] - Che * (g->ez[top] - g->ez[offset]);
	else
		g->hx[offset] = 0.0;

	// Calculate Hy
	if ((col == M - 1) || (col == M - 2))
		right--;
	if (offset < size_Hy)
		g->hy[offset] = 1.0 * g->hy[offset] + Che * (g->ez[right] - g->ez[offset]);
	else
		g->hy[offset] = 0.0;
}

__global__ void EzUpdate1D_Kernel(double *dEz, double *dHy, int DIM)
{
	// Map from threadIdx/blockIdx to cell position
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = row * blockDim.x *gridDim.x + col;

	//Find values for Ceze and Cezh
	double Ceze = 1.0;
	double Cezh = 266.3885498084424;

	int left = offset - 1;

	if (col == 0) {
		left++;
		dEz[offset] = 0.0;
	}
	else {
		if (offset < DIM - 1)
			dEz[offset] = Ceze * dEz[offset] + Cezh * (dHy[offset] - dHy[left]);
		else
			dEz[offset] = 0.0;
	}
}

__global__ void EzUpdate2D_Kernel(Grid *g, int DIM)
{
	// Map from threadIdx/blockIdx to cell position
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = row * blockDim.x *gridDim.x + col;

	int total = DIM*DIM;
	int left = offset - 1;
	int right = offset + 1;
	int top = offset + blockDim.x * gridDim.x;
	int bottom = offset - blockDim.x * gridDim.x;

	__shared__ double Ceze, Cezh;

	if (threadIdx.x == 0) {
		Ceze = 1.0;
		Cezh = 266.3885498084424;
	}
	__syncthreads();

	if (col == 0)			left++;
	if (col == DIM - 1)		right--;
	if ((row == DIM - 1))	top -= DIM;
	if (row == 0)			bottom += DIM;

	if ((col == 0) || (col == (M - 1)) || (row == 0) || (row == (N - 1)))
		g->ez[offset] = 0.0;
	else {
		if (offset < total)
			g->ez[offset] = Ceze * g->ez[offset] +
			Cezh * ((g->hy[offset] - g->hy[left]) - (g->hx[offset] - g->hx[bottom]));
		else
			g->ez[offset] = 0.0;
		}
}

__global__ void SourceUpdate2D_Kernel(Grid *g, int x, int y, int M, int time) {

	g->ez[x + y * M] = cos(2 * M_PI * 1 * time / 25);
}

//cudaError_t updateH1D_CUDA(Grid *g)
//{
//	// Declare variables needed throughout
//	double *dev_ez = 0;
//	double *dev_hy = 0;
//	cudaError_t cudaStatus;
//
//	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	// Calculate CUDA grid dimensions.  Block dimension fixed at 32x32 threads
//	int Tx = 32;
//	int Ty = 32;
//	int Bx = (g->sizeX + (Tx - 1)) / Tx;
//	dim3 BLK(Bx, 1, 1);
//	dim3 THD(Tx, Ty, 1);
//
//	int ezsize = g->sizeX * g->sizeY;
//	int hysize = (g->sizeX - 1) * g->sizeY;
//	int size = M;
//
//	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//	// Prelim check, then allocate GPU buffer space
//
//	// Choose which GPU to run on, change this on a multi-GPU system.
//	cudaStatus = cudaSetDevice(1);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//		goto Error;
//	}
//
//	// Allocate GPU buffers for Hy, Ez, Chyh, Chye
//	cudaStatus = cudaMalloc((void**)&dev_ez, size * sizeof(double));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMalloc((void**)&dev_hy, hysize * sizeof(double));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//	// Copy input vectors from host memory to GPU buffers.
//	cudaStatus = cudaMemcpy(dev_ez, g->ez, ezsize * sizeof(double), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMemcpy(dev_hy, g->hy, hysize * sizeof(double), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//	// Launch a kernel on the GPU with one thread for each element.
//	HyUpdate_Kernel << <BLK, THD >> >(dev_hy, dev_ez, g->sizeX, 1);
//
//	// Check for any errors launching the kernel
//	cudaStatus = cudaGetLastError();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//		goto Error;
//	}
//
//	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//	// Do a device sync and copy Hy from device back to host
//
//	// cudaDeviceSynchronize waits for the kernel to finish, and returns
//	// any errors encountered during the launch.
//	cudaStatus = cudaDeviceSynchronize();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//		goto Error;
//	}
//
//	// Copy output vector from GPU buffer to host memory.
//	cudaStatus = cudaMemcpy(g->hy, dev_hy, hysize * sizeof(double), cudaMemcpyDeviceToHost);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//	// Free pointers no longer needed and load the next set needed for Hy calculation
//	cudaFree(dev_ez);
//	cudaFree(dev_hy);
//
//
//Error:
//
//	return cudaStatus;
//}

// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
cudaError_t updateH2D_CUDA(Grid *g, int M, int N, dim3 BLK, dim3 THD)
{
	// Declare variables needed throughout
	cudaError_t cudaStatus;
	// Launch a kernel on the GPU with one thread for each element.
	HxHyUpdate_Kernel << <BLK, THD >> >(g, M, N);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "updateHxHy launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

Error:

	return cudaStatus;
}

cudaError_t updateE2D_CUDA(Grid *g, int M, int N, dim3 BLK, dim3 THD)
{
	// Declare variables needed throughout
	cudaError_t cudaStatus;

	EzUpdate2D_Kernel << <BLK, THD >> >(g, M);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "EzUpdate launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}


Error:

	return cudaStatus;
}

cudaError_t updatesource_CUDA(Grid *g, int x, int y, int M, int time) {

	cudaError_t cudaStatus;

	SourceUpdate2D_Kernel << <1, 1 >> >(g, x, y, M, time);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "updatesource launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	Error:

	return cudaStatus;

}

