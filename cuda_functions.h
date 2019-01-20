#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//cudaError_t updateH1D_CUDA(Grid *g);
cudaError_t updateH2D_CUDA(Grid *g, int M, int N, dim3 BLK, dim3 THD);
cudaError_t updateE2D_CUDA(Grid *g, int M, int N, dim3 BLK, dim3 THD);
cudaError_t updatesource_CUDA(Grid *g, int x, int y, int M, int time);

__global__ void HxUpdate_Kernel(double *dHx, double *dEz, int DIM);
__global__ void HyUpdate_Kernel(double *dHy, double *dEz, int DIM, int type);
__global__ void EzUpdate1D_Kernel(double *dEz, double *dHy, int DIM);
__global__ void EzUpdate2D_Kernel(Grid *g, int DIM);
__global__ void HxHyUpdate_Kernel(Grid *g, int M, int N);
