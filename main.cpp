// V1.01 eliminates Chxh, Chxe, Chyh, Chye, Ceze, Cezh memory transfers.  //
// The kernel will determine what value should be used based on location  //
// in the grid.															  //

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "parameters.h"
#include "grid_2d.h"
#include "cuda_functions.h"
#include "source.h"
#include <stdio.h>
#include <time.h>
#include "macros.h"


int main()
{
	//--------------------------Variables for main------------------------//
	cudaError_t cudaStatus;
	struct Grid *g; // = new Grid;
	cudaCalloc1((void**)&g, sizeof(struct Grid), 1);
	int src_x_pos = (int)(0.85 * M);
	int src_y_pos = (int)(N / 2);
	const int arraySize = g->sizeX*(g->sizeY - 1);

	int Tx =32;	int Ty = 32;
	int Bx = (M + (Tx - 1))/Tx;
	int By = (N + (Ty - 1))/Ty;
	dim3 BLK(Bx, By, 1);
	dim3 THD(Tx, Ty, 1);

	int hxsize = M * (N - 1);
	int ezsize = M*N;
	int hysize = (M - 1) * N;

//-----------------------Start clock----------------------------------//
	clock_t startTime = clock();
	//-------------------------Initializations----------------------------//
	gridInit(g);
	//abcInit(g);
	//tfsfInit(g);
	ezIncInit(g);
	//snapshotInit2d(g);

	//----------------------Main Loop Start-------------------------------//
	for (int time = 0; time < maxTime; time++) {
//		printf("time = %i\n", time);

		// Update magnetic fields
		updateH2D_CUDA(g, M, N, BLK, THD);
		updateE2D_CUDA(g, M, N, BLK, THD);
		updatesource_CUDA(g, src_x_pos, src_y_pos, M, time);

		//-----------Update total field/scattered field at boundaries---------//
		//tfsfUpdate(g);

		//------------------Update electric field-----------------------------//

		//---------------Apply absorbing boundary conditions------------------//
		//abc(g);

		//---------------------------Add source-------------------------------//
		//g->ez[src_x_pos + src_y_pos * M] = cos(2 * M_PI * 1 * time / ppw);

		// Save data
		//snapshot2d(g);

		//printf("."); //keep alive
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

//	cudaFree(g);

	clock_t endTime = clock();
	clock_t clockTicksTaken = endTime - startTime;
	double timeInSeconds = clockTicksTaken / (double)CLOCKS_PER_SEC;
	printf("Time elapsed: %f s.\n", timeInSeconds);

	//cudaProfilerStop();
	return 0;
}
