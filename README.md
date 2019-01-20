# 2DFDTD_CUDA_DEMO_1
GPU version of Dr. Schneider's 2D FDTD TMZ code

Functionally identical, but all field updates takes place on the GPU.  This has been tested in Windows/MS Visual Studio 
and Linux (Mint 17/18 and Ubuntu 17/18)/Nsight Eclipse Edition.  You must have CUDA installed, including the nvcc compiler.
If you import into MS Visual Studio or NSight Eclipse Edition, use "CUDA Project" to get library paths right to begin with.  
Otherwise, you may have to hand library and header locations down manually and include them in the project settings.
