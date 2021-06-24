#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Chroma.h"

__global__ void chroma_CUDA(unsigned char* Image, unsigned char* Input_Image2, int Channels);

void Image_Chroma_CUDA(unsigned char* Input_Image, unsigned char* Input_Image2, int Height, int Width, int Channels) {
	unsigned char* Dev_Input_Image = NULL;
	unsigned char* Dev_Input_Image2 = NULL;
	//allocate the memory in gpu
	cudaMalloc((void**)& Dev_Input_Image, Height * Width * Channels);
	cudaMalloc((void**)& Dev_Input_Image2, Height * Width * Channels);

	//copy data from CPU to GPU
	cudaMemcpy(Dev_Input_Image, Input_Image, Height * Width * Channels, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Input_Image2, Input_Image2, Height * Width * Channels, cudaMemcpyHostToDevice);

	dim3 Grid_Image(Width, Height);
	chroma_CUDA << <Grid_Image, 1 >> > (Dev_Input_Image, Dev_Input_Image2, Channels);

	//copy processed data back to cpu from gpu
	cudaMemcpy(Input_Image, Dev_Input_Image, Height * Width * Channels, cudaMemcpyDeviceToHost);

	//free gpu mempry
	cudaFree(Dev_Input_Image);
}

__global__ void chroma_CUDA(unsigned char* Image, unsigned char* Image2, int Channels) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int idx = (x + y * gridDim.x) * Channels;

	for (int i = 0; i < Channels; i++) {
		if (Image[idx + i + 1] == 255)
		{
			Image[idx + i] = Image2[idx + i];
			Image[idx + i + 1] = Image2[idx + i + 1];
			Image[idx + i + 2] = Image2[idx + i + 2];
		}

	}
}