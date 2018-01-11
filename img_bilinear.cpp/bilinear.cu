/*****************************************************************
Name :
Date : 2018/01/08
By   : CharlotteHonG
Final: 2018/01/08
*****************************************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
using namespace std;

#include "CudaMem\CudaMem.cuh"
#include "timer.hpp"

#define BLOCK_DIM 16

// 宣告GPU紋理變數(只能放全域)
texture<float, 2, cudaReadModeElementType> rT;
// 紋理線性取值核心
__global__ void biliner_kernel(float* dst, int srcW, int srcH, float ratio) {
	int idxX = blockIdx.x * blockDim.x + threadIdx.x,
		idxY = blockIdx.y * blockDim.y + threadIdx.y;
	if(idxX < srcW*ratio && idxY < srcH*ratio) { // 會多跑一點點要擋掉
		float srcX = idxX / ratio;
		float srcY = idxY / ratio;
		size_t idx = (idxY*srcW*ratio + idxX);
		dst[idx] = tex2D(rT, srcX+0.5, srcY+0.5);
	}
}
// 紋理線性取值函式
#define AutoMem_Style
#ifdef AutoMem_Style
__host__ void biliner_texture_core(float *dst, const float* src,
	size_t dstW, size_t dstH, float ratio)
{
	Timer T; T.priSta = 1;
	// 設置GPU所需長度
	int srcSize = dstW*dstH;
	int dstSize = srcSize*ratio*ratio;

	// 宣告 texture2D陣列並綁定
	T.start();
	CudaMemArr<float> cuArray(src, dstW, dstH);
	cudaBindTextureToArray(rT, cuArray);
	T.print("  GPU new 紋理空間");

	// 設置 插植模式and超出邊界補邊界
	rT.filterMode = cudaFilterModeLinear;
	rT.addressMode[0] = cudaAddressModeClamp;
	rT.addressMode[1] = cudaAddressModeClamp;

	// 要求GPU空間
	T.start();
	CudaData<float> gpu_dst(dstSize);
	T.print("  GPU new 一般空間");

	// 設置執行緒
	dim3 block(BLOCK_DIM, BLOCK_DIM);
	dim3 grid(ceil((float)dstW*ratio / BLOCK_DIM), ceil((float)dstH*ratio / BLOCK_DIM));
	T.start();
	biliner_kernel <<< grid, block >> > (gpu_dst, dstW, dstH, ratio);
	T.print("  核心計算");

	// 取出GPU值
	T.start();
	gpu_dst.memcpyOut(dst, dstSize);
	T.print("  GPU 取出資料");
}

#else
__host__ void biliner_texture_core(float *dst, const float* src,
	size_t dstW, size_t dstH, float ratio)
{
	Timer T; T.priSta = 1;
	// 設置GPU所需長度
	int srcSize = dstW*dstH;
	int dstSize = srcSize*ratio*ratio;

	// 宣告 texture2D陣列並綁定
	T.start();
	cudaChannelFormatDesc chDesc = cudaCreateChannelDesc<float>();
	cudaArray* cuArray = nullptr;
	cudaMallocArray(&cuArray, &chDesc, dstW, dstH);
	cudaMemcpyToArray(cuArray, 0, 0, src, srcSize*sizeof(float), cudaMemcpyHostToDevice);
	cudaBindTextureToArray(rT, cuArray);
	T.print("  GPU new 紋理空間");

	// 設置 插植模式and超出邊界補邊界
	rT.filterMode = cudaFilterModeLinear;
	rT.addressMode[0] = cudaAddressModeClamp;
	rT.addressMode[1] = cudaAddressModeClamp;

	// 要求GPU空間
	T.start();
	float* gpu_dst = nullptr;
	cudaMalloc((void**)&gpu_dst, dstSize*sizeof(float));
	T.print("  GPU new 一般空間");

	// 設置執行緒
	dim3 block(BLOCK_DIM, BLOCK_DIM);
	dim3 grid(ceil((float)dstW*ratio / BLOCK_DIM), ceil((float)dstH*ratio / BLOCK_DIM));
	T.start();
	biliner_kernel << < grid, block >> > (gpu_dst, dstW, dstH, ratio);
	T.print("  核心計算");

	// 取出GPU值
	T.start();
	cudaMemcpy(dst, gpu_dst, dstSize*sizeof(float), cudaMemcpyDeviceToHost);
	T.print("  GPU 取出資料");

	// 釋放GPU記憶體
	cudaUnbindTexture(rT);
	cudaFreeArray(cuArray);
	cudaFree(gpu_dst);
}
#endif // AutoMem_Style

// 紋理線性取值函式 vector 轉介介面
__host__ void biliner_texture(vector<float>& dst, const vector<float>& src,
	size_t width, size_t height, float ratio)
{
	Timer T; T.priSta = 0;
	T.start();
	dst.resize(width*ratio * height*ratio);
	T.print(" CPU new 儲存空間");
	T.start();
	biliner_texture_core(dst.data(), src.data(), width, height, ratio);
	T.print(" GPU 全部");
}











