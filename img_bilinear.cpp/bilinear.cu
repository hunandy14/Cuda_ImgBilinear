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

// �ŧiGPU���z�ܼ�(�u������)
texture<float, 2, cudaReadModeElementType> rT;
// ���z�u�ʨ��Ȯ֤�
__global__ void biliner_kernel(float* dst, int srcW, int srcH, float ratio) {
	int idxX = blockIdx.x * blockDim.x + threadIdx.x,
		idxY = blockIdx.y * blockDim.y + threadIdx.y;
	if(idxX < srcW*ratio && idxY < srcH*ratio) { // �|�h�]�@�I�I�n�ױ�
		float srcX = idxX / ratio;
		float srcY = idxY / ratio;
		size_t idx = (idxY*srcW*ratio + idxX);
		dst[idx] = tex2D(rT, srcX+0.5, srcY+0.5);
	}
}
// ���z�u�ʨ��Ȩ禡
#define AutoMem_Style
#ifdef AutoMem_Style
__host__ void biliner_texture_core(float *dst, const float* src,
	size_t dstW, size_t dstH, float ratio)
{
	Timer T; T.priSta = 1;
	// �]�mGPU�һݪ���
	int srcSize = dstW*dstH;
	int dstSize = srcSize*ratio*ratio;

	// �ŧi texture2D�}�C�øj�w
	T.start();
	CudaMemArr<float> cuArray(src, dstW, dstH);
	cudaBindTextureToArray(rT, cuArray);
	T.print("  GPU new ���z�Ŷ�");

	// �]�m ���ӼҦ�and�W�X��ɸ����
	rT.filterMode = cudaFilterModeLinear;
	rT.addressMode[0] = cudaAddressModeClamp;
	rT.addressMode[1] = cudaAddressModeClamp;

	// �n�DGPU�Ŷ�
	T.start();
	CudaData<float> gpu_dst(dstSize);
	T.print("  GPU new �@��Ŷ�");

	// �]�m�����
	dim3 block(BLOCK_DIM, BLOCK_DIM);
	dim3 grid(ceil((float)dstW*ratio / BLOCK_DIM), ceil((float)dstH*ratio / BLOCK_DIM));
	T.start();
	biliner_kernel <<< grid, block >> > (gpu_dst, dstW, dstH, ratio);
	T.print("  �֤߭p��");

	// ���XGPU��
	T.start();
	gpu_dst.memcpyOut(dst, dstSize);
	T.print("  GPU ���X���");
}

#else
__host__ void biliner_texture_core(float *dst, const float* src,
	size_t dstW, size_t dstH, float ratio)
{
	Timer T; T.priSta = 1;
	// �]�mGPU�һݪ���
	int srcSize = dstW*dstH;
	int dstSize = srcSize*ratio*ratio;

	// �ŧi texture2D�}�C�øj�w
	T.start();
	cudaChannelFormatDesc chDesc = cudaCreateChannelDesc<float>();
	cudaArray* cuArray = nullptr;
	cudaMallocArray(&cuArray, &chDesc, dstW, dstH);
	cudaMemcpyToArray(cuArray, 0, 0, src, srcSize*sizeof(float), cudaMemcpyHostToDevice);
	cudaBindTextureToArray(rT, cuArray);
	T.print("  GPU new ���z�Ŷ�");

	// �]�m ���ӼҦ�and�W�X��ɸ����
	rT.filterMode = cudaFilterModeLinear;
	rT.addressMode[0] = cudaAddressModeClamp;
	rT.addressMode[1] = cudaAddressModeClamp;

	// �n�DGPU�Ŷ�
	T.start();
	float* gpu_dst = nullptr;
	cudaMalloc((void**)&gpu_dst, dstSize*sizeof(float));
	T.print("  GPU new �@��Ŷ�");

	// �]�m�����
	dim3 block(BLOCK_DIM, BLOCK_DIM);
	dim3 grid(ceil((float)dstW*ratio / BLOCK_DIM), ceil((float)dstH*ratio / BLOCK_DIM));
	T.start();
	biliner_kernel << < grid, block >> > (gpu_dst, dstW, dstH, ratio);
	T.print("  �֤߭p��");

	// ���XGPU��
	T.start();
	cudaMemcpy(dst, gpu_dst, dstSize*sizeof(float), cudaMemcpyDeviceToHost);
	T.print("  GPU ���X���");

	// ����GPU�O����
	cudaUnbindTexture(rT);
	cudaFreeArray(cuArray);
	cudaFree(gpu_dst);
}
#endif // AutoMem_Style

// ���z�u�ʨ��Ȩ禡 vector �श����
__host__ void biliner_texture(vector<float>& dst, const vector<float>& src,
	size_t width, size_t height, float ratio)
{
	Timer T; T.priSta = 0;
	T.start();
	dst.resize(width*ratio * height*ratio);
	T.print(" CPU new �x�s�Ŷ�");
	T.start();
	biliner_texture_core(dst.data(), src.data(), width, height, ratio);
	T.print(" GPU ����");
}











