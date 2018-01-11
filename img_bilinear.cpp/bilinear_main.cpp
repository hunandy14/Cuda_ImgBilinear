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
#include <string>
using namespace std;

#include "bilinear.cuh"
#include "BMP_RW\BMP_RW.h"
#include "CudaMem\CudaMem.cuh"
#include "timer\timer.hpp"


vector<float> tofloat(const uch* img, size_t size) {
	vector<float> temp(size);
	for(size_t i = 0; i < size; i++) {
		temp[i] = img[i];
	} return temp;
}
vector<uch> touch(const float* img, size_t size) {
	vector<uch> temp(size);
	for(size_t i = 0; i < size; i++) {
		temp[i] = img[i];
	} return temp;
}

int main(){
	size_t a;
	// 讀取
	ImgData img1("img//kanna.bmp");
	img1.gray();

	// 處理
	float ratio = 5;
	vector<float> img_gpuRst, img_data = tofloat(img1.data, img1.size);

	double time;
	//time = biliner_texture(img_gpuRst, img_data, img1.width, img1.height, ratio);
	//time = biliner_share(img_gpuRst, img_data, img1.width, img1.height, ratio);
	time = biliner_CPU(img_gpuRst, img_data, img1.width, img1.height, ratio);


	// 輸出
	vector<unsigned char> img_out =  touch(img_gpuRst.data(), img_gpuRst.size());
	string name = "img//Out-texture_"+to_string(time)+".bmp";
	bmpWrite(name.c_str(), img_out.data(), img1.width*ratio, img1.height*ratio, 8);
    return 0;
}