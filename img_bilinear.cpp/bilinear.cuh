/*****************************************************************
Name : 
Date : 2018/01/08
By   : CharlotteHonG
Final: 2018/01/08
*****************************************************************/
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using std::vector;
__host__ void biliner_texture(vector<float>& dst, const vector<float>& src, 
	size_t width, size_t height, float ratio);