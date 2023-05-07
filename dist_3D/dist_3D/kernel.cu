
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define W 32
#define H 32
#define D 32
#define TX 8 // number of threads per block along x-axis
#define TY 8 // number of threads per block along y-axis
#define TZ 8 // number of threads per block along z-axis

int divUp(int a, int b) { return (a + b - 1) / b; }

__device__
double distance(int c, int r, int s, double *pos) {
	return sqrtf((c - pos[1])*(c - pos[1]) + (r - pos[2])*(r - pos[2]) +
		(s - pos[3])*(s - pos[3]));
}

__global__
void distanceKernel(double *d_out, int w, int h, int d, double *pos) {
	const int c = blockIdx.x * blockDim.x + threadIdx.x; // column
	const int r = blockIdx.y * blockDim.y + threadIdx.y; // row
	const int s = blockIdx.z * blockDim.z + threadIdx.z; // stack
	const int i = c + r * w + s * w*h;
	if ((c >= w) || (r >= h) || (s >= d)) return;
	d_out[i] = distance(c, r, s, pos); // compute and store result
}

int main() {
	double *out = (double*)calloc(W*H*D, sizeof(double));
	double *d_out = 0;
	cudaMalloc(&d_out, W*H*D * sizeof(double));
	const double *pos[3] = 0; // set reference position
	const dim3 blockSize(TX, TY, TZ);
	const dim3 gridSize(divUp(W, TX), divUp(H, TY), divUp(D, TZ));
	distanceKernel << <gridSize, blockSize >> > (d_out, W, H, D, pos);
	cudaMemcpy(out, d_out, W*H*D * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_out);
	free(out);
	return 0;
}