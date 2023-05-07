//integral
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>   
#include <time.h> 
#include <math.h>

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

const int N = 1024 * 1024 * 64;     	//积分时划分的份数
const int threadsPerBlock = 256;	//block中的线程数
const int blocksPerGrid = 64;		//grid中的block数

//缺省__host__，表明CPU运行，CPU调用
double function_for_cpu(double x) {
	return 4 / (1 + x * x);
}

//__device__修饰，只能被内核函数调用，表明GPU运行，GPU调用
__device__ double function_for_gpu(double x) {
	return 4 / (1 + x * x);
}

__global__ void trap(double *a, double *b, double *integral) {
	__shared__ double cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	//
	int count = 0;
	double x, temp = 0;
	// 一共N份，tid=0,1...
	while (tid < N) {
		x = *a + (double)(*b - *a) / N * (tid + 0.5);
		temp += function_for_gpu(x);
		tid += blockDim.x * gridDim.x;
	}
	count += 1;
	cache[cacheIndex] = temp;

	__syncthreads();//每个块里32线程，每个kernel负责一个块，串行

	int i = blockDim.x / 2;//两个块各32个缓存组
	while (i != 0) {
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0)
		integral[blockIdx.x] = cache[0];
}


void trap_by_cpu(double a, double b, double *integral) {
	int i;
	double x, temp = 0;
	for (i = 0; i < N; i++) {
		x = a + (double)(b - a) / N * (i + 0.5);//步长
		temp += function_for_cpu(x);
	}
	temp *= (double)(b - a) / N;
	*integral = temp;
}

int main(void) {
	// 主机端
	double   integral;
	double	*partial_integral;
	double a, b;
	// 设备端
	double   *dev_partial_integral;
	double *dev_a, *dev_b;

	cudaEvent_t start, stop;
	float tm;
	// 计时
	clock_t  clockBegin, clockEnd;
	float duration;

	a = 0;
	b = 1;

	//使用CPU计算PI的值
	clockBegin = clock();
	trap_by_cpu(a, b, &integral);
	clockEnd = clock();
	duration = (float)1000 * (clockEnd - clockBegin) / CLOCKS_PER_SEC;
	printf("CPU Result: %.6f\n", integral);
	printf("CPU Elapsed time: %.6fms\n", duration);

	//	getchar();

	//使用GPU+CPU计算PI的值
	//使用event计算时间
	cudaEventCreate(&start); //创建event
	cudaEventCreate(&stop);  //创建event
	cudaEventRecord(start, 0);  //记录当前时间

	// 申请CPU存储空间
	partial_integral = (double*)malloc(blocksPerGrid * sizeof(double));

	// 申请GPU存储空间
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_partial_integral, blocksPerGrid * sizeof(double)));

	//将'a'和'b'复制到GPU
	HANDLE_ERROR(cudaMemcpy(dev_a, &a, sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, &b, sizeof(double), cudaMemcpyHostToDevice));

	//启动内核函数，启动的线程块的数量是64个，每个线程块的线程数量是256
	trap << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_partial_integral);

	//将计算结果数组'dev_partial_integral'从GPU复制到CPU
	HANDLE_ERROR(cudaMemcpy(partial_integral, dev_partial_integral,
		blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost));

	//在CPU上进行归约操作，得到最终的计算结果
	integral = 0;
	for (int i = 0; i < blocksPerGrid; i++) {
		integral += partial_integral[i];
	}
	integral *= (double)(b - a) / N;

	cudaEventRecord(stop, 0);  //记录当前时间
	cudaEventSynchronize(stop);  //等待stop event完成
	cudaEventElapsedTime(&tm, start, stop);  //计算时间差（毫秒级）
	printf("GPU Result: %.20lf\n", integral);
	printf("GPU Elapsed time:%.6f ms.\n", tm);

	//释放GPU内存
	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_partial_integral));

	//释放CPU内存
	free(partial_integral);

	getchar();
}
