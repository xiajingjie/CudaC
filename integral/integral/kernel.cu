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

/**
 * __global__修饰，内核函数，表明GPU运行，CPU调用
 * 在GPU上使用积分法并行计算PI
 * @param a 积分下界
 * @param b 积分上界
 * @param integral 存储积分结果
 * @return
 */
__global__ void trap(double *a, double *b, double *integral) {
	/**
	 * __shared__修饰，声明一个共享内存缓冲区，名字为cache
	 * 表示数据存放在共享存储器中，每个线程块都有该变量的一个副本；
	 * 只有在块内的线程可以访问，其它块内的线程不能访问；
	 */
	__shared__ double cache[threadsPerBlock];
	//共享内存中保证分配的共享内存的大小和线程数一致。64个   int16 2字节，int32 4字节，64x8/2；double 8字节。64x8
	/**
	 * 该步的目的是计算初始线程索引，其中threadIdx, blockIdx, blockDim都是内置变量
	 * threadIdx是存储线程信息的结构体，对于线程0来说，threadIdx.x=0;
	 * blockDim.x表示block在x维度的线程数量，本例中使用的是一维线程块，
	 *     因此只需用到blockDim.x
	 * blockIdx.x表示block的索引，对于第一个线程块来说，blockIdx.x=0;
	 *     对于第二个线程块来说，blockIdx.x=1...
	 * 在计算tid线程索引时，需要要在threadIdx.x 的基础上加上一个基地址，
	 *     实际上就是将二维索引空间转换为线性空间
	 * 64个block，每个block256个thread.
	 */
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	/** blockdim 256
	对于共享变量来说，由于编译器将为 每个 线程块生成共享变量的一个副本，因此只需根据线程块中线程的数量来分配内存
	对于单个块，该共享内存缓存中的偏移就等于线程索引。
	 * 共享内存缓存中的偏移就等于线程索引，
	 * * * 线程块索引与该偏移无关，
	 * 因为每个线程块都拥有该共享内存的私有副本
	 */
	int cacheIndex = threadIdx.x;
	//
	int count = 0;
	double x, temp = 0;
	// 一共N份，tid=0,1...
	while (tid < N) {
		x = *a + (double)(*b - *a) / N * (tid + 0.5);
		temp += function_for_gpu(x);
		/**
		 * 在每个线程计算完当前索引上的任务后，需要对索引进行递增，
		 * 其中，递增的步长为线程格中正在运行的线程数量，
		 * 这个数值等于线程块中的线程数量乘以线程格中线程块的数量，
		 * 即 blockDim.x * gridDim.x
		 *
		 * 该方法类似于多CPU或多核CPU的并行，数据迭代的增量不是1，
		 *     而是CPU的数量；在GPU实现中，一般将并行线程数量看做处理器的数量
		 */
		tid += blockDim.x * gridDim.x;
	}
	count += 1;
	/*一个block的内存都存储，完全同步后循环4096次。
	设置cache中相应位置上的值*/
	cache[cacheIndex] = temp;

	/**
	 * 对线程块中的线程进行同步，该操作用于确保
	 *     所有对共享数组cache[]写入操作在读取cache之前完成
	 */
	__syncthreads();//每个块里32线程，每个kernel负责一个块，串行

	/**
	 * 对于归约运算来说，以下代码要求threadsPerBlock必须是2的幂，
	 * 因为每次合并，要求分成的两部分数组的长度要一致
	 * 基本思想：
	 *     每个线程将cache[]中的两个值相加起来，然后将结果保存回cache[]
	 *     由于每个线程都将两个值合并为一个值，那么在完成这个步骤后，
	 *     得到的结果数量就是计算开始时数值数量的一半。接着，对这一半
	 *     进行相同操作，直到cache[]中256个值归约为1个值
	 */
	
	int i = blockDim.x / 2;//两个块各32个缓存组
	while (i != 0) {
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	//将这个值保存到全局内存后，内核函数结束
	//这里使用了索引为0的线程将cache[0]写入全局内存
	if (cacheIndex == 0)
		integral[blockIdx.x] = cache[0];
}

/**
 * 在CPU上使用积分法串行计算PI
 * @param a 积分下界
 * @param b 积分上界
 * @param integral 存储积分结果
 */
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
	HANDLE_ERROR(cudaMalloc((void**)&dev_partial_integral,blocksPerGrid * sizeof(double)));

	//将'a'和'b'复制到GPU
	HANDLE_ERROR(cudaMemcpy(dev_a, &a, sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, &b, sizeof(double), cudaMemcpyHostToDevice));

	//启动内核函数，启动的线程块的数量是64个，每个线程块的线程数量是256
	trap << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_partial_integral);

	//将计算结果数组'dev_partial_integral'从GPU复制到CPU
	HANDLE_ERROR(cudaMemcpy(partial_integral, dev_partial_integral,
		blocksPerGrid * sizeof(double),	cudaMemcpyDeviceToHost));

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
