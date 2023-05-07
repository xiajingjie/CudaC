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

const int N = 1024 * 1024 * 64;     	//����ʱ���ֵķ���
const int threadsPerBlock = 256;	//block�е��߳���
const int blocksPerGrid = 64;		//grid�е�block��

//ȱʡ__host__������CPU���У�CPU����
double function_for_cpu(double x) {
	return 4 / (1 + x * x);
}

//__device__���Σ�ֻ�ܱ��ں˺������ã�����GPU���У�GPU����
__device__ double function_for_gpu(double x) {
	return 4 / (1 + x * x);
}

/**
 * __global__���Σ��ں˺���������GPU���У�CPU����
 * ��GPU��ʹ�û��ַ����м���PI
 * @param a �����½�
 * @param b �����Ͻ�
 * @param integral �洢���ֽ��
 * @return
 */
__global__ void trap(double *a, double *b, double *integral) {
	/**
	 * __shared__���Σ�����һ�������ڴ滺����������Ϊcache
	 * ��ʾ���ݴ���ڹ���洢���У�ÿ���߳̿鶼�иñ�����һ��������
	 * ֻ���ڿ��ڵ��߳̿��Է��ʣ��������ڵ��̲߳��ܷ��ʣ�
	 */
	__shared__ double cache[threadsPerBlock];
	//�����ڴ��б�֤����Ĺ����ڴ�Ĵ�С���߳���һ�¡�64��   int16 2�ֽڣ�int32 4�ֽڣ�64x8/2��double 8�ֽڡ�64x8
	/**
	 * �ò���Ŀ���Ǽ����ʼ�߳�����������threadIdx, blockIdx, blockDim�������ñ���
	 * threadIdx�Ǵ洢�߳���Ϣ�Ľṹ�壬�����߳�0��˵��threadIdx.x=0;
	 * blockDim.x��ʾblock��xά�ȵ��߳�������������ʹ�õ���һά�߳̿飬
	 *     ���ֻ���õ�blockDim.x
	 * blockIdx.x��ʾblock�����������ڵ�һ���߳̿���˵��blockIdx.x=0;
	 *     ���ڵڶ����߳̿���˵��blockIdx.x=1...
	 * �ڼ���tid�߳�����ʱ����ҪҪ��threadIdx.x �Ļ����ϼ���һ������ַ��
	 *     ʵ���Ͼ��ǽ���ά�����ռ�ת��Ϊ���Կռ�
	 * 64��block��ÿ��block256��thread.
	 */
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	/** blockdim 256
	���ڹ��������˵�����ڱ�������Ϊ ÿ�� �߳̿����ɹ��������һ�����������ֻ������߳̿����̵߳������������ڴ�
	���ڵ����飬�ù����ڴ滺���е�ƫ�ƾ͵����߳�������
	 * �����ڴ滺���е�ƫ�ƾ͵����߳�������
	 * * * �߳̿��������ƫ���޹أ�
	 * ��Ϊÿ���߳̿鶼ӵ�иù����ڴ��˽�и���
	 */
	int cacheIndex = threadIdx.x;
	//
	int count = 0;
	double x, temp = 0;
	// һ��N�ݣ�tid=0,1...
	while (tid < N) {
		x = *a + (double)(*b - *a) / N * (tid + 0.5);
		temp += function_for_gpu(x);
		/**
		 * ��ÿ���̼߳����굱ǰ�����ϵ��������Ҫ���������е�����
		 * ���У������Ĳ���Ϊ�̸߳����������е��߳�������
		 * �����ֵ�����߳̿��е��߳����������̸߳����߳̿��������
		 * �� blockDim.x * gridDim.x
		 *
		 * �÷��������ڶ�CPU����CPU�Ĳ��У����ݵ�������������1��
		 *     ����CPU����������GPUʵ���У�һ�㽫�����߳���������������������
		 */
		tid += blockDim.x * gridDim.x;
	}
	count += 1;
	/*һ��block���ڴ涼�洢����ȫͬ����ѭ��4096�Ρ�
	����cache����Ӧλ���ϵ�ֵ*/
	cache[cacheIndex] = temp;

	/**
	 * ���߳̿��е��߳̽���ͬ�����ò�������ȷ��
	 *     ���жԹ�������cache[]д������ڶ�ȡcache֮ǰ���
	 */
	__syncthreads();//ÿ������32�̣߳�ÿ��kernel����һ���飬����

	/**
	 * ���ڹ�Լ������˵�����´���Ҫ��threadsPerBlock������2���ݣ�
	 * ��Ϊÿ�κϲ���Ҫ��ֳɵ�����������ĳ���Ҫһ��
	 * ����˼�룺
	 *     ÿ���߳̽�cache[]�е�����ֵ���������Ȼ�󽫽�������cache[]
	 *     ����ÿ���̶߳�������ֵ�ϲ�Ϊһ��ֵ����ô�������������
	 *     �õ��Ľ���������Ǽ��㿪ʼʱ��ֵ������һ�롣���ţ�����һ��
	 *     ������ͬ������ֱ��cache[]��256��ֵ��ԼΪ1��ֵ
	 */
	
	int i = blockDim.x / 2;//�������32��������
	while (i != 0) {
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	//�����ֵ���浽ȫ���ڴ���ں˺�������
	//����ʹ��������Ϊ0���߳̽�cache[0]д��ȫ���ڴ�
	if (cacheIndex == 0)
		integral[blockIdx.x] = cache[0];
}

/**
 * ��CPU��ʹ�û��ַ����м���PI
 * @param a �����½�
 * @param b �����Ͻ�
 * @param integral �洢���ֽ��
 */
void trap_by_cpu(double a, double b, double *integral) {
	int i;
	double x, temp = 0;
	for (i = 0; i < N; i++) {
		x = a + (double)(b - a) / N * (i + 0.5);//����
		temp += function_for_cpu(x);
	}
	temp *= (double)(b - a) / N;
	*integral = temp;
}

int main(void) {
	// ������
	double   integral;
	double	*partial_integral;
	double a, b;
	// �豸��
	double   *dev_partial_integral;
	double *dev_a, *dev_b;
	cudaEvent_t start, stop;
	float tm;
	// ��ʱ
	clock_t  clockBegin, clockEnd;
	float duration;

	a = 0;
	b = 1;

	//ʹ��CPU����PI��ֵ
	clockBegin = clock();
	trap_by_cpu(a, b, &integral);
	clockEnd = clock();
	duration = (float)1000 * (clockEnd - clockBegin) / CLOCKS_PER_SEC;
	printf("CPU Result: %.6f\n", integral);
	printf("CPU Elapsed time: %.6fms\n", duration);

//	getchar();

	//ʹ��GPU+CPU����PI��ֵ
	//ʹ��event����ʱ��
	cudaEventCreate(&start); //����event
	cudaEventCreate(&stop);  //����event
	cudaEventRecord(start, 0);  //��¼��ǰʱ��

	// ����CPU�洢�ռ�
	partial_integral = (double*)malloc(blocksPerGrid * sizeof(double));

	// ����GPU�洢�ռ�
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_partial_integral,blocksPerGrid * sizeof(double)));

	//��'a'��'b'���Ƶ�GPU
	HANDLE_ERROR(cudaMemcpy(dev_a, &a, sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, &b, sizeof(double), cudaMemcpyHostToDevice));

	//�����ں˺������������߳̿��������64����ÿ���߳̿���߳�������256
	trap << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_partial_integral);

	//������������'dev_partial_integral'��GPU���Ƶ�CPU
	HANDLE_ERROR(cudaMemcpy(partial_integral, dev_partial_integral,
		blocksPerGrid * sizeof(double),	cudaMemcpyDeviceToHost));

	//��CPU�Ͻ��й�Լ�������õ����յļ�����
	integral = 0;
	for (int i = 0; i < blocksPerGrid; i++) {
		integral += partial_integral[i];
	}
	integral *= (double)(b - a) / N;

	cudaEventRecord(stop, 0);  //��¼��ǰʱ��
	cudaEventSynchronize(stop);  //�ȴ�stop event���
	cudaEventElapsedTime(&tm, start, stop);  //����ʱ�����뼶��
	printf("GPU Result: %.20lf\n", integral);
	printf("GPU Elapsed time:%.6f ms.\n", tm);

	//�ͷ�GPU�ڴ�
	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_partial_integral));

	//�ͷ�CPU�ڴ�
	free(partial_integral);

	getchar();
}
