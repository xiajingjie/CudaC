#include<stdio.h>
#include<string.h>
#include<math.h>
#include<engine.h>
#include<mat.h>
#include<matrix.h>
#define K 2 //组
#define M 2	//行
#define N 3 //列

int main()
{	
	double B[2][2][3] = {	1.21,1.41,1.61,
							2.52,2.11,2.51,

							31.21,31.41,31.61,
							31.51,32.12,32.51
						};
	double E[2][3];
	for (int k = 0; k < K; k++)//组2
	{
		for (int i = 0; i < M; i++)//行2
		{
			for (int j = 0; j < N; j++)//列3
			{
				//printf("%d  ", (M*N)*k + i * N + j);//行优先C
				E[i][j] = B[k][i][j];//列优先matlab  因为要写入MATLAB中，先写到一维
				printf("%f ", E[k][j]);
			}
			printf("\n");
		}
		printf("\n");
	}

	// 生成.mat文件    
	double outA[K*M*N] = {0.0};
	printf("writing new array as follows:\n");
	for (int k = 0; k < K; k++)//组2
	{
		for (int i = 0; i < M; i++)//行2
		{
			for (int j = 0; j < N; j++)//列3
			{	
				//printf("%d  ", (M*N)*k + i * N + j);//行优先C
				outA[(M*N)*k + j * M + i] = B[k][i][j];//列优先matlab  因为要写入MATLAB中，先写到一维
				printf("%f ",B[k][i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}

	//写入mat文件
	MATFile* pmatFile1 = matOpen("AA1.mat", "w");
	if (pmatFile1 == NULL) {
		printf("Error creating file \n");
		return(EXIT_FAILURE);
	}
	//mxArray *pMxArray2 = mxCreateDoubleMatrix(M, N, mxREAL);
	const mwSize dims[3] = {2,3,2};

	mxArray *pMxArray1 = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);

	mxSetDoubles(pMxArray1, outA);

	if (matPutVariable(pmatFile1, "AA1", pMxArray1)!= 0) {
		printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
		return(EXIT_FAILURE);
	}

	if (matClose(pmatFile1) != 0) {
		printf("Error closing file %s\n", pmatFile1);
		return(EXIT_FAILURE);

	}

	printf("\nRead array:\n");
	//读取mat文件
	const char *file = "AA1.mat";
	MATFile *pmatFile2 = matOpen(file, "r");
	//MATFile *pmatFile2 = matOpen("J:\\CudaCode\\read_mat_plot\\read_mat_plot\\A.mat", "r");

	if (pmatFile2 == 0)
	{
		printf("Open .mat Failed!");
	}
	else
		printf("success read .mat");

	mxArray* pMxArray2 = matGetVariable(pmatFile2, "AA1");
	int M1 = mxGetM(pMxArray2);
	int N1 = mxGetN(pMxArray2);
	printf("行和列：%d,%d\n", M1, N1);

	mxDouble* initA;//一维指针
	initA = mxGetDoubles(pMxArray2);
	printf("%f", initA[1]);
	printf("\n");

	//double(*A)[M][N] = (double(*)[M][N])malloc(sizeof(double) * K * M * N);
	double A[K][M][N];
	for (int k = 0; k < K; k++)//组2
	{
		for (int i = 0; i < M; i++)//行2
		{
			for (int j = 0; j < N; j++)//列3
			{
				//printf("%d  ", (M*N)*k + i * N + j);//行优先C
				//*(*(*(A + k) + i) + j) = initA[(M*N)*k + j * M + i];
				A[k][i][j] = initA[(M*N)*k + j * M + i];//从一维里恢复出C的三维

				printf("%f  ", A[k][i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}

	matClose(pmatFile2);
	mxFree(initA);
	printf("\n");

	return 0;
}