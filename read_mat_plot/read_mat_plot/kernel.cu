#include<stdio.h>
#include<string.h>
#include<math.h>
#include<engine.h>
#include<mat.h>
#include<matrix.h>
#define M 2
#define N 3

int main()
{	
	double B[2][3] = { 1,2.1,3.4,2.5,6.66,7.3 };
	// 生成.mat文件    
	double *outA = new double[M*N];
	printf("writing new array as follows:\n");
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			outA[M*j + i] = B[i][j];
			printf("%f  ", B[i][j]);
		}
		printf("\n");
	}

	MATFile* pmatFile2 = matOpen("A1.mat", "w");
	if (pmatFile2 == NULL) {
		printf("Error creating file \n");
		return(EXIT_FAILURE);
	}
	
//	mxArray *pMxArray2 = mxCreateDoubleMatrix(M, N, mxREAL);

	const mwSize dims[3] = { 2,3 };
	mxArray *pMxArray2 = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);

	mxSetDoubles(pMxArray2, outA);

	if (matPutVariable(pmatFile2, "A1", pMxArray2)!= 0) {
		printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
		return(EXIT_FAILURE);
	}

	if (matClose(pmatFile2) != 0) {
		printf("Error closing file %s\n", pmatFile2);
		return(EXIT_FAILURE);

	}

	printf("\nRead array:\n");

	// MATFile *pmatFile = matOpen("J:\\CudaCode\\read_mat_plot\\read_mat_plot\\initUrban1.mat", "r");
	MATFile *pmatFile = matOpen("J:\\CudaCode\\read_mat_plot\\read_mat_plot\\A1.mat", "r");

	if (pmatFile == 0)
	{
		printf("Open .mat Failed!");
	}
	else
		printf("success read .mat");

	//mxArray* pMxArray = matGetVariable(pmatFile, "initA");
	mxArray* pMxArray = matGetVariable(pmatFile, "A1");

	mxDouble*initA;
	initA = mxGetDoubles(pMxArray);

	int M1 = mxGetM(pMxArray);
	int N1 = mxGetN(pMxArray);
	printf("行和列：%d,%d\n", M1, N1);

	double A[2][3] = { 0.0 };
	for (int i = 0; i < M1; i++)
	{
		for (int j = 0; j < N1; j++)
		{
			A[i][j] = initA[M1*j + i];
			printf("%f  ", A[i][j]);
		}
		printf("\n");
	}
	matClose(pmatFile);
	mxFree(initA);
	printf("\n");

	Engine* plot_egine = NULL;
	if (!(plot_egine = engOpen(NULL)))
	{
		printf("Open matlab enging fail!");
		return 1;
	}
	printf("Init Success\n");
	if (engPutVariable(plot_egine, "X", pMxArray2) != 0)   //把数据传递到matlab工作空间,并命名为X
		printf("engPutVariable error\n");
	engEvalString(plot_egine, "plot(X(1,:),'*')"); //运行绘图命令,engEvalString(plot_egine, "hold on");也行
	getchar();//直接读取缓冲区中的字符,直到缓冲区中的字符读完为后,才等待用户按键，用户不按键就中断
	if (plot_egine)
		engClose(plot_egine);
	return 0;
}