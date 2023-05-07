// 画图 sine
#include<stdio.h>
#include<string.h>
#include<math.h>
#include<engine.h>

#define dataNum 100
int main()
{
	int ret = 0;

	//判断Matlab画图引擎是否打开 
	Engine* plot_egine = NULL;
	if (!(plot_egine = engOpen(NULL)))
	{
		printf("Open matlab enging fail!");
		return 1;
	}
	printf("Init Success\n");

	// 代码加载
	double xtemp[dataNum] = { 0 };
	double ytemp[dataNum] = { 0 };
	for (int i = 0; i < dataNum; i++)
	{
		xtemp[i] = i * 2.0 * 3.1415926 / 100.0;
		ytemp[i] = sin(xtemp[i]);

	}

	mxArray *X = mxCreateDoubleMatrix(1, dataNum, mxREAL);//创建matlab存储数据的指针
	mxArray *Y = mxCreateDoubleMatrix(1, dataNum, mxREAL);

	memcpy(mxGetPr(X), xtemp, dataNum * sizeof(double));
	//数据复制
	memcpy(mxGetPr(Y), ytemp, dataNum * sizeof(double));

	if ((ret = engPutVariable(plot_egine, "X", X)) != 0)   //把数据传递到matlab工作空间,并命名为X
		printf("engPutVariable error：%d\n", ret);
	if ((ret = engPutVariable(plot_egine, "Y", Y)) != 0)
		printf("engPutVariable error：%d\n", ret);

	engEvalString(plot_egine, "plot(X,Y)"); //运行绘图命令,engEvalString(plot_egine, "hold on");也行
	getchar();//直接读取缓冲区中的字符,直到缓冲区中的字符读完为后,才等待用户按键，用户不按键就中断
	if (plot_egine)
		engClose(plot_egine);
	return 0;
}
