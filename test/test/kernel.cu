#include <stdio.h>    
#include <stdlib.h>    
int main()
{
	int i, j, k;
	int value = 1;
	//申请一个3*3*3的整型数组    
	int(*a)[3][3] = (int(*)[3][3])malloc(sizeof(int) * 3 * 3 * 3);


// 遍历三维数组赋值
	for (i = 0; i < 3; i++)
		for (j = 0; j < 3; j++)
			for (k = 0; k < 3; k++)
				*(*(*(a + i) + j) + k) = value++;
	// 遍历三维数组 打印元素值
	for (i = 0; i < 3; i++)
		for (j = 0; j < 3; j++)
			for (k = 0; k < 3; k++)
				printf("values[%d][%d][%d] = %d\n", i, j, k, a[i][j][k]);
	// 堆空间回收
	free(a);
	return 0;
}