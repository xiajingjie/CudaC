#include <stdio.h>    
#include <stdlib.h>    
int main()
{
	int i, j, k;
	int value = 1;
	//����һ��3*3*3����������    
	int(*a)[3][3] = (int(*)[3][3])malloc(sizeof(int) * 3 * 3 * 3);


// ������ά���鸳ֵ
	for (i = 0; i < 3; i++)
		for (j = 0; j < 3; j++)
			for (k = 0; k < 3; k++)
				*(*(*(a + i) + j) + k) = value++;
	// ������ά���� ��ӡԪ��ֵ
	for (i = 0; i < 3; i++)
		for (j = 0; j < 3; j++)
			for (k = 0; k < 3; k++)
				printf("values[%d][%d][%d] = %d\n", i, j, k, a[i][j][k]);
	// �ѿռ����
	free(a);
	return 0;
}