#include "Jet.h"


CJet::CJet(int N/* = 256*/)
{
	R = new double[N];
	G = new double[N];
	B = new double[N];

	double step = 1.0 / N * 4;
	int seg = N / 8;

	int flagR, flagG, flagB;
	for (int k = 0; k < N; k++)
	{
		if (k < seg)
		{
			flagR = 0;
			flagG = 0;
			flagB = 1;
		}
		else if (k < 3 * seg)
		{
			flagR = 0;
			flagG = 1;
			flagB = 0;
		}
		else if (k < 5 * seg)
		{
			flagR = 1;
			flagG = 0;
			flagB = -1;
		}
		else if (k < 7 * seg)
		{
			flagR = 0;
			flagG = -1;
			flagB = 0;
		}
		else
		{
			flagR = -1;
			flagG = 0;
			flagB = 0;
		}

		if (k == 0)
		{
			R[0] = 0;
			G[0] = 0;
			B[0] = 0.5 + step;
		}
		else
		{
			R[k] = R[k - 1] + flagR*step;
			G[k] = G[k - 1] + flagG*step;
			B[k] = B[k - 1] + flagB*step;
		}
	}
}


CJet::~CJet()
{
	Destory();
}


void CJet::Destory()
{
	if (R)		delete[] R;
	if(G)		delete[] G;
	if(B)		delete[] B;

	R = nullptr;
	G = nullptr;
	B = nullptr;
}