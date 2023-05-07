#pragma once
class CJet
{
public:
	double*		R;
	double*		G;
	double*		B;

public:
	CJet(int N = 256);
	~CJet();

public:
	void Destory();
};

