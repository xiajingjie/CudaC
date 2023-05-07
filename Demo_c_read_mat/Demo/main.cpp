// Read data from MAT-file and store them in Mat
// 2016-12-30

#include <Windows.h>
#include <mat.h>
#include "Jet.h"

#include <iostream>
using namespace std;

#include <opencv2/opencv.hpp>
using namespace cv;


/*
	data.mat
	
	Name				Value
	mat2D				4*5 double
	mat3D				2*4*3 double
	lena					1*1 struct ------------- Field						Value
																lenaColor				400*500*3 uint8
																lenaGray				400*500 uint8
*/


void main()
{
	// Name of MAT-file
	const char* filename = "data\\data.mat";
	// Open the MAT-file
	MATFile* pMATFile = matOpen(filename, "r");

	// Define some variables for globally using
	int D = 0;			// Dimensions
	int M = 0;			// Rows
	int N = 0;			// Column


///////////////////// mat2D /////////////////////
	// Get variable "mat2D" from MAT-file
	mxArray* p2D = matGetVariable(pMATFile, "mat2D");
	// Get real part of variable
	double* p2DReal = mxGetPr(p2D);
	// Get rows
	M = mxGetM(p2D);
	// Get Cols
	N = mxGetN(p2D);
	// Define a Mat to store data
	Mat mat2D(M, N, CV_64FC1);
	// Storing...
	for (int i = 0; i < M; i++)
	{
		double* pData = mat2D.ptr<double>(i);
		for (int j = 0; j < N; j++)
			pData[j] = p2DReal[j*M + i];
	}

	// Show the storage mode of "mat2D" in mxArray
	cout << "mat2D in mxArray:\n";
	for (int k = 0; k < M*N; k++)
		cout << *(p2DReal++) << "\t";
	cout << "\n\n";

	// Show the storage mode of "mat2D" Mat
	cout << "mat2D in Mat:\n" << mat2D << "\n\n";


///////////////////// mat3D /////////////////////
	// Get variable "mat3D" from MAT-file
	mxArray* p3D = matGetVariable(pMATFile, "mat3D");
	// Get real part of variable
	double* p3DReal = mxGetPr(p3D);
	// Get dimensions
	D = mxGetNumberOfDimensions(p3D);
	// Get rows
	M = mxGetM(p3D);
	// Get Cols
	N = mxGetN(p3D) / D;
	// Define a Mat to store data
	Mat mat3D(M, N, CV_64FC3);
	// Storing...
	for (int i = 0; i < M; i++)
	{
		double* pData = mat3D.ptr<double>(i);

		for (int j = 0; j < N; j++)
		{
			pData[3 * j + 0] = p3DReal[0 * M*N + j*M + i];
			pData[3 * j + 1] = p3DReal[1 * M*N + j*M + i];
			pData[3 * j + 2] = p3DReal[2 * M*N + j*M + i];
		}

		//for (int j = 0; j < N*D; j++)
		//	pData[j] = p3DReal[(j % D*N + j / D)*M + i];		// Special handling
	}

	// Show the storage mode of "mat2D" in mxArray
	cout << "mat3D in mxArray:\n";
	for (int k = 0; k < M*N*D; k++)
		cout << *(p3DReal++) << "\t";
	cout << "\n\n";

	// Show the storage mode of "mat3D" Mat
	cout << "mat3D in Mat:\n" << mat3D << "\n\n";


///////////////////// lenaGray /////////////////////
	// Get variable "lena" from MAT-file, here "lena" is a structure
	mxArray* pStruct = matGetVariable(pMATFile, "lena");

	// Get field "lenaGray" from structure "lena"
	mxArray* pGray = mxGetField(pStruct, 0, "lenaGray");
	uchar* pGrayReal = (uchar*)mxGetPr(pGray);
	M = mxGetM(pGray);
	N = mxGetN(pGray);
	Mat lenaGray(M, N, CV_8UC1);
	for (int i = 0; i < M; i++)
	{
		uchar* pData = lenaGray.ptr<uchar>(i);
		for (int j = 0; j < N; j++)
			pData[j] = pGrayReal[j*M + i];
	}

	// Show Image
	imshow("lenaGray", lenaGray);


///////////////////// lenaColor /////////////////////
	// Get field "lenaColor" from structure "lena"
	mxArray* pColor = mxGetField(pStruct, 0, "lenaColor");
	uchar* pColorReal = (uchar*)mxGetPr(pColor);
	D = mxGetNumberOfDimensions(pColor);
	M = mxGetM(pColor);
	N = mxGetN(pColor) / D;
	Mat lenaColor(M, N, CV_8UC3);
	for (int i = 0; i < M; i++)
	{
		uchar* pData = lenaColor.ptr<uchar>(i);

		for (int j = 0; j < N; j++)
		{
			pData[3 * j + 0] = pColorReal[2 * M*N + j*M + i];
			pData[3 * j + 1] = pColorReal[1 * M*N + j*M + i];
			pData[3 * j + 2] = pColorReal[0 * M*N + j*M + i];
		}
			
		/*for (int j = 0; j < N*D; j++)
			pData[j] = pColorReal[(j % D*N + j / D)*M + i];*/
	}

	// Show image
	imshow("lenaColor", lenaColor);
	
	// Close MAT-file
	matClose(pMATFile);


///////////////////// Gray2Jet /////////////////////
	// Initialize jet
	CJet jet;
	// Define a Mat to store data
	Mat lenaJet(400, 500, CV_8UC3);
	for (int i = 0; i < 400; i++)
	{
		uchar* pDataJet = lenaJet.ptr<uchar>(i);
		uchar* pDataGray= lenaGray.ptr<uchar>(i);
		for (int j = 0; j < 500; j++)
		{
			pDataJet[3 * j + 0] = jet.B[pDataGray[j]] * 255;
			pDataJet[3 * j + 1] = jet.G[pDataGray[j]] * 255;
			pDataJet[3 * j + 2] = jet.R[pDataGray[j]] * 255;
		}
	}

	//Destory jet
	jet.Destory();

	// Show image
	namedWindow("lenaJet", WINDOW_KEEPRATIO);
	imshow("lenaJet", lenaJet);

	// 调整窗口尺寸和位置，需包含头文件Windows.h
	HWND hWnd = (HWND)cvGetWindowHandle("lenaJet");
	HWND hParentWnd = GetParent(hWnd);
	SetWindowPos(hParentWnd, NULL, 0, 0, 800, 600, SWP_NOMOVE);


	waitKey();
}
