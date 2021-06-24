#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Chroma.h"

using namespace std;
using namespace cv;

int main() {
	Mat Input_Image = imread("aaa.png");
	Mat Input_Image2 = imread("aaa2.png");

	cout << "Height: " << Input_Image.rows << ", Width: " << Input_Image.cols << ", Channels: " << Input_Image.channels() << endl;
	cout << "Height: " << Input_Image2.rows << ", Width: " << Input_Image2.cols << ", Channels: " << Input_Image2.channels() << endl;

	Image_Chroma_CUDA(Input_Image.data, Input_Image2.data, Input_Image.rows, Input_Image.cols, Input_Image.channels());

	imwrite("chroma.png", Input_Image);
	
	system("pause");
	return 0;
}