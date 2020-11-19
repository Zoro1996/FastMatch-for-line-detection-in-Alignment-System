#include "MyFunction.h"
#include <opencv2/ximgproc.hpp>


vector<int> GetFreemanDeltaCode(Mat img)
{
	vector<int>freemanCode, freemanDeltaCode, freemanDeltaNormCode;
	CvChain* chain = 0;
	CvMemStorage* storage = 0;
	storage = cvCreateMemStorage(0);
	cvFindContours(&IplImage(img), storage, (CvSeq**)(&chain), sizeof(*chain), CV_RETR_LIST, CV_CHAIN_CODE);
	for (; chain != NULL; chain = (CvChain*)chain->h_next)
	{
		CvSeqReader reader;
		int i, total = chain->total;
		cvStartReadSeq((CvSeq*)chain, &reader, 0);
		printf("--------------------chain\n");
		for (i = 0; i < total; i++)
		{ 
			char code;
			CV_READ_SEQ_ELEM(code, reader);
			int x = (int)(code);
			freemanCode.push_back(x);
			if (i > 0)
			{
				int deltaCodeValue = (freemanCode[i] - freemanCode[i - 1] + 8) % 8;
				if (deltaCodeValue!=0)
				{
					freemanDeltaCode.push_back(deltaCodeValue);
				}
			}
		}
	}

	unsigned int bestF = INT_MAX;
	unsigned int bestIndex = 0;




	vector<int>resultCode;
	for (int i = 0; i < 10; i++)
	{
		resultCode.push_back(freemanDeltaCode[bestIndex + i]);
	}

	return freemanDeltaCode;
}


int main(int argc, char *argv[])
{
	int index = 2;
	char  srcPath[100];

	Mat maskImageL = imread("F:\\数据集\\对位\\3号台圆弧盖板图像\\maskImageL.bmp", 0);
	Mat srcImageL = imread("F:\\数据集\\对位\\3号台圆弧盖板图像\\对位2\\F01-20201114223316-1.bmp", 0);

	Mat dst1,dst2;
	maskImageL = maskImageL > 150;
	Canny(maskImageL, dst1, 150, 200);

	srcImageL = srcImageL > 150;
	Canny(srcImageL, dst2, 150, 200);

	//dst = Mat::zeros(Size(1000, 1000), CV_8UC1);
	//Mat resultImage;
	//cvtColor(dst, resultImage, CV_GRAY2BGR);
	//circle(dst, Point2f(dst.cols / 2, dst.rows / 2),
	//	50, Scalar(255), -1);
	//Rect region = Rect(100, 100, 500, 500);
	//rectangle(dst, region, Scalar(255), 1, 8);
	//line(dst, Point2f(100, 100), Point2f(100, 500), Scalar(255), 1);
	//dst1 = maskImageL.clone();
	//dst2 = srcImageL.clone();

	//vector<int>maskFreemanCode, maskFreemanDeltaCode, maskFreemanDeltaNormCode;
	//vector<int>freemanCode, freemanDeltaCode, freemanDeltaNormCode;
	//maskFreemanDeltaCode = GetFreemanDeltaCode(dst1);
	//freemanDeltaCode = GetFreemanDeltaCode(dst2);
	/*Mat resultImage = Mat::zeros(Size(2000, 2000), srcImageL.type());
	Point2f pt = Point2f(1000, 1000);
	resultImage.at<uchar>(pt) = 255;
	for (int i = 0; i < freemanCodeSet.size(); i++)
	{
		int code = freemanCodeSet[i];
		if (code == 0)pt = Point2f(pt.x + 1, pt.y);
		if (code == 1)pt = Point2f(pt.x + 1, pt.y - 1);
		if (code == 2)pt = Point2f(pt.x, pt.y - 1);
		if (code == 3)pt = Point2f(pt.x - 1, pt.y - 1);
		if (code == 4)pt = Point2f(pt.x - 1, pt.y);
		if (code == 5)pt = Point2f(pt.x - 1, pt.y + 1);
		if (code == 6)pt = Point2f(pt.x, pt.y + 1);
		if (code == 7)pt = Point2f(pt.x + 1, pt.y + 1);
		resultImage.at<uchar>(pt) = 255;
	}*/

	vector<vector<Point> > contours1, contours2;
	vector<Vec4i> hierarchy1,hierarchy2;
	findContours(dst1, contours1, hierarchy1, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	findContours(dst2, contours2, hierarchy2, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	//for (int ci = 0; ci < contours2.size(); ci++)
	//{
	//	for (int cj = 0; cj < contours2[cj].size(); cj++)
	//	{
	//		for (int ck = 0; ck < contours2[cj].size(); ck++)
	//		{
	//		}
	//	}
	//}

	float score;
	float bestScore = INFINITY;
	int bestPosition, bestIndexContours;
	vector<Point2f>bestContours;
	int i, j, k;
	for (i = 0; i < contours2.size(); i++)
	{
		for (j = 0; j < contours2[i].size() - contours1.size(); j++)
		{
			vector<Point2f>contours2_part;
			for (k = 0; k < contours1[0].size(); k++)
			{
				contours2_part.push_back(contours2[i][k + j]);
			}

			score = matchShapes(contours1[0], contours2_part, CV_CONTOURS_MATCH_I1, 0);
			if (score < bestScore)
			{
				bestScore = score;
				bestPosition = j;
				bestIndexContours = i;
				bestContours = contours2_part;
			}
		}
	}

	//Mat resultImage = Mat::zeros(Size(5000, 5000), srcImageL.type());
	Mat resultImage;
	cvtColor(srcImageL, resultImage, CV_GRAY2BGR);
	for (int m = 0; m < bestContours.size(); m++)
	{
		//resultImage.at<uchar>(bestContours[m]) = 255;
		resultImage.at<Vec3b>(bestContours[m])[0] = 0;
		resultImage.at<Vec3b>(bestContours[m])[1] = 0;
		resultImage.at<Vec3b>(bestContours[m])[2] = 255;
	}

	waitKey(0);
	return 0;
}