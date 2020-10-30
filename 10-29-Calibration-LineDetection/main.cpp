#include "MyFunction.h"


int main(int argc, char *argv[])
{
	int index = 2;
	char  srcPath[100], maskPath[100];
	sprintf_s(maskPath, "E:\\dataset\\TEST-BD\\templateLineL.bmp");

	for (int index = 1; index < 10; index++)
	{
		sprintf_s(srcPath, "E:\\dataset\\TEST-BD\\BD\\ImgsL\\T-L-%d.jpeg", index);
		//sprintf_s(srcPath, "E:\\dataset\\TEST-BD\\R\\R0.jpeg");

		Mat src = imread(srcPath, 0);
		Mat srcRGB = imread(srcPath, 1);

		Mat mask = imread(maskPath, 0);
		Mat maskRGB = imread(maskPath, 1);

		if (src.empty() || srcRGB.empty() || mask.empty())
		{
			cout << "load image failed !" << endl;
		}

		Mat srcImage = src.clone();
		Mat maskImage = mask.clone();

		/*计算最佳变换参数*/
		float delta = 0.1;
		float epsilon = 1;
		float factor = 0.5;
		float resolution = 2 / srcImage.rows;


		clock_t t5 = clock();

		tuple<int, int, float> bestTrans = FastMatch(srcImage, maskImage, delta, epsilon, factor, resolution);

		int tx = 2 * get<0>(bestTrans);
		int ty = 2 * get<1>(bestTrans);
		float theta = get<2>(bestTrans);

		Rect maskRegion = Rect(tx, ty, 2 * maskImage.cols, 2 * maskImage.rows);
		Mat srcRegion;

		src(maskRegion).copyTo(srcRegion);

		Point2f circleRegionPoint = GetCrossPoint(srcRegion);
		Point2f circlePoint = Point2f(tx + circleRegionPoint.x, ty + circleRegionPoint.y);

		clock_t t6 = clock();
		cout << "Total time is :" << (t6 - t5)* 1.0 / CLOCKS_PER_SEC << "s\n" << endl;

		circle(srcRGB, circlePoint, 8, Scalar(0, 0, 255), -1);

		namedWindow(srcPath, 0);
		imshow(srcPath, srcRGB);
	}

	waitKey(0);
	return 0;
}