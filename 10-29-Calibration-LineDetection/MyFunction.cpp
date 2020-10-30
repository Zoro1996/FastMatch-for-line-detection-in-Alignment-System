#include "MyFunction.h"


float kern0[3][3] = {
	-1,0,1,
	-2,0,2,
	-1,0,1
};

float kern90[3][3] = {
	-1, -2, -1,
	0, 0, 0,
	1, 2, 1
};


CircleData findCircle2(Point2f pt1, Point2f pt2, Point2f pt3)
{
	float A1, A2, B1, B2, C1, C2, temp;
	A1 = pt1.x - pt2.x;
	B1 = pt1.y - pt2.y;
	C1 = (pow(pt1.x, 2) - pow(pt2.x, 2) + pow(pt1.y, 2) - pow(pt2.y, 2)) / 2;
	A2 = pt3.x - pt2.x;
	B2 = pt3.y - pt2.y;
	C2 = (pow(pt3.x, 2) - pow(pt2.x, 2) + pow(pt3.y, 2) - pow(pt2.y, 2)) / 2;

	temp = A1 * B2 - A2 * B1;

	CircleData CD;

	if (temp == 0) {
		CD.center.x = pt1.x;
		CD.center.y = pt1.y;
	}
	else {
		CD.center.x = (C1*B2 - C2 * B1) / temp;
		CD.center.y = (A1*C2 - A2 * C1) / temp;
	}

	CD.radius = sqrtf((CD.center.x - pt1.x)*(CD.center.x - pt1.x) + (CD.center.y - pt1.y)*(CD.center.y - pt1.y));
	return CD;
}


Point2f GetCrossPoint(Mat&srcImage)
{
	Mat edges;
	Mat dstImage = Mat::zeros(srcImage.size(), srcImage.type());
	// Find the edges in the image using canny detector
	Canny(srcImage, edges, 50, 200);
	// Create a vector to store lines of the image
	vector<Vec4f> lines;
	vector<Point>linePointX, linePointY;
	// Apply Hough Transform
	HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 50, 100);

	// Draw lines on the image
	float epsilon = 0.001;
	for (size_t i = 0; i < lines.size(); i++) {
		Vec4f l = lines[i];
		//line(dstImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 3, LINE_AA);
		if (abs((l[3] - l[1]) / (l[2] - l[0] + epsilon)) > 5)
		{
			linePointY.push_back(Point(l[0], l[1]));
			linePointY.push_back(Point(l[2], l[3]));
		}
		else
		{
			linePointX.push_back(Point(l[0], l[1]));
			linePointX.push_back(Point(l[2], l[3]));
		}
	}

	Vec4f fitLineX, fitLineY;
	//拟合方法采用最小二乘法
	fitLine(linePointX, fitLineX, CV_DIST_HUBER, 0, 0.01, 0.01);
	fitLine(linePointY, fitLineY, CV_DIST_HUBER, 0, 0.01, 0.01);

	float ka, kb;
	ka = (float)(fitLineX[1] / (fitLineX[0] + 1e-6)); //求出LineA斜率
	kb = (float)(fitLineY[1] / (fitLineY[0] + 1e-6)); //求出LineB斜率

	float ma, mb;
	ma = fitLineX[3] - ka * fitLineX[2];
	mb = fitLineY[3] - kb * fitLineY[2];

	Point2f crossPoint;
	crossPoint.x = (mb - ma) / (ka - kb + 1e-6);
	crossPoint.y = (ma * kb - mb * ka) / (kb - ka + 1e-6);

	return crossPoint;
}


Point2f AfterTrans(tuple<int, int, float>&curTrans, Point2f curPoint, float centerX, float centerY)
{
	Point2f resultPoint;

	int transX = get<0>(curTrans);
	int transY = get<1>(curTrans);
	float theta = get<2>(curTrans);

	resultPoint.x = cos(theta) * (curPoint.x - centerX) - sin(theta)*(curPoint.y - centerY) + centerX + transX;
	resultPoint.y = sin(theta) * (curPoint.x - centerX) + cos(theta)*(curPoint.y - centerY) + centerY + transY;

	return resultPoint;
}


float GradientAtCurPoint(Mat&image, Point2f &pt, float kernel[][3])
{
	float gradient = 0;
	for (int x = -1; x < 2; x++)
	{
		for (int y = -1; y < 2; y++)
		{
			float curValue = (float)(image.at<uchar>(pt.y + y, pt.x + x));
			float curKernelValue = (float)(kernel[y + 1][x + 1]);
			gradient += curKernelValue * curValue;
		}
	}

	//返回归一化后的梯度值
	//float result = (gradient + 4 * 255) / (8 * 255);
	float result = gradient / 255;

	return result;
}


Contours ExistedMaskImage(Mat& srcImage, Mat &maskImage)
{
	vector<Point2f> PointSet;
	Contours maskStruct;

	/*计算边缘点的梯度模长和方向*/
	for (int row = 0; row < maskImage.rows; row++)
	{
		for (int col = 0; col < maskImage.cols; col++)
		{
			//变换至待测图中心位置，之后的仿射变换平移原点选在待测图的中心
			Point2f pt = Point2f(col, row);
			maskStruct.pointSet.push_back(pt);
		}
	}
	float centerX = floor(maskImage.cols / 2);
	float centerY = floor(maskImage.rows / 2);
	maskStruct.centerX = centerX;
	maskStruct.centerY = centerY;

	return maskStruct;
}


vector<Gradient> GetTableFromPointSet(vector<Point2f> &pointSet, Mat&image)
{
	Gradient gradient;
	vector<Gradient> curGradientTable;

	float gradientX, gradientY;
	int index = 0;

	for (int i = 0; i < pointSet.size(); i++)
	{
		Point2f pt = pointSet[i];
		gradientX = GradientAtCurPoint(image, pt, kern0);
		gradientY = GradientAtCurPoint(image, pt, kern90);

		gradient.margin = sqrt(pow(gradientX, 2) + pow(gradientY, 2));
		gradient.theta = atan2(gradientX, gradientY);//[-Pi,Pi]

		curGradientTable.push_back(gradient);
	}

	return curGradientTable;
}


/*计算某一变换对应的特征距离*/
int SingleTransEvaluation(Mat &srcImage, Mat& maskImage, vector<Point2f> &subMaskPointSet,
	tuple<int, int, float>& curTrans, float epsilon)
{
	/*获取模板信息*/
	float centerX = floor(maskImage.cols / 2);
	float centerY = floor(maskImage.rows / 2);

	int num = 0;
	for (int i = 0; i < subMaskPointSet.size(); i++)
	{
		Point2f curPoint = subMaskPointSet[i];
		Point2f transPoint = AfterTrans(curTrans, curPoint, centerX, centerY);
		if (transPoint.x >= 0 && transPoint.x < srcImage.cols - 1 &&
			transPoint.y >= 0 && transPoint.y < srcImage.rows - 1)
		{
			float value1 = maskImage.at<float>(curPoint);
			float value2 = srcImage.at<float>(transPoint);

			if (abs(value1 - value2) < 0.2)
			{
				num++;
			}
		}
	}

	return num;
}


/*构建变换网络*/
vector<tuple<int, int, float>> ConstructNet(int row, int col, float delta)
{
	int lowX = 0;//-5472
	int highX = col;
	int lowY = 0;//-3648
	int highY = row;
	float lowR = -PI;
	float highR = PI;

	int tx_step = int(delta * col);
	int ty_step = int(delta * row);
	float theta_step = delta * 2 * PI;

	int tx, ty;
	float r;
	vector<tuple<int, int, float>> Trans;

	for (int tx_index = lowX; tx_index < highX; tx_index += tx_step)
	{
		tx = tx_index;
		for (int ty_index = lowY; ty_index < highY; ty_index += ty_step)
		{
			ty = ty_index;
			for (float r_index = lowR; r_index < highR; r_index += theta_step)
			{
				r = r_index;
				tuple<int, int, float>curTrans{ tx,ty,r };
				Trans.push_back(curTrans);
			}
		}
	}

	return Trans;
}


/*计算当前TransNet下的最佳变换*/
CurrentBestReusult GetBestTrans(Mat& srcImage, Mat& maskImage, vector<Point2f>& subMaskPointSet,
	vector<tuple<int, int, float>>& TransNet, float delta, float epsilon)
{
	CurrentBestReusult bestResult;
	tuple<int, int, float> bestTrans;

	int bestNum = 0;
	for (int i = 0; i < TransNet.size(); i++)
	{
		tuple<int, int, float> curTrans = TransNet[i];
		int curNum = SingleTransEvaluation(srcImage, maskImage, subMaskPointSet, curTrans, epsilon);
		
		bestResult.TransNum.push_back(curNum);

		if (bestNum < curNum)
		{
			bestTrans = curTrans;
			bestNum = curNum;
		}
	}

	bestResult.transX = get<0>(bestTrans);
	bestResult.transY = get<1>(bestTrans);
	bestResult.theta  = get<2>(bestTrans);
	bestResult.bestNum = bestNum;

	return bestResult;
}


vector <tuple<int, int, float >> GetNextNet(Mat&srcImage, vector<tuple<int, int, float >> &GoodTransNet,
	vector<Point2f>&subMaskPointSet, float centerX, float centerY, float delta)
{
	int lowX = -1;
	int highX = 1;
	int lowY = -1;
	int highY = 1;
	float lowR = -1;
	float highR = 1;

	vector<tuple<int, int, float>> nextTransNet;
	tuple<int, int, float> extendedTrans;
	for (int i = 0; i < GoodTransNet.size(); i++)
	{
		nextTransNet.push_back(GoodTransNet[i]);
		for (int outerX = lowX; outerX <= highX; outerX++)
		{
			for (int outerY = lowY; outerY <= highY; outerY++)
			{
				for (float outerTheta = lowR; outerTheta <= highR; outerTheta++)
				{
					if (outerX == 0 && outerY == 0 && outerTheta == 0)
					{
						continue;
					}
					get<0>(extendedTrans) = get<0>(GoodTransNet[i]) + outerX * delta * srcImage.cols;
					get<1>(extendedTrans) = get<1>(GoodTransNet[i]) + outerY * delta * srcImage.rows;
					get<2>(extendedTrans) = get<2>(GoodTransNet[i]) + outerTheta * delta * 2 * PI;
					nextTransNet.push_back(extendedTrans);
				}
			}
		}
	}

	return nextTransNet;
}


/* I1 : mask; I2 : src*/
tuple<int, int, float> FastMatch(Mat &srcImage, Mat &maskImage,
	float delta, float epsilon, float factor,float resolution)
{
	/*Step 0 :Sample the subMaskPointSet, Normalize*/
	cout << "Step 0:Prepare work : Sample subMaskPointSet from the whole maskImage's PointSet." << endl;
	clock_t t1 = clock();

	resize(srcImage, srcImage, Size(srcImage.cols / 2, srcImage.rows / 2));
	resize(maskImage, maskImage, Size(maskImage.cols / 2, maskImage.rows / 2));

	srcImage.convertTo(srcImage, CV_32FC1);
	maskImage.convertTo(maskImage, CV_32FC1);
	maskImage /= 255;
	srcImage /= 255;

	float row = maskImage.rows;
	float col = maskImage.cols;
	float centerX = floor(col / 2);
	float centerY = floor(row / 2);

	//点集均匀采样
	vector<Point2f>subMaskPointSet;
	float sampleRate = 0.1;
	for (float x = 0; x < col; x += col * sampleRate)
	{
		for (float y = 0; y < row; y += row * sampleRate)
		{
			subMaskPointSet.push_back(Point2f(x, y));
		}
	}

	clock_t t2 = clock();
	cout << "Step 0: subMaskPointSet's size is :" << subMaskPointSet.size() << endl;
	cout << "Step0: Prepare work has been finished !" << endl;
	cout << "Time is :" << (t2 - t1) * 1.0 / CLOCKS_PER_SEC << "s\n" << endl;


	/*Step 1 : Construct the N(δ) net */
	//T: [translationX, translationY, Rtate] 
	cout << "Step 1:Construct the N(δ) net." << endl;
	clock_t t3 = clock();
	vector<tuple<int, int, float>> TransNet, GoodTransNet;

	//建立初始网络
	TransNet = ConstructNet(srcImage.rows, srcImage.cols, delta);

	clock_t t4 = clock();
	cout << "Step 1:δis:" << delta << endl;
	cout << "Step 1:Size of the N(δ) net is:" << TransNet.size() << endl;
	cout << "Step 1:Construct the N(δ) net has been finished !" << endl;
	cout << "Time is :" << (t4 - t3)* 1.0 / CLOCKS_PER_SEC << "s\n" << endl;


	/*Step 2: Iterate, update and calculate the best translation.*/
	cout << "Step 2:Iterate, update and calculate the best translation." << endl;
	float distance = 0; 
	float L_Delta;
	int N_Delta;

	int index = 0;

	float bestDistance = FLT_MAX;
	float curDistance = 0;
	float temp_distance;
	vector<float>bestDistanceSet;

	int bestNum = 0;
	float curNum = 0;
	vector<int>bestNumSet;

	vector<tuple<int, int, float>> tempTransNet;
	tuple<int, int, float>  bestTrans;
	int bestTransX, bestTransY;
	float bestTheta;
	CurrentBestReusult bestResult;

	while (true)
	{
		index++;
		cout << "Current TransNet's size is: " << TransNet.size() << endl;

		/*计算当前变换网络下的bestTrans + bestDistance*/
		clock_t ta = clock();
		bestResult = GetBestTrans(srcImage, maskImage, subMaskPointSet, TransNet, delta, epsilon);		clock_t tb = clock();
		cout << "GetBestTrans's time is: " << (tb - ta)* 1.0 / CLOCKS_PER_SEC << "s" << endl;

		//if (bestDistance > bestResult.BestDistance)
		//{
		//	bestDistance = bestResult.BestDistance;
		//	get<0>(bestTrans) = bestResult.transX;
		//	get<1>(bestTrans) = bestResult.transY;
		//	get<2>(bestTrans) = bestResult.theta;
		//}

		if (bestNum < bestResult.bestNum)
		{
			bestNum = bestResult.bestNum;
			get<0>(bestTrans) = bestResult.transX;
			get<1>(bestTrans) = bestResult.transY;
			get<2>(bestTrans) = bestResult.theta;
		}

		//cout << "bestDistance is :" << bestDistance << endl;
		cout << "bestNum is :" << bestNum << endl;

		bestDistanceSet.push_back(bestDistance);
		bestNumSet.push_back(bestNum);

		//if (bestDistanceSet.size() >= 3 && abs(bestDistanceSet[index - 1] - bestDistanceSet[index - 3]) < 0.1)
		//{
		//	break;
		//}

		if (bestNumSet.size() >= 3 && abs(bestNumSet[index - 1] - bestNumSet[index - 3]) < 0.1)
		{
			break;
		}


		/*计算和最佳变换相近的次优解集合GoodTransNet*/
		float curDistance = 0;

		//L_Delta = alpha * bestDistance + beta;
		float alpha = 0.2, beta = 0.01;

		//N_Delta = (int)(alpha * N_Delta);
		N_Delta = (int)((1 - sampleRate * sampleRate*bestNum) / (sampleRate*sampleRate));

		for (int i = 0; i < TransNet.size(); i++)
		{
			tuple<int, int, float> curTrans = TransNet[i];//获取当前变换

			//curDistance = bestResult.TransDistance[i];
			curNum = bestResult.TransNum[i];

			if (abs(curNum - bestNum) < N_Delta)
			{
				tempTransNet.push_back(curTrans);
			}
		}

		//清空GoodTransNet，存入当前的tempTransNet
		vector<tuple<int, int, float>>().swap(GoodTransNet);
		GoodTransNet = tempTransNet;

		//清空tempTransNet
		vector<tuple<int, int, float>>().swap(tempTransNet);

		/*更新L_Delta与δ*/
		//L_Delta /= 2;
		N_Delta /= 2;
		delta = delta * factor;

		/*根据新的δ和GoodTransNet更新变换网络TransNet*/
		vector<tuple<int, int, float>>().swap(TransNet);//清空TransNet
		TransNet = GetNextNet(srcImage, GoodTransNet, subMaskPointSet, centerX, centerY, delta);

		cout << "the " << index << "th's GoodTransNet's size is :" << GoodTransNet.size() << endl;
		cout << "Next TransNet's size is: " << TransNet.size() << endl;
		cout << "		bestTransX is: " << get<0>(bestTrans) << endl;
		cout << "		bestTransY is: " << get<1>(bestTrans) << endl;
		cout << "		bestTransTheta is: " << get<2>(bestTrans) << "\n" << endl;


		/*清空vector*/
		vector<float>().swap(bestDistanceSet);
		vector<tuple<int, int, float>>().swap(GoodTransNet);

		if (TransNet.size() > 5000 || delta < resolution)
		{
			break;
		}
	}

	cout << "Step 2:Size of the next_N(δ) net is:" << TransNet.size() << endl;
	//cout << "Step 2:delta is :" << delta << "; bestDistance is :" << bestDistance << endl;
	cout << "Step 2:delta is :" << delta << "; bestNum is :" << bestNum << endl;
	cout << "Step 2:Iterate, update and calculate the best translation has been finished !" << endl;

	return bestTrans;
}
