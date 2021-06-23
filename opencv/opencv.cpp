#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <vector>
#include <queue>
#include <limits>
#include <string>
 
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui/highgui.hpp>
 
using namespace cv;
using namespace std;
 
int main(int argc, char** argv)
{
 
    while (1)
    {
 
        Mat frameMat01 = imread("output.jpg");
        Mat frameMat02 = imread("output2.jpg");
        
        clock_t begin, end;
        begin = clock();
 
        cuda::GpuMat cuFrameMat01, cuFrameMat02;
        cuFrameMat01.upload(frameMat01);
        cuFrameMat02.upload(frameMat02);
        
        cuda::cvtColor(cuFrameMat01, cuFrameMat01, COLOR_RGB2GRAY);
        cuda::cvtColor(cuFrameMat02, cuFrameMat02, COLOR_RGB2GRAY);
 
        Ptr<cuda::ORB> orb = cuda::ORB::create(1000, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20, false);
 
        cuda::GpuMat cuMask01(frameMat01.rows, frameMat01.cols, CV_8UC1, cv::Scalar::all(1)); //330,215
        cuda::GpuMat cuMask02(frameMat02.rows, frameMat02.cols, CV_8UC1, cv::Scalar::all(1)); //315,235
 
        cuda::GpuMat cuKeyPoints01, cuKeyPoints02;
        cuda::GpuMat cuDescriptors01, cuDescriptors02;
 
        orb->detectAndComputeAsync(cuFrameMat01, cuMask01, cuKeyPoints01, cuDescriptors01);
        orb->detectAndComputeAsync(cuFrameMat02, cuMask02, cuKeyPoints02, cuDescriptors02); 
 
        Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
        cuda::GpuMat gpuMatchesMat;
        matcher->knnMatchAsync(cuDescriptors01, cuDescriptors02, gpuMatchesMat, 2, noArray());
        vector<vector<DMatch>> knnMatchesVec;
        
 
        waitKey();
    }
 
 
    waitKey(0);
    return 0;
}
