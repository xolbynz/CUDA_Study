#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <vector>
#include <queue>
#include <limits>
#include <string>
 
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
 using namespace cv;
using namespace std;
 void overlayImage(Mat& src, const Mat& over, const Point& pos)
{
    CV_Assert(src.type() == CV_8UC3);
    CV_Assert(over.type() == CV_8UC4);
 
    int sx = std::max(pos.x, 0);
    int sy = std::max(pos.y, 0);
    int ex = std::min(pos.x + over.cols, src.cols);
    int ey = std::min(pos.y + over.rows, src.rows);
 
    for (int y = sy; y < ey; y++) {
        int y2 = y - pos.y; // y coordinate in overlay image
 
        Vec3b* pSrc = src.ptr<Vec3b>(y);
        const Vec4b* pOvr = over.ptr<Vec4b>(y2);
 
        for (int x = sx; x < ex; x++) {
            int x2 = x - pos.x; // x coordinate in overlay image
 
            float alpha = (float)pOvr[x2][3] / 255.f;
 
            if (alpha > 0.f) {
                pSrc[x][0] = saturate_cast<uchar>(pSrc[x][0] * (1.f - alpha) 
                        + pOvr[x2][0] * alpha);
                pSrc[x][1] = saturate_cast<uchar>(pSrc[x][1] * (1.f - alpha) 
                        + pOvr[x2][1] * alpha);
                pSrc[x][2] = saturate_cast<uchar>(pSrc[x][2] * (1.f - alpha) 
                        + pOvr[x2][2] * alpha);
            }
        }
    }
}
 
int main()
{
    cv::Mat imgDst;
    Mat frameMat01 = imread("heart.png",2);
    Mat frameMat02 = imread("normal.png",2);
    cv::resize(frameMat01, frameMat01, cv::Size(200, 200), 2);
    cv::resize(frameMat02, frameMat02, cv::Size(200, 200), 2);
    std::cout<<frameMat01.rows<<","<<frameMat01.cols<<std::endl;
    std::cout<<frameMat02.rows<<","<<frameMat02.cols<<std::endl;
    cv::cuda::GpuMat imgGpuSrc, imgGpuDst;
    imgGpuSrc.upload(frameMat01);
    imgGpuDst.upload(frameMat02);
    
    cv::cuda::addWeighted (imgGpuSrc,0.7,imgGpuDst,0.3,0.0,imgGpuSrc);
    //cv::resize(frameMat01,frameMat01,cv::Size(frameMat02.rows,frameMat02.cols));
    //overlayImage(frameMat01, frameMat02, cv::Point(20,50));
    imgGpuSrc.download(imgDst);
    imshow("src", imgDst);
 
    waitKey(0);


    return 0;

}
