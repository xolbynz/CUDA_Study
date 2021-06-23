#include <stdio.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>

#define NUM_REPEAT 1000

int main()
{
	cv::Mat imgSrc = cv::imread("000014.jpg");
	//cv::imshow("imgSrc", imgSrc);

	{
		cv::Mat imgDst;
		const auto& t0 = std::chrono::steady_clock::now();
		for (int i = 0; i < NUM_REPEAT; i++) cv::resize(imgSrc, imgDst, cv::Size(1920, 1080), 2);
		const auto& t1 = std::chrono::steady_clock::now();
		std::chrono::duration<double> timeSpan = t1 - t0;
		printf("CPU = %.3lf [msec]\n", timeSpan.count() * 1000.0 / NUM_REPEAT);
		// std::cout<<"imgDst"<<imgDst<<std::endl;
		//cv::imshow("CPU", imgDst);
	}

	{
		cv::cuda::GpuMat imgGpuSrc, imgGpuDst;
		cv::Mat imgDst;
		imgGpuSrc.upload(imgSrc);

		const auto& t0 = std::chrono::steady_clock::now();
		for (int i = 0; i < NUM_REPEAT; i++) {
			cv::cuda::resize(imgGpuSrc, imgGpuDst, cv::Size(1920, 1080), 2);
		}
		const auto& t1 = std::chrono::steady_clock::now();
		std::chrono::duration<double> timeSpan = t1 - t0;
		printf("GPU = %.3lf [msec]\n", timeSpan.count() * 1000.0 / NUM_REPEAT);

		for(int i = 0; i < imgGpuDst.rows; i++){
			for(int j = 0; j < imgGpuDst.cols; j++){
				std::cout<<"imgGpuDst"<<imgGpuDst[j][i]<<std::endl;
				
			}
		}

		imgGpuDst.download(imgDst);
		//cv::imshow("GPU", imgDst);
	}

	cv::waitKey(0);
	return 0;
}

