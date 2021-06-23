#include <iostream>
#include <cuda_runtime.h> 

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudafilters.hpp> 

//comment this definition for using pinned memory instead of unified memory
#define USE_UNIFIED_MEM   

int main()
{
     //std::cout << cv::getBuildInformation() << std::endl; 

const char* gst = "nvarguscamerasrc  ! video/x-raw(memory:NVMM), format=(string)NV12, width=(int)640, height=(int)480, framerate=(fraction)30/1 ! \
			nvvidconv    ! video/x-raw, format=(string)BGRx, framerate=(fraction)30/1 ! \
  			videoconvert ! queue ! video/x-raw, format=(string)BGR, framerate=(fraction)30/1 ! \
			appsink"; 

    cv::VideoCapture cap(gst, cv::CAP_GSTREAMER);
    if(!cap.isOpened()) {
	std::cout<<"Failed to open camera."<<std::endl;
	return (-1);
    }
    
    unsigned int width  = cap.get(cv::CAP_PROP_FRAME_WIDTH); 
    unsigned int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT); 
    unsigned int fps    = cap.get(cv::CAP_PROP_FPS);
    unsigned int pixels = width*height;
    std::cout <<"Frame size : "<<width<<" x "<<height<<", "<<pixels<<" Pixels "<<fps<<" FPS"<<std::endl;

cv::namedWindow("frame_out", cv::WINDOW_AUTOSIZE );
    bool hasOpenGlSupport = true;
    try {
        cv::namedWindow("d_frame_out", cv::WINDOW_AUTOSIZE | cv::WINDOW_OPENGL);
    }
    catch(cv::Exception& e) {
	hasOpenGlSupport = false;
    }

    unsigned int frameByteSize = pixels * 3; 

#ifndef USE_UNIFIED_MEM
    /* Pinned memory. No cache */
    std::cout << "Using pinned memory" << std::endl;
    void *device_ptr, *host_ptr;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc((void **)&host_ptr, frameByteSize, cudaHostAllocMapped);
    cudaHostGetDevicePointer((void **)&device_ptr, (void *) host_ptr , 0);
    cv::Mat frame_out(height, width, CV_8UC3, host_ptr);
    cv::cuda::GpuMat d_frame_out(height, width, CV_8UC3, device_ptr);
#else
    /* Unified memory */
    std::cout << "Using unified memory" << std::endl;
    void *unified_ptr;
    cudaMallocManaged(&unified_ptr, frameByteSize);
    cv::Mat frame_out(height, width, CV_8UC3, unified_ptr);
    cv::cuda::GpuMat d_frame_out(height, width, CV_8UC3, unified_ptr);
#endif

    cv::Ptr< cv::cuda::Filter > filter = cv::cuda::createSobelFilter(CV_8UC3, CV_8UC3, 1, 1, 1, 1, cv::BORDER_DEFAULT);
    cv::Mat frame_in;

    while(1)
    {
    	if (!cap.read(frame_in)) {
		std::cout<<"Capture read error"<<std::endl;
		break;
	}
	else  {
	        frame_in.copyTo(frame_out);
	        // no need to copy to device
		filter->apply(d_frame_out, d_frame_out);
		if (hasOpenGlSupport)
			cv::imshow("d_frame_out", d_frame_out);
	        // no need to copy back to host
		cv::imshow("frame_out", frame_out); 
		cv::waitKey(1); 
	}	
    }

    cap.release();

    return 0;
}
