#pragma once
//a wrapper class using opencv
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

template <class T> class OpenCVImage
{
private:
  IplImage* imgp;
  public:
  OpenCVImage(IplImage* img=0) {imgp=img;}
  ~OpenCVImage(){
	  if (imgp!=0)
		  cvReleaseImage(&imgp);//release the image
	  imgp=0;
  }
  void operator=(IplImage* img) {imgp=img;}
  inline T* operator[](const int rowIndx) {
    return ((T *)(imgp->imageData + rowIndx*imgp->widthStep));}
  inline IplImage* getIplImage() const{return imgp;}//const member function
};

typedef struct{
  unsigned char b,g,r;
} RgbPixel; 
 
typedef struct{
  float b,g,r;
} RgbPixelFloat; 
 
typedef OpenCVImage<RgbPixel>       RgbImage;
typedef OpenCVImage<RgbPixelFloat>  RgbImageFloat;
typedef OpenCVImage<unsigned char>  BwImage;
typedef OpenCVImage<float>          BwImageFloat;

void ReadJPGImage(const char *fname,RgbImage *&rgbImage,int paddedX,int paddedY,unsigned char *&img,int &ncols,int &nrows);
void FreeImgObj(RgbImage *&rgbImage);
void ReadColorImage(const char *fname,RgbImage *&rgbImage,int paddedX,int paddedY);
void ConvertToUnChar(RgbImage *rgbImage,unsigned char *img);//Convert a image in rgb format to 1-D array unsigned char
void WriteColorImage(const char *fname,const RgbImage *rgbImage);
RgbImage *GenerateImagePatches(const RgbImage *rgbImage,int x,int y,int window_height,int window_width);


void ComputeHistogramPatch(const RgbImage* patch, int n_bins,float histVector[],float mean[],float variance[]);
float CompHistAffinity(const float histi[],const float histj[],int length);

void DrawLine(IplImage *image, int neighLen, int x0, int x1, int y0, int y1, const uchar rgb[3]);
void DrawRectangle(cv::Mat &image, const cv::Rect &rect, const uchar rgb[3]);
float GetAvgIntensityValue(const IplImage *image, size_t center_x, size_t center_y, size_t width, size_t height);
bool CheckBackGroundPixel(const float *mean_app, const IplImage *cur_image, size_t center_x, size_t center_y, size_t width, size_t height, float threshold);//check whether it is a background pixel
float GetOverLapPercentage(const cv::Rect &rect1, const cv::Rect &rect2);
cv::Point GetRectCenter(const cv::Rect &rect);
cv::Rect AvgRectPair(const cv::Rect &r1, const cv::Rect &r2, float weight1, float weight2);