#include "OpenCVImage.h"
using namespace cv;

/*Integration of image reading process*/
void ReadJPGImage(const char *fname,RgbImage *&rgbImage,int paddedX,int paddedY,unsigned char *&img,int &ncols,int &nrows)
{
	//We use the information computed in gaussian border for padding the image border
	ReadColorImage(fname,rgbImage,paddedX,paddedY);
	IplImage *image=rgbImage->getIplImage();
	ncols=image->width;
	nrows=image->height;
	if(img==NULL)//have not allocated memory space for img
	{
		img=(unsigned char*)malloc(sizeof(unsigned char)*ncols*nrows);
	}
	ConvertToUnChar(rgbImage,img);
}

/*Read a color image from a file and may allocate space for object*/
void ReadColorImage(const char *fname,RgbImage *&rgbImage,int paddedX,int paddedY)
{
	IplImage *image,*image_b;
	image=cvLoadImage(fname);
	if(image==NULL)
	{
		fprintf(stderr,"Cannot load image %s\n",fname);
		assert(image!=NULL);
	}
	image_b=cvCreateImage(cvSize(image->width+paddedX*2,image->height+paddedY*2),image->depth,image->nChannels);//the padded image
	CvPoint offset=cvPoint(paddedX,paddedY);
	cvCopyMakeBorder(image,image_b, offset, IPL_BORDER_REPLICATE, cvScalarAll(0));
	if(rgbImage==NULL)//We should allocate space for it
		rgbImage=new RgbImage(image_b);//Note how we allocate space (pass by value)
	else
		*rgbImage=image_b;
	cvReleaseImage(&image);//Release the original image
}

/*Write the modified image to the file with the specific file name*/
void WriteColorImage(const char *fname,const RgbImage *rgbImage)
{
	assert(rgbImage!=NULL&&rgbImage->getIplImage()!=NULL);
	IplImage *image=rgbImage->getIplImage();
	if(!cvSaveImage(fname,image))
		fprintf(stderr,"Error in saving the image %s\n",fname);
}

/*Free the rgbImage object and the image it has loaded*/
void FreeImgObj(RgbImage *&rgbImage)
{
	if(rgbImage!=NULL)
		delete rgbImage;
	rgbImage=NULL;
}

/*Convert a rgbImage into 1D unsigned char array (into gray image)*/
void ConvertToUnChar(RgbImage *rgbImage,unsigned char *img)
{
	assert(rgbImage!=NULL&&img!=NULL);
	IplImage *iplImg=rgbImage->getIplImage();
	IplImage *greyImg=cvCreateImage(cvSize(iplImg->width,iplImg->height),iplImg->depth,1);
	cvCvtColor(iplImg,greyImg,CV_BGR2GRAY);//Convert to greyscale image
	int height     = greyImg->height;
	int width      = greyImg->width;
	int step       = greyImg->widthStep/sizeof(uchar);
	uchar* data    = (uchar *)greyImg->imageData;
	for(int i=0;i<height;++i)
	{
		for(int j=0;j<width;++j)
			img[j+i*width]=data[i*step+j];//store the values to img
	}
	cvReleaseImage(&greyImg);
}

/*Draw line with color rgb*/
void DrawLine(IplImage *image, int neighLen, int x0, int x1, int y0, int y1, const uchar rgb[3])
{
	CvScalar color=CV_RGB(rgb[0],rgb[1],rgb[2]);
	CvPoint pt1=cvPoint(x0,y0);
	CvPoint pt2=cvPoint(x1,y1);
	cvLine(image,pt1,pt2,color,neighLen);
}

/*Draw a rectangle*/
void DrawRectangle(cv::Mat &image, const cv::Rect &r, const uchar rgb[3])
{
	rectangle(image, r.tl(), r.br(), cv::Scalar(rgb[2],rgb[1],rgb[0]), 4);
}

/*Generate image path centered at <x,y> with window_height * window_width*/
RgbImage *GenerateImagePatches(const RgbImage *rgbImage,int x,int y,int window_height,int window_width)
{
	if(rgbImage!=NULL&&rgbImage->getIplImage()!=NULL)
	{
		IplImage *patch=cvCreateImage(cvSize(window_width,window_height),IPL_DEPTH_8U,3);
		CvPoint2D32f center=cvPoint2D32f(x,y);//col,row
		cvGetRectSubPix(rgbImage->getIplImage(),patch,center);
		return new RgbImage(patch);
	}
	return NULL;
}

/*Return the histogram vetor with n_bins for an image patch centered at <x,y>,
with window_width * window_height
Also we compute the mean & variance for each color plane*/
void ComputeHistogramPatch(const RgbImage* Patch, int n_bins,float histVector[],float mean[],float variance[])//The caller must make sure that the space for spaceVector has been allocated
{ 
	assert(Patch!=NULL);
	IplImage *patch=Patch->getIplImage();
	assert(patch!=NULL&&histVector!=NULL);
	IplImage *plane[3];//for r,g,b color plane respectively
	int totalPix=patch->width*patch->height;
	for(int k=0;k<3;k++)
	{
		plane[k]=cvCreateImage(cvSize(patch->width,patch->height),patch->depth,1);
	}
	float range[]={0,255};
	float *rarray[]={range};
	CvHistogram *hist[3];
	cvCvtPixToPlane(patch,plane[0],plane[1],plane[2],NULL);
	//int *histVector=(int*)malloc(n_bins*3*sizeof(int));//3 color planes combined into one vector
	int step=plane[0]->widthStep/sizeof(uchar);
	for(int k=0;k<3;k++)
	{
		float sum=0,sum2=0;
		hist[k]=cvCreateHist(1,&n_bins,CV_HIST_ARRAY,rarray,1);
		cvCalcHist(plane+k,hist[k]);
		uchar *data=(uchar*)plane[k]->imageData;
		for(int i=0;i<plane[k]->height;i++)
		{
			for(int j=0;j<plane[k]->width;j++)
			{
				sum+=data[i*step+j];
			}
		}
		mean[k]=sum/totalPix;//calculate the mean value

		for(int i=0;i<plane[k]->height;i++)
		{
			for(int j=0;j<plane[k]->width;j++)
			{
				sum2+=pow(data[i*step+j]-mean[k],2);
			}
		}
		variance[k]=sum2/totalPix;//Calculate the variance value
	}
	/*extract the value*/
	for(int k=0;k<3;k++)
	{
		for(int i=0;i<n_bins;i++)
		{
			float number=cvQueryHistValue_1D(hist[k],i);
			histVector[i+n_bins*k]=number;
			//printf("number=%g\n",number);
		}
		cvReleaseImage(plane+k);//Release the resources
		cvReleaseHist(hist+k);
	}
}

/*Compute the Chi-square distance between two histogrames
length = 3* number of bins*/
float CompHistAffinity(const float histi[],const float histj[],int length)
{
	assert(histi!=NULL&&histj!=NULL);
	float sum=0;
	for(int i=0;i<length;i++)
	{
		float avg=(histi[i]+histj[i])/2;
		assert(histi[i]>=0&&histj[i]>=0);
		float delta=pow((histi[i]-avg),2)/(avg+1);
		sum+=delta;
	}
	return sum;

	/*float sum=0;
	for(int i=0;i<length;i++)
	{
		float up=fabs(histi[i]-histj[i]);
		float down=histi[i]+histj[i]+1;
		sum+=up/down;
	}
	return sum;*/
}

/*Calculate the average intensity value (RGB/3) at (center_y,center_x)*/
float GetAvgIntensityValue(const IplImage *image, size_t center_x, size_t center_y, size_t width, size_t height)
{
	assert(image!=NULL);
	Mat img(image);
	size_t width_h=width/2;//half size
	size_t height_h=height/2;
	float sum(0);
	for(size_t i=center_x-width_h;i<=center_x+width_h;++i)
	{
		for(size_t j=center_y-height_h;j<=center_y+width_h;++j)
		{
			if(i>=0&&i<img.cols&&j>=0&&j<img.rows)
			{
				Vec3b value=img.at<Vec3b>(j,i);//value at jth row, ith col
				uchar blue=value.val[0];
				uchar green=value.val[1];
				uchar red=value.val[2];
				float intensity=(blue+green+red)/3.0f;
				sum+=intensity;
			}
		}
	}
	float avg_value = sum/(width*height);
	return avg_value;
}

/*Check whether the pixel at (ceneter_y,center_y) is considered as background pixel*/
bool CheckBackGroundPixel(const float *mean_app, const IplImage *cur_image,
	size_t center_x, size_t center_y, size_t width, size_t height, float threshold)
{
	float appValue = GetAvgIntensityValue(cur_image,  center_x, center_y, width, height);
	int img_width = cur_image->width;
	int img_height = cur_image->height;
	int index = center_x + center_y*img_width;
	float mean_app_value = mean_app[index];
	if(fabs(appValue-mean_app_value)<threshold)//non-significant change
	{
		return true;//a background pixel
	}else//there is significant change on that position
		return false;//foreground pixel
}

/*Get overlapping percentage of two rectangles*/
float GetOverLapPercentage(const cv::Rect &rect1, const cv::Rect &rect2)
{
	Rect intersection = rect1 & rect2;
	int area_intersect = intersection.area();
	if(area_intersect==0)
		return 0;
	int area_union = rect1.area() + rect2.area() - area_intersect;
	float score = area_intersect/(float)(area_union);
	if (score<=0)
	{
		return 0;
	}
	assert(score>0);
	return score;
}

/*Get center of rectangle*/
Point GetRectCenter(const Rect &rect)
{
	return Point(rect.x+rect.width/2,rect.y+rect.height/2);
}

/*Average two rectangles*/
cv::Rect AvgRectPair(const cv::Rect &r1, const cv::Rect &r2, float weight1, float weight2)
{
	float center_1[] = {float(r1.x + r1.width/2.0f + 0.5f), float(r1.y + r2.height/2.0f + 0.5f)};
	float center_2[] = {float(r2.x + r2.width/2.0f + 0.5f), float(r2.y + r2.height/2.0f + 0.5f)};
	float avg_center[] = {center_1[0]*weight1 + center_2[0]*weight2, center_1[1]*weight1 + center_2[1]*weight2};
	float avg_scale[] = {r1.width*weight1 + r2.width*weight2, r1.height*weight1 + r2.height*weight2};
	cv::Rect final(int(avg_center[0] - avg_scale[0]/2 + 0.5f), int(avg_center[1] - avg_scale[1]/2 + 0.5f), int(avg_scale[0] + 0.5f), int(avg_scale[1] + 0.5f));
	return final;
}