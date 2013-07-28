#include "BackgroundGMM.h"
using namespace cv;
using namespace std;
using namespace bgm;

static int ThreeDMapping(int width, int w, int height, int h, int C, int c_i);

static bool CompareFunc(const Rank &r1, const Rank &r2);

BackgroundGMM::BackgroundGMM(const Mat &first_frame, bool repeat, int n):
C(2),M(2),D(2.5f),alpha(0.0001f),thresh(0.4f),sd_init(6),width(first_frame.cols),height(first_frame.rows)
{
	first_frame.copyTo(fr);
	cvtColor(fr,fr_bw,CV_RGB2GRAY);
	w = new float[width*height*C];
	mean = new float[width*height*C];
	sd = new float[width*height*C];
	u_diff = new float[width*height*C];
	p = alpha/(1/(float)C);
	//set random mean value
	for(int i = 0; i<width*height*C; ++i)
	{
		mean[i] = (float)(rand()%256);
	}
	//init values
	std::fill(w,w+width*height*C,1/(float)C);
	std::fill(sd,sd+width*height*C,sd_init);
	std::fill(u_diff,u_diff+width*height*C,0.0f);

	if(repeat)//repeat for the first frame to generate a robust model
	{
		PreInit(first_frame, n);
	}
}


BackgroundGMM::~BackgroundGMM(void)
{
	delete[] w;
	delete[] mean;
	delete[] sd;
	delete[] u_diff;
}

/*Processing frame by frame*/
void BackgroundGMM::Processing(const cv::Mat &frame)
{
	assert(frame.cols == fr.cols && frame.rows == fr.rows);
	frame.copyTo(fr);
	cvtColor(fr,fr_bw,CV_RGB2GRAY);//convert to gray scale
	//calculate difference of pixel values from mean (difference from background mean)
	for(int i = 0; i<height; ++i)
	{
		for(int j = 0; j<width; ++j)
		{
			for(int k = 0; k<C; ++k)
			{
				int index = ThreeDMapping(width,j,height,i,C,k);
				u_diff[index] = fabsf((float)fr_bw.at<uchar>(i,j) - mean[index]);
			}
		}
	}

	//before updating, preprocessing data structures
	this->fg.release();
	this->fg.create(height,width,cv::DataType<uchar>::type);//allocate space
	this->fg_mask.release();
	this->fg_mask.create(height,width,cv::DataType<bool>::type);
	this->bg_bw.release();
	bg_bw.create(height,width,cv::DataType<float>::type);

	//update gaussian components
	for(int i = 0; i<height; ++i)//For each pixel
	{
		for(int j = 0; j <width; ++j)
		{
			ProcessingOnePix(i,j);
		}
	}
}

/*Repeat several times for the same image*/
void BackgroundGMM::PreInit(const cv::Mat &frame, int n)
{
	for(int i = 0; i<n; ++i)
	{
		Processing(frame);
	}
}

/*Processing for individual pixels*/
void BackgroundGMM::ProcessingOnePix(int i, int j)
{
	bool match = false;//whether match one of background component
	float sum_c = 0;
	int index = ThreeDMapping(width,j,height,i,C,0);
	uchar pix_val = fr_bw.at<uchar>(i,j);

	for(int k = 0; k<C; ++k)
	{
		int index_k = index+k;
		if(fabsf(u_diff[index_k])<=D*sd[index_k])//match one component
		{
			match = true;
			//update weights, mean, sd, p
			w[index_k] = (1-alpha)*w[index_k] + alpha;
			p = alpha/w[index_k];
			mean[index_k] = (1-p)*mean[index_k] + p*pix_val;
			sd[index_k] =  sqrtf((1-p)*(sd[index_k]*sd[index_k]) + p*((pix_val - mean[index_k])*(pix_val - mean[index_k])));

			sum_c+=w[index_k];
		}else//don't match
		{
			w[index_k] = (1-alpha)*w[index_k];
			sum_c+=w[index_k];
		}
	}

	int min_index = -1;
	float min_weight = FLT_MAX;
	for(int k = 0; k<C; ++k)
	{
		int index_k = index+k;
		//normalize for each pixel
		w[index_k]/=sum_c;
		//no match, find the min_index and min_weight
		if(!match)
		{
			if(w[index_k]<min_weight)
			{
				min_index = index_k;
				min_weight = w[index_k];
			}
		}
	}

	bg_bw.at<float>(i,j) = 0; //note the format should be CV_32F
	for(int k = 0; k<C; ++k)
	{
		int index_k = index+k;
		bg_bw.at<float>(i,j) = bg_bw.at<float>(i,j) + mean[index_k]*w[index_k];
	}

	//if no components match, create new component
	if(!match)
	{
		assert(min_index!=-1);
		mean[min_index] = pix_val;//create with new mean and sd
		sd[min_index] = sd_init;
	}

	Rank *rank_array = new Rank[C];
	for(int k=0; k<C; ++k)
	{
		int index_k = index+k;
		rank_array[k].val = w[index_k]/sd[index_k];
		rank_array[k].index = k;
	}
	//sort in descending order
	std::sort(rank_array,rank_array+C,CompareFunc);

	match = false;
	fg.at<uchar>(i,j) = 0;//note this foreground image should be uchar
	int k = 0;
	
	while(!match && k<M)//only check first M components
	{
		int index_k = index + rank_array[k].index;
		if(w[index_k]>=thresh)//valid background component
		{
			if(fabs(u_diff[index_k])<=D*sd[index_k])
			{
				//accepted as background
				fg.at<uchar>(i,j) = 0;
				match = true;//once match -> background
			}else
			{
				fg.at<uchar>(i,j) = pix_val;
			}
		}
		++k;
	}

	if(!match)//foreground pixel
	{
		fg_mask.at<bool>(i,j) = true;
	}else
	{
		fg_mask.at<bool>(i,j) = false;
	}

	delete[] rank_array;
}

/*Returns true if it is considered as foreground pixel*/
bool BackgroundGMM::DetermineForeground(int i, int j)
{
	return fg_mask.at<bool>(i,j);
}

/*Store foreground and background image*/
void BackgroundGMM::StoreImg(int frameNO)
{
	//foreground image
	ostringstream oss;
	oss<<"res/foreground_"<<frameNO<<".jpg";
	imwrite(oss.str(),this->fg);
	//background image
	Mat background;
	//convert to CV_8U
	this->bg_bw.convertTo(background,cv::DataType<uchar>::type);
	oss.str("");
	oss.clear();
	oss<<"res/background_"<<frameNO<<".jpg";
	imwrite(oss.str(),background);
}

/*Get background image*/
void BackgroundGMM::GetBackgroundImage(cv::Mat &bg_img)
{
	this->bg_bw.convertTo(bg_img,cv::DataType<uchar>::type);
}

/*Get foreground image*/
void BackgroundGMM::GetForegroundImage(cv::Mat &fg_img)
{
	this->fg.copyTo(fg_img);
}

static int ThreeDMapping(int width, int w, int height, int h, int C, int c_i)
{
	return (h*width+w)*C+c_i;
}

static bool CompareFunc(const Rank &r1, const Rank &r2)
{
	return r1.val>r2.val;//sort in descending order
}