#include "KLT.h"
using namespace bgm;

const double KLT::qualityLevel(0.04);
const double KLT::minDistance(10.); //min dist gap to select features
const bool KLT::useHarrisDetector(false);
const double KLT::k(0.04);
const cv::Size KLT::winSize(21,21);
const int KLT::maxLevel(3);
const double KLT::minEigThreshold(5e-4);

KLT::KLT(void):enable_bgm(false),bgm(NULL),maxCorners(-1)
{
}


KLT::~KLT(void)
{
	if(bgm!=NULL)
		delete bgm;
}

/*Use the background model--init step*/
void KLT::UseBackgroundModel(const cv::Mat &first_frame)
{
	assert(!this->enable_bgm && !this->bgm);

	enable_bgm = true;//enable bgm
	bgm = new BackgroundGMM(first_frame);
}

/*select good features from ROI and store to fl*/
void KLT::SelectGoodFeaturesToTrack(const cv::Mat &frame, KLT_FeatureList fl, const cv::Mat &ROI)
{
	assert(ROI.total() == 0 || (ROI.cols == frame.cols && ROI.rows == frame.rows));
	
	this->maxCorners = fl->nFeatures; // full fill size of list
	std::vector<cv::Point2f> features; //Feature point location

	//Determine the mask
	cv::Mat mask;
	
	if(enable_bgm)
	{
		mask.create(frame.rows,frame.cols,cv::DataType<uchar>::type);
		for(int i = 0 ; i< frame.rows; ++i)
		{
			for(int j = 0; j < frame.cols; ++j)
			{
				mask.at<uchar>(i,j) = uchar(ROI.at<uchar>(i,j) & bgm->DetermineForeground(i,j));
			}
		}
	}else
	{
		mask = ROI;
	}
	cv::Mat fr_gray;
	cv::cvtColor(frame,fr_gray,CV_RGB2GRAY);//convert to gray image
	cv::goodFeaturesToTrack(fr_gray,features,maxCorners,qualityLevel,minDistance,mask,3,useHarrisDetector,k);//has to be single channel image
	//find sub-pixel accurancy
	if(!features.empty())
		cv::cornerSubPix(fr_gray,features,cv::Size(10,10),cv::Size(-1,-1),cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01));

	//store to fl
	assert(features.size()<=fl->nFeatures);
	for(int i = 0; i<fl->nFeatures; ++i)
	{
		if(i<features.size())
		{
			fl->feature[i]->pos_x = features[i].x;
			fl->feature[i]->pos_y = features[i].y;
			fl->feature[i]->val = 1;
		}else //fillin invalid values
		{
			fl->feature[i]->pos_x = -1;
			fl->feature[i]->pos_y = -1;
			fl->feature[i]->val = -1;
		}
	}
}

/*Track features stored in fl*/
void KLT::TrackFeatures(std::vector<cv::Mat> &pre_pyr, std::vector<cv::Mat> &nxt_pyr, KLT_FeatureList fl)
{
	assert(!nxt_pyr.empty());

	//check if previous frame & pyr have been stored
	if(this->pre_pyr.empty())
	{
		assert(!pre_pyr.empty());
		this->pre_pyr.assign(pre_pyr.cbegin(),pre_pyr.cend());
	}

	//Generate feature point vector
	std::vector<cv::Point2f> Pts[2];
	Pts[0].reserve(fl->nFeatures);
	std::vector<int> index_fl;
	index_fl.reserve(fl->nFeatures);

	for(int i = 0; i<fl->nFeatures; ++i)
	{
		if(fl->feature[i]->val>=0)
		{
			Pts[0].push_back(cv::Point2f(fl->feature[i]->pos_x, fl->feature[i]->pos_y));
			index_fl.push_back(i);
		}
	}

	std::vector<uchar> status;
	std::vector<float> err;

	//Track features
	if(Pts[0].size()>0)
	{
		cv::calcOpticalFlowPyrLK(this->pre_pyr,nxt_pyr,Pts[0],Pts[1],status,err
			,KLT::winSize,KLT::maxLevel,
			cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), 0, KLT::minEigThreshold);
	}

	//Output results to feature list
	assert(Pts[1].size() == index_fl.size());
	for(size_t i = 0; i<Pts[1].size(); ++i)
	{
		int fl_inx = index_fl[i];

		if(status[i] == 1)
		{
			fl->feature[fl_inx]->val = 0;
			fl->feature[fl_inx]->pos_x = Pts[1][i].x;
			fl->feature[fl_inx]->pos_y = Pts[1][i].y;
		}else
		{
			fl->feature[fl_inx]->val = -1; //set to unsuccessfully tracked features
			fl->feature[fl_inx]->pos_x = -1;
			fl->feature[fl_inx]->pos_y = -1;
		}
	}

	//store for next tracking frame
	this->pre_pyr.assign(nxt_pyr.cbegin(), nxt_pyr.cend());
}

/*Process the next image. Returns the pyramid image for feature tracking*/
void KLT::ProcessNxtImg(const cv::Mat &nxt_frame, std::vector<cv::Mat> &img_pry)
{
	if(this->enable_bgm)
	{
		this->bgm->Processing(nxt_frame);
	}

	//retrieve pyramid image
	cv::Mat fr_gray;
	fr_gray.create(nxt_frame.rows,nxt_frame.cols,cv::DataType<uchar>::type);
	cv::cvtColor(nxt_frame,fr_gray,CV_RGB2GRAY);
	cv::buildOpticalFlowPyramid(fr_gray,img_pry,KLT::winSize,KLT::maxLevel);
}

/*Determine whether (i,j) is foreground*/
bool KLT::DetermineForeground(int i, int j)
{
	return this->bgm->DetermineForeground(i,j);
}

/*Write Feature List to image*/
void KLT_FeatureListRec::WriteToImage(const std::string imgName, const cv::Mat &img) const
{
	cv::Mat img_to_write;
	img.copyTo(img_to_write);
	const uchar color[3] = {255,0,0};
	int dot_size = 1;
	for(int i = 0; i<this->nFeatures ; ++i)
	{
		int x = int(feature[i]->pos_x+0.5f);
		int y = int(feature[i]->pos_y+0.5f);
		for(int xx = x-dot_size; xx <= x+dot_size; ++xx)
		{
			if(xx>=0 && xx<img.cols)
			{
				for(int yy = y-dot_size; yy <=y+dot_size; ++yy)
				{
					if(yy>=0 && yy<img.rows)
					{
						cv::Vec3b &val = img_to_write.at<cv::Vec3b>(yy,xx);
						val.val[0] = color[2];
						val.val[1] = color[1];
						val.val[2] = color[0];
					}
				}
			}
		}
	}

	cv::imwrite(imgName,img_to_write);
}

KLT_FeatureList KLTCreateFeatureList(
  int nFeatures)
{
  KLT_FeatureList fl;
  KLT_Feature first;
  int nbytes = sizeof(KLT_FeatureListRec) +
    nFeatures * sizeof(KLT_Feature) +
    nFeatures * sizeof(KLT_FeatureRec);
  int i;
	
  /* Allocate memory for feature list */
  fl = (KLT_FeatureList)malloc(nbytes);
	
  /* Set parameters */
  fl->nFeatures = nFeatures; 

  /* Set pointers */
  fl->feature = (KLT_Feature *) (fl + 1);
  first = (KLT_Feature) (fl->feature + nFeatures);
  for (i = 0 ; i < nFeatures ; i++) {
    fl->feature[i] = first + i;
  }
  /* Return feature list */
  return(fl);
}

void KLTFreeFeatureList(
  KLT_FeatureList fl)
{ 
  free(fl);
}

/*Count the # of valid features in feature list*/
int KLTCountRemainingFeatures(const KLT_FeatureList fl)
{
	int count = 0;
	for(int i = 0; i<fl->nFeatures; ++i)
	{
		if(fl->feature[i]->val>=0)
			++count;
	}

	return count;
}

void FillMap(cv::Mat &ROI, const KLT_FeatureList fl, int gap)
{
	for(int i = 0 ;i<fl->nFeatures; ++i)
	{
		if(fl->feature[i]->val>=0)
		{
			int x = int(fl->feature[i]->pos_x);
			int y = int(fl->feature[i]->pos_y);
			for(int xx = x-gap;xx<=x+gap;++xx)
			{
				if(xx>=0 && xx<ROI.cols)
				{
					for(int yy = y-gap; yy<=y+gap; ++yy)
					{
						if(yy>=0 && yy<ROI.rows)
						{
							ROI.at<uchar>(yy,xx) = 0;
						}
					}
				}
			}
		}
	}
}

void ShrinkFeatureList(KLT_FeatureList &featurelist)
{
	int originalLength=featurelist->nFeatures;
	int shrinkablenum=0;
	//Count from back until meets a valid feature
	for(int i=originalLength-1;featurelist->feature[i]->val<0&&i>=0;i--,shrinkablenum++);//count the shrinkable number
	if(shrinkablenum==0)
		return;
	//fprintf(stderr,"%d shrinkable units!\n",shrinkablenum);
	KLT_FeatureList newlist = KLTCreateFeatureList(originalLength-shrinkablenum);
	memcpy(newlist->feature[0],featurelist->feature[0],newlist->nFeatures*sizeof(KLT_FeatureRec));//copy the feature data
	KLTFreeFeatureList(featurelist);
	featurelist=newlist;
}