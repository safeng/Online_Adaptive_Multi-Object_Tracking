#pragma once

#include "BackgroundGMM.h"
#include "OpenCVImage.h"

//KLT_Feature
typedef struct  {
  float pos_x;
  float pos_y; //position of feature point with subpixel accuracy
  int val;
  float velocity[2]; //instant vec per frame
}  KLT_FeatureRec, *KLT_Feature;

//Feature List
typedef struct  {
  int nFeatures;
  KLT_Feature *feature;
  void WriteToImage(const std::string imgName, const cv::Mat &img) const;
}  KLT_FeatureListRec, *KLT_FeatureList;

//KLT optical flow with OpenCV
class KLT
{
private:
	//cv::GoodFeaturesToTrackDetector feature_detector; //Feature detector to select good features
	bgm::BackgroundGMM *bgm;

	//params for background model
	bool enable_bgm; //whether we use the background model

	//params for feature detector
	int maxCorners; //change with len of feature list 
	static const double qualityLevel;
	static const double minDistance;
	static const bool useHarrisDetector;
	static const double k;

	//params for KL optical flow tracking
	static const cv::Size winSize;
	static const int maxLevel;
	static const double minEigThreshold;

	//Stored frame pyramid for further tracking
	std::vector<cv::Mat> pre_pyr;
public:
	bool DetermineForeground(int i, int j); //ith row, jth col is foreground
	bool CheckBgEnabled() const { return this->enable_bgm; } //check whether background model is enabled 
	void ProcessNxtImg(const cv::Mat &nxt_frame, std::vector<cv::Mat> &img_pry); //Process the next img
	void GetBackgroundImage(cv::Mat &bg_img); //Get bg image
	void UseBackgroundModel(const cv::Mat &first_frame); //Use & init background model
	void SelectGoodFeaturesToTrack(const cv::Mat &frame, KLT_FeatureList fl, const cv::Mat &ROI); //Select good features and store them to fl, only with in ROI 
	void TrackFeatures(std::vector<cv::Mat> &pre_pyr, std::vector<cv::Mat> &nxt_pyr, KLT_FeatureList fl); //Tracking features in 
	KLT(void);
	~KLT(void);
};

int KLTCountRemainingFeatures(
	const KLT_FeatureList fl);

KLT_FeatureList KLTCreateFeatureList(
  int nFeatures);

void KLTFreeFeatureList(
  KLT_FeatureList fl);

void FillMap(cv::Mat &ROI, const KLT_FeatureList fl, int gap);

void ShrinkFeatureList(
	KLT_FeatureList &featurelist);