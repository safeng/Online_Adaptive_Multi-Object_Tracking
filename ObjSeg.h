#pragma once
#include "MotionFeature.h"
#include <list>

class ObjSeg
{
private:
	MotionFeature *motionFeature;// Motion feature track set as our ground database to maintain features
	void fillMap(cv::Mat &ROI, const KLT_FeatureList fl, int gap);
	void setMappingArray();
	void moveFeature(KLT_FeatureList dst,KLT_FeatureList src,int dsti,int srci);
	KLT_FeatureList AppendNewFeaturesToList(KLT_FeatureList inputList,KLT_FeatureList appendedList,int startIndex,int length);
	KLT_FeatureList AppendNewFeaturesToList(KLT_FeatureList inputList,KLT_FeatureList appendedList,int startIndex,int length,const std::list<int> &missList);
public:
	int frameNO;
	int *mappingArray;//mapping each feature track to its actual storage space
	KLT_TrackingContext tc; //klt tracking class handler
	inline void SetFrameNO(int frameNO){this->frameNO = frameNO;}// called at every frame
	inline MotionFeature *GetMotionFeature() const {return motionFeature;}//Const function
	inline MotionFeature *GetMotionFeature() {return motionFeature;}//Const function
	ObjSeg();//The motion feature set should live throughout the main program
	~ObjSeg(void);
	void ResetOccupationMap();
	void SegmentObj(int frameNO);//Segmentation at frameNO
	void Delete_Append_FeatureTrackSet(KLT_FeatureList list, std::list<int> &missedList);
	void Add_FeatureTrackSet(KLT_FeatureList list,int orignalLength, const std::list<int> &missedList);
	void CalAvgVecBwtFrames(int frame_gap);
	KLT_FeatureList ReplaceFeatureList(const cv::Mat &img, KLT_FeatureList inputlist,const std::list<int> &missList, int mindist, bool type);
	void WriteTrackToFile(char *fname); // Write the feature trajectory set to file
	void ResetMappingArray(); //Separately free the sparse graph and the point set
	void WriteFeatureListToRGBImg(const cv::Mat &img, const KLT_FeatureList fl, const char *fileName);//Write the feature list to an rgb image
};