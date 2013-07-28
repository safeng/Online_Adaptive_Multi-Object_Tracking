#pragma once
#include "Detector.h"
#include "StrongClassifier.h"
#include "StrongClassifierDirectSelection.h"
#include "StrongClassifierStandardSemi.h"

class SemiBoostingTracker
{
private:
	Rect validROI; // validROI for frame

	StrongClassifierStandardSemi *classifier; //supervised & semi-supervised trained classifier
	StrongClassifier *classifierOff;
	Detector *detector;
	enum{
		NUM_SELECTORS = 50,
		NUM_WEAK_CLASSIFIERS = 50,
		ITERATION_INIT = 50, //time of iterations when perform one-shot training in first frame
	};
	int initTracking(ImageRepresentation *image, Rect trackedPatch, Size validROI, Patches *trackingPatches);
	int updateOn(ImageRepresentation *image, Rect detectionPatch);
	void reInit(ImageRepresentation *image, Rect trackedPatch, Size validROI, Patches *patches);
	bool checkValidTrackPatch(Rect trackedPatch);
	void DrawConfidenceMap(const cv::Mat &ROI, const Rect searchRegion, Patches *patches);
public:
	cv::Rect searchRegion;
	bool tracking_lost; //whether appearance tracking is lost
	bool available; // Whether semi boosting classifier is valid
	
	void SetSearchRegion(const cv::Rect &search_org, const cv::Rect &trackedRegion);
	bool TrackSearchRegion(ImageRepresentation *image, const cv::Rect &trackedPatch, float minMargin, cv::Rect &result, std::vector<cv::Rect> &posRect, std::vector<float> &posWeight);
	static uchar* getGrayImage(const cv::Mat &frame);
	static void SetImageRepresentation(ImageRepresentation *image, const cv::Mat &frame);
	Rect getTrackingROI(Rect trackedPatch, float searchFactor);
	void SupervisedNegative(ImageRepresentation *image, const cv::Rect &region);
	int SupervisedTraining(ImageRepresentation *image, const cv::Rect &region, Size validROI, bool one_shot); // supervised training. Indicate whether it is one-shot training at starting stage
	void SemiSupervisedTraining(ImageRepresentation *image, const cv::Rect &region);
	float Evaluate(ImageRepresentation *image, const cv::Rect &region, bool use_negative); //evaluate
	void Clear(); //clear all the setting
	SemiBoostingTracker(void);
	~SemiBoostingTracker(void);
};

