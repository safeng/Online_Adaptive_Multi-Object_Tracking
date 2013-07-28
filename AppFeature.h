#pragma once
/*Define appearance feature space*/
#include "OnlineBoosting/SemiBoostingTracker.h"
#include "ObjSeg.h"
#include "SIRFilter.h"

/*There are some cases we can not do appearance matching. When the size of target is not large enough*/
class AppFeature
{
public:
	enum{
		BIN_SIZE = 8,
		MIN_WIDTH = 32,
		MIN_HEIGHT = 64,
	};

private:
	float colorHist[BIN_SIZE*BIN_SIZE*BIN_SIZE]; //color histogram
	EstimatedGaussDistribution m_pos[BIN_SIZE*BIN_SIZE*BIN_SIZE]; //estimated distribution pos
	void accumulateToHist(ImageRepresentation *image, const cv::Rect &bound);
	
public:
	bool valid;//Indicate whether the appearance model is valid
	SemiBoostingTracker tracker; //SemiBoosting Tracker
	sir_filter::SIRFilter particle_filter; //particle filter
	sir_filter::State InitParticleFilter(const cv::Rect &detection); //set the initial samples & weight for particle filter
	sir_filter::State InitParticleFilter(const sir_filter::State &state);
	
	void SuperviseTraining(ImageRepresentation *image, const cv::Rect &innerbound);
	void SemiSupervisedTraining(ImageRepresentation *image, const cv::Rect &innerbound);
	void Clear(); //Clear all member instances
	void DetectionCheck(const float vec[2], const cv::Rect &detection,const std::list<ImageRepresentation*> &frame_buffer, int frame_no, const cv::Rect &predict_bound, sir_filter::State &max_state); // particle filtering when associated detection is available
	float CalWeightForParticle(const cv::Rect &detection, ImageRepresentation *image, const std::vector<sir_filter::State> &particles, std::vector<float> &weight, sir_filter::State &max_state); // Calculate weight for each particle at detection stage or check stage
	
	static void extractHistFeature(float hist[BIN_SIZE*BIN_SIZE*BIN_SIZE],const cv::Mat &ROI);
	float evalHist(const float hist[BIN_SIZE*BIN_SIZE*BIN_SIZE]) const;

	sir_filter::State GetObjState();
	float TestSimi(ImageRepresentation *image, const cv::Rect &detection, bool use_negative); //Test similarity score of detection area (This may be determined by both svrs. Positive correlated)
	float TestHistSimi(ImageRepresentation *image, const cv::Rect &detection) const; //generative hist descriptor
	static cv::Rect CropDetection(const cv::Rect &original); //Crop detection to rigid blob for appearance tracker
	AppFeature();
	~AppFeature(void);
};