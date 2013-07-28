#pragma once
#include "AppFeature.h"

/*State of objSubTrack at frame_no*/
struct DetectionModel
{
	int start_frame_no; //the frame no when this detection model is added
	float center_x, center_y; //the central position for each detection model
	float width,height;
	bool supported; // Indicate whether the model is supported by the feature tracks (false if no feature tracks included)
	bool external_detection;
	float vec[2]; // average velocity 

	//By default it inherent vec from previous frame
	DetectionModel();
	DetectionModel(float center_x, float center_y, float width, float height, const float vec[2], int start_frame_no);

	cv::Rect ConvertToRect() const
	{
		return cv::Rect( int(center_x - width/2 + 0.5f), int(center_y - height/2 + 0.5f), int(width+0.5f), int(height+0.5f));
	}

	void SetFromState(const sir_filter::State &avg_state);
	void SetFromDetection(const cv::Rect &detection);
};

struct FeatureNode
{
	int featureNO;
	float dist;
	friend bool operator < (FeatureNode a, FeatureNode b)
	{
		return a.dist > b.dist;//Smaller distance higher priority
	}
};

//space, scale affinity
struct AffinityVal
{
	int index;
	float probScale;
	float probPos;
	float proApp;
};

/*Define a track for each detection. In ideal situation, one person has one track*/
class ObjSubTrack
{
private:
	static int curTrackNO; //current subtrack index
	std::list<ObjSubTrack*> negTrackList;
	float GetVelocityDifference(const float vec_1[2], const float vec_2[2], float scale) const;
	float VelocityDifferenceNoScaling(const float vec_1[2], const float vec_2[2]) const;
	void ComputeVecDiff(const MotionFeature *motionFeature, const int mappingArray[], int frameNO, 
		std::list<FeatureNode> &featureNO_list);//selected set of feature tracks
	float MotionConformity(const FeatureTrack track, int frameNO, const float most_cur_vec[2]) const;
	int SelectFeatureTrackMotionDescriptor(std::list<FeatureNode> &feat_no_list, const MotionFeature *motionFeature, int frame_no, bool instant);
	bool GeneratePreVecByPre(const MotionFeature *motionFeature,int frameNO);
	bool UpdateInstantVec(const ObjSeg *objSeg); //whether supported by feature tracks 
	void StoreToConfFeatList(const std::list<FeatureNode> &feat_no_list, const MotionFeature *motionFeature);
	void FindSpatialCandidateFeatures(int cur_frame_no, const MotionFeature *motionFeature, const int mappingArray[]);
	void setStatesFromEvidence();
	void generateVecFromReliableSources(int cur_frame_no);
	void EstimateVec(float vec[2], const sir_filter::State &max_state, int cur_frame_no) const;
	void EstimateVec(float vec[2], const cv::Rect &evidence, int cur_frame_no) const;
	void averageScale(cv::Rect &detection);
	bool AppearanceCheck(const ObjSeg *objseg, std::list<ImageRepresentation*> frame_buffer, sir_filter::State &max_state, int &valid_frame_no);

public:
	uchar color[3]; // color of track
	bool inactive; // Indicate whether the track is inactive because of no evidence for long period
	bool main_in_track; //whether it is a main subtrack to represent an object
	std::list<int> confidence_feat_list; //Confidence feature list. Updated when supported. Used when it meets cluttered background
	float cur_weight; //weight in the current frame
	size_t matched_len; //# of evidence from external detection
	int subTrackNO; //The index of each subTrack
	int index; //pos in objsubtrack
	size_t missing_count; //num of continuous missing count
	size_t total_prediction_count; // # of times we predict model only by constant velocity model
	bool isValid; //Indicate whether it is storage valid

	std::list<int> spatialFeatTracks; //Feature tracks that obeys spatial temporal conformity
	std::list<DetectionModel> detectionSubTrack; //The detection sub track for head
	DetectionModel predicted_model; // Hypothesized state predicted by motion model
	AppFeature appModel; //appearance model. Using Mean-shift to find best matching

	static int IncreaseSubTrackCount();
	bool TestMostSimilar(ImageRepresentation *image, const cv::Rect &detection, float histSimi, float gap) const;
	void AddToNegList(const std::list<ObjSubTrack *> &negTrackList);

	bool TestFeatureSupport(const ObjSeg *objseg, int curFrameNO, const cv::Rect &detection) const;
	float KLTFeatureSimi(const ObjSeg *objseg, int curFrameNO, const cv::Rect &detection) const;

	void SupervisedTraining(const std::list<ImageRepresentation *> &framePool);
	void SemiSupervisedTraining(const std::list<ImageRepresentation *> &framePool);

	void Release(const ObjSeg *objSeg); //Release resources
	void UpdateProbability(const ObjSeg *objSeg);
	bool PredictModel(const ObjSeg *objSeg, const std::list<ImageRepresentation*> &frame_buffer, bool external_detection);
	bool ParticleFilteringWithApp(const ObjSeg *objSeg, const std::list<ImageRepresentation*> &frame_buffer); //Tracking without external detection. On appearance model
	bool UpdateModel(const ObjSeg *objSeg, const std::list<ImageRepresentation*> &frame_buffer, const cv::Rect &detection);//supported by some head evidence
	bool ComputeAffWithObjSubTrack(const ObjSubTrack *track) const; //Compute aff with another track
	bool TestAddModel(const cv::Rect &evidence) const;
	void ReplaceWithNewDetection(const ObjSeg *objSeg, const cv::Rect &evidence); //Replace original sub track to init a new track
	bool TestValidity();
	void ComputeAffinityValue(const cv::Rect &detection, float &scaleProb, float &posProb);
	void PaintPariticleStates(cv::Mat &img) const;
	ObjSubTrack(void);
	~ObjSubTrack(void);
};