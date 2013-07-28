#pragma once

#include "PeopleDetector.h"
#include "ObjTrack.h"
class TrackAssociation
{
private:
	std::list<ImageRepresentation*> frame_pool; // frame pool to store inter medium frames
	ObjSeg *objSeg;
	std::vector<ObjSubTrack *> subTrackVector;//track vector for each detection
	std::vector<ObjTrack *> trackVector;//The vector used to store all the current object trajectories (validity is denoted by the field of the objtrack)
	PeopleDetector humanDetector;
	void AppendNewNode_InitNewTrack(int frameNO,const ObjSeg *objSeg);//Append new node to the existing tracks and init new tracks
	void PeopleDetection(int frameNO, const cv::Mat &image, std::vector<cv::Rect> &evidence, std::vector<float> &weight_evidence);
	void UpdateModelsByPrediction();
	void CalculateAffinityScore(const std::vector<cv::Rect> &evidence, std::vector<AffinityVal> &affinity_mat);
	void FindMatch(std::vector<AffinityVal> &affinity_mat, const std::vector<cv::Rect> &evidence, std::vector<int> &assign_update, std::vector<bool> &evidence_used, std::vector<bool> &model_used);
	void UpdateModel(const ObjSeg *objSeg, const std::vector<cv::Rect> &evidence, const std::vector<float> &weight_evidence, const std::vector<int> &assign_match, std::vector<bool> &evidence_used,
			std::list<ObjSubTrack*> &new_track_list, std::list<float> &new_weight_list);
	void CreateNewObjTracks(const ObjSeg *objSeg, const std::vector<cv::Rect> &evidence, const std::vector<float> &evidence_weight, const std::vector<bool> &evidence_used, 
		const std::list<ObjSubTrack*> &new_track_list, const std::list<float> &new_weight_list);
	bool TestForegroundDetection(const cv::Rect &detection, float th);//test whether 'detection' comes from foreground
	void DrawDetectionModels(const cv::Mat &frame, int frameNO);//Draw detection models
	void CleanSubTrackVector();
	void InitAppearanceModelForMainTracks();
	void TrackWithoutExternalDetection(const ObjSeg *objSeg);
	void SuperviseTrainingForSubTracks();
public:
	void ConstructFramePoolPointer(std::list<ImageRepresentation*> &frame_p_list);
	size_t GetPoolSize() const {return frame_pool.size();}
	void StoreToFramePool(const cv::Mat &cur_frame, int frameNO);
	void ClearFramePool();
	TrackAssociation(void);
	~TrackAssociation(void);
	void ObjectTrackingAtKeyFrame(int frameNO, cv::Mat &image, bool external_detection);//key frame multi-object tracking
	ObjSeg *GetObjSeg(){return objSeg;}
};