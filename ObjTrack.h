#pragma once
#include "ObjSubTrack.h"

class ObjTrack
{
private:
	static const float init_th; //we init a new sub trajectory when we have enough confidence on weight of each detection submodel
	static const float valid_th; //we consider a sub model to be valid 
	static const int upper_limit; //upper limit of # of sub detection models in this model
	void MatchingUpdate(std::vector<ObjSubTrack*> &subTrackVector); //Find matching detections to support existing sub models
	ObjSubTrack* SortWeight(); //Sort to find possible candidates
	void CreateNewSubModel(const ObjSeg *objSeg, std::vector<ObjSubTrack*> &subTrackVector, const cv::Rect &evidence, float weight);//Create new sub model for unmatched evidence
	void NormalizeWeight(); //Normalize the weight for all sub models
	void RemoveIns(const ObjSubTrack *ins);
public:
	void CreateWithNewEvidence(const ObjSeg *objSeg, std::vector<ObjSubTrack*> &subTrackVector, const cv::Rect &evidence, float weight);
	void CreateWithObjSubTrack(ObjSubTrack *sub_track, float weight); // Init an object track with an existing obj sub track
	bool UpdateEachModel(const ObjSeg *objSeg); //Update each objsubmodel inside (update and remove outliers)
	std::list<float> norm_weight; //normalized weight for each detection
	std::list<float> weight; //original weight for each detection
	std::list<ObjSubTrack*> detection_model; //each detection model (A vector of objsubtracks)
	bool ProduceResult(const ObjSeg *objSeg, const std::list<ImageRepresentation*> &frame_buffer, const std::vector<cv::Rect> &evidence, const std::vector<float> &evidence_weight, 
		std::vector<bool> &evdence_used, std::vector<ObjSubTrack*> &subTrackVector, std::list<ObjSubTrack*> &new_track_list, std::list<float> &new_weight_list);//sort norm_weight*match_length/sigma to produce result for the current stage
	bool isValidStorage;//Mark whether this track is a valid storage
	ObjTrack(void);
	~ObjTrack(void);
};