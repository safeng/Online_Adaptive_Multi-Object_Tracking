#pragma once
//#define _CRTDBG_MAP_ALLOC
//#include <crtdbg.h>
#include "KLT.h"
#include "trackingParam.h"

typedef KLT* KLT_TrackingContext; 

typedef struct
{
	float mean_vec[2];//Mean vec between key frame gaps
	int featureNo;//the ith feature trajectory
	int startFrame;//the start frame number
	int length;// the total length of the trajectory
	int capacity;// it's current capacity for the features
	KLT_Feature *feature;// feature array that dynamically increase as we incrementally gain more frames
	int indexOfObjTrack; // The index of the objTrack it is associated
	int num_matches; //# of times this feature is selected as confident feature track
}FeatureTrackRec,*FeatureTrack;//Struct for the feature track

class MotionFeature
{
private:
	int capacityOfFeatureSet;// set the initial number of trajectory set
	int capacityOfFeatureTrack;// set the initial length of a track (how many features in the track)
	void freeTrack(FeatureTrack track);// free the space of track
public:
	int numberOfFeatureTracks;//total number of feature tracks
	FeatureTrack *featureSet;// a set of feature trajectories varing in length
	//note that we need to allocate memory space for all attributes
	bool *trackAllocationMap;//Keep the allocation map for the trajectories set
	int *feature_map;//mark the occupation of feature map;
	void ResetOccupationMap();
	MotionFeature(int capacitySet, int capacityTrack);
	~MotionFeature(void);
	//construct the new KLT_Feature as the copy of the input. If the second parameter is NULL, we allocate new space
	static KLT_Feature copyKLT_Feature(KLT_Feature feature, KLT_Feature out = NULL);

	void AddNewTrack(KLT_Feature feature,int featureNo,int startFrame);/* add a new track in the set, memory allocation needed. And the new track only has one feature at the moment*/
	
	void AppendNewFeatureToTrack(int featureNo,KLT_Feature feature);/*append a new feature to the ith trajectory.Note: we just copy the value and allocate memory space.Thus we can free the memeory tracked by KLT*/
	
	void RemoveTrack(int featureNo);/*Remove the whole trajectory from trajectory set. Note: we should free the memory space*/
	
	void WriteFeatureTracksToFile(char *fileName);/*Add one optical flow from to the trajectory i's histogram. We add the vector appears most recently  by default and retrieve every other feature*/
	
	bool CheckValidityOfFeatTrack(int i, const KLT_TrackingContext tc, int &replace);//Return false, if the track is considered as static, true otherwise
};