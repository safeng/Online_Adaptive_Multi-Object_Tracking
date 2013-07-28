#pragma once
#include "ObjTrack.h"
#include <queue>

typedef std::queue<ObjTrack*> ObjTrackQueue; 
typedef std::queue<ObjSubTrack*> ObjSubTrackQueue;

/*space management of objtracks & objsubtracks*/
class TrackSpaceAlloc
{
private:
	ObjTrack *trackSet; // Defines the internal space for object trajectories
	ObjSubTrack *subTrackSet;//Defines pool of instances
	int subTrackNO;
	ObjTrackQueue objTrackQueue;
	ObjSubTrackQueue objSubTrackQueue;
	int IncreaseSubTrackCount(){if(subTrackNO>INT_MAX) subTrackNO = 1; else ++subTrackNO; return subTrackNO;} //assign a unique index
public:
	void push_free_objTrack(ObjTrack *objTrack);
	ObjTrack *pop_free_objTrack();
	ObjSubTrack *pop_free_objSubTrack();
	void push_free_objSubTrack(ObjSubTrack *subTrack);
	TrackSpaceAlloc(void);//space allocation
	~TrackSpaceAlloc(void);//space deallocation
};