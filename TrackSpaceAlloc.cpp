#include "TrackSpaceAlloc.h"

TrackSpaceAlloc::TrackSpaceAlloc(void):subTrackNO(1)
{
	trackSet = new ObjTrack[MAX_OBJ_TACKSET_SIZE];
	subTrackSet = new ObjSubTrack[MAX_OBJSUBTRACKSET_SIZE];

	for(int i=0;i<MAX_OBJ_TACKSET_SIZE;i++)
	{
		objTrackQueue.push(trackSet+i);
	}

	for(int i=0;i<MAX_OBJSUBTRACKSET_SIZE;i++)
	{
		objSubTrackQueue.push(subTrackSet+i);
	}
}

TrackSpaceAlloc::~TrackSpaceAlloc(void)
{
	delete[] trackSet;
	delete[] subTrackSet;
}

/*Add one free objTrack object*/
void TrackSpaceAlloc::push_free_objTrack(ObjTrack *objTrack)
{
	assert(objTrack->isValidStorage);
	objTrack->isValidStorage = false;
	objTrackQueue.push(objTrack);
}

/*Pop one free objTrack for use*/
ObjTrack *TrackSpaceAlloc::pop_free_objTrack()
{
	if(objTrackQueue.empty())
		return NULL;
	ObjTrack *objTrack=objTrackQueue.front();
	assert(!objTrack->isValidStorage);//First false
	objTrack->isValidStorage = true;
	objTrackQueue.pop();
	return objTrack;
}

/*Pop one free objsubtrack*/
ObjSubTrack *TrackSpaceAlloc::pop_free_objSubTrack()
{
	if(objSubTrackQueue.empty())
		return NULL;
	ObjSubTrack *objSubTrack = objSubTrackQueue.front();
	assert(!objSubTrack->isValid);
	objSubTrack->isValid = true;
	objSubTrack->subTrackNO = IncreaseSubTrackCount();
	objSubTrackQueue.pop();
	return objSubTrack;
}

/*Push one invalid objsubTrack*/
void TrackSpaceAlloc::push_free_objSubTrack(ObjSubTrack *subTrack)
{
	assert(subTrack->isValid);//must be valid first
	subTrack->isValid = false;
	objSubTrackQueue.push(subTrack);
}