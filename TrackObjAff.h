#pragma once
#include "PeopleDetector.h"
#include "stdfunc.h"
/*It defines how we compute the object affinity, this object must live through the whole application*/

class TrackObjAff
{
private:
	float bimodalNormalDistribution(const float firstMean[2], const float secondMean[2],
		float firstStd,float secondStd, const float value[2], bool pre = true) const;//Return the value of a bimodal normal distribution
public:
	PeopleDetector detector; //Human detector
	TrackObjAff(void);
};