#include "TrackObjAff.h"

TrackObjAff::TrackObjAff(void)
{
}


float TrackObjAff::bimodalNormalDistribution(const float firstMean[2],const float secondMean[2],
	float firstStd,float secondStd,const float value[2], bool pre) const
{
	const float first_Std[]={firstStd,firstStd};
	const float second_Std[]={secondStd,secondStd};
	float prob1=stdfunc::GaussianDistribution(firstMean,first_Std,2,value,pre);
	float prob2=stdfunc::GaussianDistribution(secondMean,second_Std,2,value,pre);
	if(pre)
		return prob1+prob2;
	else
		return (prob1+prob2)/2;
}