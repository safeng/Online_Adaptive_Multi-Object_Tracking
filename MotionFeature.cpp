#define _USE_MATH_DEFINES
#include "MotionFeature.h"

MotionFeature::MotionFeature(int capacitySet,int capacityTrack):feature_map(NULL)
{
	capacityOfFeatureSet=capacitySet;
	capacityOfFeatureTrack=capacityTrack;
	numberOfFeatureTracks=0;
	trackAllocationMap=(bool*)malloc(capacityOfFeatureSet*sizeof(bool));
	featureSet=(FeatureTrack*)calloc(capacitySet,sizeof(FeatureTrack));//allocate space for feature trajectories array
}

MotionFeature::~MotionFeature(void)
{
	//Since our memory is not continuously allocated, we need to free it one by one
	for(int i=0,p=0;i<numberOfFeatureTracks;++i,++p)//Since the track set are not continue, there may be memory leak here
	{
		//find the next valid feature trajectory
		while(trackAllocationMap[p])
			p++;
		freeTrack(featureSet[p]);
	}
	free(trackAllocationMap);
	free(featureSet);
	delete[] feature_map;
}

void MotionFeature::AddNewTrack(KLT_Feature feature,int featureNo,int startFrame)
{
	if(++numberOfFeatureTracks>capacityOfFeatureSet)//over the total capacity
	{
		/*Function to enlarge the capacity of track set*/
		fprintf(stderr,"Track set overloads!\n");
		assert(false);
	}
	// the ith feature track
	int nbytes=sizeof(FeatureTrackRec)+capacityOfFeatureTrack*sizeof(KLT_Feature);
	featureSet[featureNo]=(FeatureTrack)malloc(nbytes);//allocate the memory for the ith feature track and the pointers for all its features
	FeatureTrack ftr=featureSet[featureNo];// Insert at the featureNo place
	ftr->indexOfObjTrack = -1;// (invalid index)
	ftr->num_matches = 0;
	ftr->featureNo = featureNo;
	ftr->startFrame = startFrame;
	ftr->capacity = capacityOfFeatureTrack;
	ftr->length = 1;//set the length to 1
	//set the pointer
	ftr->feature=(KLT_Feature*)(ftr+1);//the starting address for feature pointers
	ftr->feature[0] = copyKLT_Feature(feature);// the first feature in the ith trajectory
	trackAllocationMap[featureNo]=false;//Mark not free
}

void MotionFeature::RemoveTrack(int featureNo)
{
	//We must make sure that we do not delete the same track for twice
	if(featureSet[featureNo]==NULL)//have been deleted
		return;
	//We assume that feature number equal to the position of the track
	if(featureSet[featureNo]->featureNo!=featureNo)
	{
		fprintf(stderr,"The feature No.%d is not compatible with %d in that track!",featureNo,featureSet[featureNo]->featureNo);
		assert(featureSet[featureNo]->featureNo==featureNo);
		return;
	}
	//free up all the memory used by the track
	numberOfFeatureTracks--;
	freeTrack(featureSet[featureNo]);// Now that the featureNo th track has been empty
	//set the allocation map to indicate that featureNo track is free now
	featureSet[featureNo] = NULL;//avoid dangling pointer
	trackAllocationMap[featureNo]=true;
}

void MotionFeature::AppendNewFeatureToTrack(int featureNo,KLT_Feature feature)
{
	//we set the featureNo equal to the position of the feature trajectory in the set
	if(featureSet[featureNo]->featureNo!=featureNo)
	{
		fprintf(stderr,"The feature No.%d is not compatible with %d in that track!",featureNo,featureSet[featureNo]->featureNo);
		assert(featureSet[featureNo]->featureNo==featureNo);
		return;
	}
	FeatureTrack track = featureSet[featureNo];
	++track->length;
	if(track->length>track->capacity)// Over the current capacity
	{

		/*事实上我们只需要将多余的信息删除掉，只留下最后reserved帧的信息*/
		int reserved=track->capacity/2;
		for(int i=0;i<reserved;i++)
		{
			memcpy(track->feature[reserved-1-i],track->feature[track->length-2-i],sizeof(KLT_FeatureRec));
		}
		//Free the extra space
		for(int i=reserved;i<track->length-1;i++)
			free(track->feature[i]);
		track->startFrame+=track->length-1-reserved;
		track->length=reserved+1;//Set the length
	}
	track->feature[track->length-1]=copyKLT_Feature(feature);
}

void MotionFeature::freeTrack(FeatureTrack track)
{
	if(track==NULL)
		return;
	for(int i=0;i<track->length;i++)
		free(track->feature[i]);
	free(track);
}

void MotionFeature::WriteFeatureTracksToFile(char *fileName)
{
	FILE *fp=fopen(fileName,"wb");
	assert(fp!=NULL);
	for(int i=0,p=0; i<numberOfFeatureTracks; i++,p++)// for each feature track
	{
		while(trackAllocationMap[p])//we find the next allocated/valid track
			p++;
		fprintf(fp,"Feature Track %3d--%3d: ",i,p);
		//Just for alignment
		for(int k=0;k<featureSet[p]->startFrame;k++)
			fprintf(fp,"(%3c,%3c)\t",'*','*');
		for(int j=0;j<featureSet[p]->length;j++) // for each feature in ith feature track
		{
			fprintf(fp,"[%3d,%3d]\t",
				int(featureSet[p]->feature[j]->pos_x+0.5f),
				int(featureSet[p]->feature[j]->pos_y+0.5f));
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}

/*check whether the ith feature trajectory satisfies the internal requirement*/
bool MotionFeature::CheckValidityOfFeatTrack(int i, const KLT_TrackingContext tc, int &replace)
{
	FeatureTrack set = featureSet[i];
	if( set->length < MO_GROUP )
		return true;
	KLT_Feature featCur = set->feature[set->length-1]; // the current added new feature
	KLT_Feature featPre = set->feature[set->length-MO_GROUP];// The pre-pre feature
	float vec_x = featCur->pos_x - featPre->pos_x;
	float vec_y = featCur->pos_y - featPre->pos_y;

	/*Note that we define the instant velocity as current minus previous MO_GROUP. 
	Thus the first MO_GROUP features will not have instant velocity*/
	featCur->velocity[0] = vec_x;//this feature has instant velocity now
	featCur->velocity[1] = vec_y;

	//可以加入一些更加成熟的检测机制,确认 feature track 是否valid
	//if(!tc->CheckBgEnabled())
	//{
		//return true;
	//}
	if(set->length > NON_KEYFRAME_GAP)//We have gained enough evidence to check whether it is a static feature
	{
		//check the recent magnitude of movement over MIN_STATIC_CHECK_LEN
		float overAllMagnitude(0);
		for(int k=0; k<NON_KEYFRAME_GAP; ++k)
		{
			int vecno = set->length-1-k;//subscript
			overAllMagnitude += abs(set->feature[vecno]->velocity[0])+abs(set->feature[vecno]->velocity[1]);
		}

		if((float)overAllMagnitude < 1.0f && set->indexOfObjTrack == -1)//considered as static feature
		{
			//check whether it comes from background
			//int x = int(featCur->pos_x + 0.5f);
			//int y = int(featCur->pos_y + 0.5f);

			//check against background
			//if(!tc->DetermineForeground(y,x)) //As background pixel
			//{
				return false;
			//}else //foreground pixel
			//	return true;
		}
	}

	//Occupation test
	int pos_x = int(featCur->pos_x + 0.5f);
	int pos_y = int(featCur->pos_y + 0.5f);

	if (pos_x>=image_width || pos_y>=image_height || pos_x<0 || pos_y<0)
	{
		return false;
	}

	int index = pos_y * image_width + pos_x;

	if (feature_map[index]==-1)//not occupied
	{
		feature_map[index] = i;
	}else
	{
		FeatureTrack set_comp = featureSet[feature_map[index]];
		if (set_comp->indexOfObjTrack!=-1)
		{
			if (set->indexOfObjTrack==-1)
			{
				return false;
			}else
			{
				if (set_comp->length>=set->length)
				{
					return false;
				}else
				{
					replace = feature_map[index];
					feature_map[index] = i;
				}
			}
		}
	}

	return true;
}

void MotionFeature::ResetOccupationMap()
{
	if (feature_map==NULL)
	{
		feature_map = new int[image_width*image_height];
		std::fill(feature_map,feature_map+image_width*image_height,-1);
	}else
	{
		std::fill(feature_map,feature_map+image_width*image_height,-1);
	}
}

/*copy the value, since feature will later be released*/
KLT_Feature MotionFeature::copyKLT_Feature(KLT_Feature feature,KLT_Feature out)
{
	assert(feature!=NULL);
	if(out==NULL)// we need to allocate new space
		out = (KLT_Feature)malloc(sizeof(KLT_FeatureRec));
	out->val = feature->val;
	out->pos_x = feature->pos_x;
	out->pos_y = feature->pos_y;
	memset(out->velocity,0,sizeof(int)*2);
	return (out);
}