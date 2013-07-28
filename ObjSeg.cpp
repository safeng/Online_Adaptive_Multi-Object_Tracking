#include "ObjSeg.h"

/*ObjSeg live though out the main application*/
ObjSeg::ObjSeg():
frameNO(0),mappingArray(NULL)
{
	motionFeature = new MotionFeature(NUMTRACKS,NUMFEATURES);
	//since we have not done operations to motion features, we cannot init object instances here
}

ObjSeg::~ObjSeg(void)
{
	delete motionFeature;//delete the instance that has the same life span
}

void ObjSeg::ResetOccupationMap()
{
	motionFeature->ResetOccupationMap();
}

/*Delete, Append and Add operations are same for tracking frames and detection frames*/
void ObjSeg::Delete_Append_FeatureTrackSet(KLT_FeatureList list, std::list<int> &missedList)
{
	for(int i = 0; i<list->nFeatures; i++)//For all the features
	{
		if(list->feature[i]->val>=0)// The successfully tracked features
		{
			//Append new features
			
			motionFeature->AppendNewFeatureToTrack(i,list->feature[i]);
			int replace=-1;

			bool valid = motionFeature->CheckValidityOfFeatTrack(i,tc,replace);
			if(!valid)//check the validity
			{
				list->feature[i]->val = -2;//Set to an invalid num (but an indication to the following step)
				motionFeature->RemoveTrack(i);
				missedList.push_back(i);
			}

			if (replace!=-1)//have to replace
			{
				assert(valid);
				list->feature[replace]->val = -2;
				motionFeature->RemoveTrack(replace);
				missedList.push_back(replace);
			}

		}else// missed features
		{
			//Delete the corresponding feature track
			motionFeature->RemoveTrack(i);
			missedList.push_back(i);//push the missed feature no. to the list
		}
	}

	missedList.sort();
}

/*list is the Feature List after replacement and addition;
originalLength is the length of original featureList before replacement;
missedList the list containing missed frame NO.*/
void ObjSeg::Add_FeatureTrackSet(KLT_FeatureList list,int originalLength,const std::list<int> &missedList)
{
	//According to the list of missing frame numbers and the length of feature list, add new tracks
	int lengthOfFeatureList=list->nFeatures;
	if(list->nFeatures-originalLength==0)//less features (Note that replace operation will not shrink feature lists' length)
	{
		std::list<int>::const_iterator iter;
		for(iter=missedList.cbegin();iter!=missedList.cend()&&list->feature[*iter]->val>0;iter++)
		{
			motionFeature->AddNewTrack(list->feature[*iter],*iter,frameNO);//Add new tracks at missing place
		}
	}else//more features, assume that all the rest new features append at last of list
	{
		for(std::list<int>::const_iterator iter = missedList.cbegin(); iter!=missedList.cend(); iter++)
		{
			if(list->feature[*iter]->val <= 0)//Make sure it's the valid new feature
			{
				printf("%d\n",list->feature[*iter]->val);
				assert(false);
			}
			motionFeature->AddNewTrack(list->feature[*iter],*iter,frameNO);
		}
		//Append the rest new features
		for(int i=originalLength;i<list->nFeatures;i++)
		{
			assert(list->feature[i]->val>0);
			motionFeature->AddNewTrack(list->feature[i],i,frameNO);
		}
	}
}

/*set each feature track its actual storage space*/
void ObjSeg::setMappingArray()
{
	assert(mappingArray==NULL);
	int numFeatureTracks=motionFeature->numberOfFeatureTracks;
	mappingArray=(int*)malloc(sizeof(int)*numFeatureTracks);
	for(int i=0,x=0;i<numFeatureTracks;++i,++x)
	{
		while(motionFeature->trackAllocationMap[x])
			x++;//find the valid track number
		mappingArray[i] = x;
	}
}

/*Calculate avg velocity for features between frame gaps*/
void ObjSeg::CalAvgVecBwtFrames(int frame_gap)
{
	//After the operations on motion feature tracks and feature lists are done
	setMappingArray();
	int numberOfFeatures = this->motionFeature->numberOfFeatureTracks;
	
	for(int i = 0; i<numberOfFeatures; i++)//loop for each feature track
	{
		int p = mappingArray[i];
		//extract the number ith feature at the current frame
		int lengthOfTracki = motionFeature->featureSet[p]->length;
		assert(lengthOfTracki==frameNO-motionFeature->featureSet[p]->startFrame+1);
		KLT_Feature feature = motionFeature->featureSet[p]->feature[lengthOfTracki-1];
		FeatureTrack track = motionFeature->featureSet[p];
		if(lengthOfTracki >= MO_GROUP)//We are able to calculate its velocity
		{
			/*The following code calculates the average velocity value*/
			if(lengthOfTracki > frame_gap) // long enough
			{
				//Use the average velocity between the non_keyframe_gap
				KLT_Feature featurePre = motionFeature->featureSet[p]->feature[lengthOfTracki-frame_gap-1];
				float move_x = feature->pos_x - featurePre->pos_x;
				float move_y = feature->pos_y - featurePre->pos_y;
				float vec_x = move_x/(float)(frame_gap/(MO_GROUP-1));
				float vec_y = move_y/(float)(frame_gap/(MO_GROUP-1));

				motionFeature->featureSet[p]->mean_vec[0] = vec_x;//caculate velocity in uint of MO_GROUP
				motionFeature->featureSet[p]->mean_vec[1] = vec_y;//caculate velocity in uint of MO_GROUP

				//把gap里每个feature的即时速度都设置成相同的
				for(int i = lengthOfTracki-1; i > lengthOfTracki-frame_gap-1; --i)
				{
					motionFeature->featureSet[p]->feature[i]->velocity[0] = vec_x;
					motionFeature->featureSet[p]->feature[i]->velocity[1] = vec_y;
				}

				if(lengthOfTracki-frame_gap-1 < MO_GROUP-1)
				{
					for(int i = 0; i<MO_GROUP-1; ++i)
					{
						motionFeature->featureSet[p]->feature[i]->velocity[0] = motionFeature->featureSet[p]->feature[MO_GROUP-1]->velocity[0];
						motionFeature->featureSet[p]->feature[i]->velocity[1] = motionFeature->featureSet[p]->feature[MO_GROUP-1]->velocity[1];
					}
				}
			}else
			{
				//Calculate velocity as long as possible
				KLT_Feature featurePre = motionFeature->featureSet[p]->feature[0];//Use the start frame
				float move_x = feature->pos_x - featurePre->pos_x;
				float move_y = feature->pos_y - featurePre->pos_y;
				float vec_x = move_x/(float)((lengthOfTracki-1)/(MO_GROUP-1));
				float vec_y = move_y/(float)((lengthOfTracki-1)/(MO_GROUP-1));
				motionFeature->featureSet[p]->mean_vec[0] = vec_x;
				motionFeature->featureSet[p]->mean_vec[1] = vec_y;
				//把即时速度都设置成一样的
				for(int i = lengthOfTracki-1; i >= 0; --i)
				{
					motionFeature->featureSet[p]->feature[i]->velocity[0] = vec_x;
					motionFeature->featureSet[p]->feature[i]->velocity[1] = vec_y;
				}
			}
		}else
		{
			motionFeature->featureSet[p]->mean_vec[0] = -1;
			motionFeature->featureSet[p]->mean_vec[1] = -1;	
		}
	}
}

/*This function must be called after we have formed objects from the clusters*/
void ObjSeg::ResetMappingArray()
{
	free(mappingArray);
	mappingArray = NULL;
}

void ObjSeg::WriteFeatureListToRGBImg(const cv::Mat &img,const KLT_FeatureList fl,const char *filename)
{
	fl->WriteToImage(std::string(filename),img);
}

void ObjSeg::fillMap(cv::Mat &ROI, const KLT_FeatureList fl, int gap)
{
	for(int i = 0 ;i<fl->nFeatures; ++i)
	{
		if(fl->feature[i]->val>=0)
		{
			int x = int(fl->feature[i]->pos_x);
			int y = int(fl->feature[i]->pos_y);
			for(int xx = x-gap; xx<=x+gap; ++xx)
			{
				if(xx>=0 && xx<ROI.cols)
				{
					for(int yy = y-gap; yy<=y+gap; ++yy)
					{
						if(yy>=0 && yy<ROI.rows)
						{
							ROI.at<uchar>(yy,xx) = 0;
						}
					}
				}
			}
		}
	}
}

/*TYPE: TRUE means it is the tracking period----border detection mode
			FALSE means it is the detection period----whole screen detection mode*/
KLT_FeatureList ObjSeg::ReplaceFeatureList(const cv::Mat &img, KLT_FeatureList inputlist,const std::list<int> &missedList, int mindist, bool type)
{
	KLT_FeatureList newlist = NULL;
	KLT_FeatureList fl;//The new features to be added 
	int list_size(0);
	if(type)//border mode
	{
		list_size = NUMINITFEATURES;
	}
	else//whole screen mode, intuitively, we give it a larger space
	{
		list_size = NUMINITFEATURES * 2;
	}

	fl=KLTCreateFeatureList(list_size);//Note that: the newly selected features may not be valid features, the rest may be bad features with value = -1

	if(type)//we are tracking
	{
		//fillMap(inputlist,mindist,searchMapTracking);//We only want to detect at the borders and non-tracked area
		//tc->SelectGoodFeaturesToTrack(img,fl,searchMapTracking);
		//memcpy(searchMapTracking,tempMapTracking,sizeof(bool) * image_width * image_height); //Restore the searchMap with original border value (border template)
	}else if(!type)//we are detecting at whole screen
	{
		//searchMapClustering is hard constraint
		cv::Mat newROI(img.rows,img.cols,cv::DataType<uchar>::type);

		fillMap(newROI,inputlist,mindist);//We only want to detect at non-tracked area (whole screen)

		tc->SelectGoodFeaturesToTrack(img,fl,newROI);
		//Ensure that list contains only valid features
		//memset(searchMapDetection,true,sizeof(bool) * image_width * image_height);//reset to all true (whole screen)
	}else
	{
		fprintf(stderr,"No such mode!\n");
		assert(false);
	}

	//Add the new features to the input feature list
	//We ensure that the newly added features are all valid
	std::list<int>::const_iterator iter = missedList.cbegin();
	int p=0;
	while(p<fl->nFeatures && iter!=missedList.cend())
	{
		if(fl->feature[p]->val<0)//meet the invalid features, the rest are all invalid
		{
			KLTFreeFeatureList(fl);
			return inputlist;//do not need to append
		}
		moveFeature(inputlist,fl,*iter,p);//Move features into destination
		p++;iter++;
	}

	if(iter!=missedList.cend() || p==fl->nFeatures&&iter==missedList.cend())
	{
		KLTFreeFeatureList(fl);
		return inputlist;
	}
	else//we need to append to the current list, enlarge the current capacity
	{
		//Compute the rest valid number
		int count=p;
		for(;p<fl->nFeatures&&fl->feature[p]->val>=0;p++);
		int numValid = p-count;
		newlist = AppendNewFeaturesToList(inputlist,fl,count,numValid);//All features are valid
	}
	KLTFreeFeatureList(fl);
	return newlist;
}

void ObjSeg::WriteTrackToFile(char *fname)
{
	motionFeature->WriteFeatureTracksToFile(fname);
}

KLT_FeatureList ObjSeg::AppendNewFeaturesToList(KLT_FeatureList inputList,KLT_FeatureList appendedList,int startIndex,int length)
{
	int originalLength=inputList->nFeatures;
	assert(originalLength==KLTCountRemainingFeatures(inputList));//All are valid
	int totalNumFeatures=originalLength+length;
	KLT_FeatureList newList=KLTCreateFeatureList(totalNumFeatures);
	//memcpy(newList->feature,inputList->feature,inputList->nFeatures);//Memory Copy
	memcpy(newList->feature[0],inputList->feature[0],inputList->nFeatures*sizeof(KLT_FeatureRec));//Note about the size
	KLTFreeFeatureList(inputList);
	//Append number of length new features to list
	for(int i=0;i<length;i++)
		moveFeature(newList,appendedList,originalLength+i,startIndex+i);
	return newList;
}

//Insert at the missed blanks of feature list
KLT_FeatureList ObjSeg::AppendNewFeaturesToList(KLT_FeatureList inputList,KLT_FeatureList appendedList,int startIndex,int length,const std::list<int> &missList)
{
	//Note that we assume that appendedList contains only valid features
	int p=startIndex;
	std::list<int>::const_iterator iter=missList.cbegin();
	while(p<startIndex+length&&iter!=missList.cend())
	{
		if(inputList->feature[*iter]->val<0)//only insert at invalid space
		{
			moveFeature(inputList,appendedList,*iter,p);
			p++;
		}
		iter++;
	}
	if(iter!=missList.cend()||iter==missList.cend()&&p==startIndex+length)//still exisits blanks
		return inputList;
	else//More to append, the returned list contains only valid features
	{
		return AppendNewFeaturesToList(inputList,appendedList,p,startIndex+length-p);
	}
}

//Move srcith feature in src into the dstith position in dst
void ObjSeg::moveFeature(KLT_FeatureList dst,KLT_FeatureList src,int dsti,int srci)
{
	dst->feature[dsti]->pos_x = src->feature[srci]->pos_x;
	dst->feature[dsti]->pos_y = src->feature[srci]->pos_y;
	dst->feature[dsti]->val = src->feature[srci]->val;
	dst->feature[dsti]->velocity[0] = src->feature[srci]->velocity[0];
	dst->feature[dsti]->velocity[1] = src->feature[srci]->velocity[1];
}