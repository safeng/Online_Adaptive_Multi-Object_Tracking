#include "ObjTrack.h"

const float ObjTrack::init_th(0.5f);
const float ObjTrack::valid_th(0.8f);
const int ObjTrack::upper_limit(3);

ObjTrack::ObjTrack(void):isValidStorage(true)
{
	
}

ObjTrack::~ObjTrack(void)
{
	isValidStorage = false;
}

void ObjTrack::NormalizeWeight()
{
	float sum_weight = 0;
	norm_weight.clear();//first clear the normalized weight
	for(std::list<float>::const_iterator citer = weight.cbegin(); citer!=weight.cend(); ++citer)
	{
		sum_weight+=*citer;
	}

	for(std::list<float>::const_iterator citer = weight.cbegin(); citer!=weight.cend(); ++citer)
	{
		norm_weight.push_back((*citer)/sum_weight);
	}
}

/*Create new sub model for unmatched evidence*/
void ObjTrack::CreateNewSubModel(const ObjSeg *objSeg, std::vector<ObjSubTrack*> &subTrackVector, const cv::Rect &evidence, float evidence_weight)
{
	static const float init_weight = 1.0f; //initial weight
	//check whether the maximun allowable # of sub model has reached th
	if(detection_model.size() >= upper_limit)
	{
		float min_weight = FLT_MAX;
		ObjSubTrack *min_track = NULL;
		int min_index = -1;
		//remove the submodel with the smallest weight
		std::list<ObjSubTrack*>::iterator iter_track = detection_model.begin();
		std::list<float>::iterator iter_weight = weight.begin();

		for(int j = 0; iter_track!=detection_model.end(); ++iter_track,++iter_weight,++j)
		{
			if((*iter_weight) < min_weight)
			{
				min_weight = *iter_weight;
				min_track = *iter_track;
				min_index = j;
			}
		}

		//proportional to (1 + original weight of detection)
		assert(min_index >= 0 && min_index<detection_model.size());
		iter_weight = weight.begin();
		std::advance(iter_weight,min_index);
		(*iter_weight) = init_weight * (1 + evidence_weight); //(we must make sure that the newly added detection will not be further removed)
		//Remove the track and Generate a new one
		assert(min_track != NULL && min_track->isValid);

		min_track->Release(objSeg); //Release
		min_track->ReplaceWithNewDetection(objSeg, evidence);
		
		
	}else
	{
		//just append to list
		ObjSubTrack *newTrack = new ObjSubTrack;
		newTrack->ReplaceWithNewDetection(objSeg,evidence);//create new model and set appearance hist
		weight.push_back(init_weight * (1+evidence_weight));
		
		assert(newTrack->isValid && newTrack->missing_count ==0 && newTrack->subTrackNO>0);
		detection_model.push_back(newTrack); //add to model
		subTrackVector.push_back(newTrack); //add to global objsubtrack vector
	}
}

/*Sort the measurement to find best sub model to represent the object*/
ObjSubTrack* ObjTrack::SortWeight()
{
	NormalizeWeight();
	assert(norm_weight.size() == detection_model.size());
	
	//Find the one with max. weight
	std::list<float>::const_iterator citer_weight = norm_weight.cbegin();
	std::list<ObjSubTrack*>::iterator iter_track = detection_model.begin();
	float max_weight = 0;
	ObjSubTrack *main_track = NULL; // record the main track
	for(;citer_weight!=norm_weight.cend(); ++citer_weight, ++iter_track)
	{
		float weight = (*citer_weight) * (*iter_track)->matched_len;
		if(weight > max_weight)
		{
			max_weight = weight;
			main_track = *iter_track;
		}
		//set all sub tracks as false
		(*iter_track)->main_in_track = false;
	}

	if(main_track != NULL)
	{
		//Judge the qualification of main track
 		if((main_track->matched_len >= 2))
 		{
 			main_track->main_in_track = true;
 		}else
 		{
 			main_track = NULL;
 		}
	}
	
	return main_track;
}

/*Init an obj track with an existing sub obj track*/
void ObjTrack::CreateWithObjSubTrack(ObjSubTrack *sub_track, float weight)
{
	assert(sub_track->isValid && sub_track->subTrackNO > 0 && !sub_track->main_in_track);
	assert(this->detection_model.empty() && this->weight.empty());
	this->detection_model.push_back(sub_track);
	this->weight.push_back(weight);
	SortWeight();
}

/*Update each submodel within it. Possible to generate new models*/
bool ObjTrack::ProduceResult(const ObjSeg *objSeg, const std::list<ImageRepresentation*> &frame_buffer, const std::vector<cv::Rect> &evidence, const std::vector<float> &evidence_weight,
	std::vector<bool> &evdence_used, std::vector<ObjSubTrack*> &subTrackVector, std::list<ObjSubTrack*> &new_track_list, std::list<float> &new_weight_list)
{
	//update matched submodels
	{
		std::list<float>::iterator iter_weight = weight.begin();

		for(std::list<ObjSubTrack*>::iterator iter_track= detection_model.begin(); iter_track!=detection_model.end();)
		{
			ObjSubTrack *subTrack = (*iter_track);

			assert(subTrack->isValid);

			if(subTrack -> cur_weight == -1) // no match
			{
				if((*iter_weight) < 0)
				{
					subTrack->Release(objSeg);
					 

					iter_track = detection_model.erase(iter_track);
					iter_weight = weight.erase(iter_weight);
				}else
				{
					if (!subTrack -> PredictModel(objSeg,frame_buffer,true))
					{
						if ((*iter_track)->appModel.valid)
						{
							(*iter_weight) -= 1;
						}else
						{
							(*iter_weight) -= 3;
						}
					}	
					++iter_weight;
					++iter_track;
				}
			}else//update (have been done outside this function)
			{
				(*iter_weight) += 1 + subTrack->cur_weight; //Increase by cur_weight
				++iter_weight;
				++iter_track;
			}
		} 
	}
	
	//Add possible overlapping submodels
	for (std::list<ObjSubTrack*>::iterator iter_track = detection_model.begin();iter_track!=detection_model.end();++iter_track)
	{
		for (size_t i = 0; i<evidence.size();++i)
		{
			if(!evdence_used[i])//for unused evidence
			{
				if((*iter_track)->TestAddModel(evidence[i]))//can be added as submodel
				{
					CreateNewSubModel(objSeg,subTrackVector,evidence[i],evidence_weight[i]);
					evdence_used[i] = true;//mark as used
				}
			}
		}
	}
	
	if(this->detection_model.size()==0)
	{
		return false;
	}

	//Sort weight. Get main track
	ObjSubTrack *main_track = SortWeight();

	if(main_track != NULL)
	{
		//Possible Split
		std::list<ObjSubTrack*>::iterator iter_track = detection_model.begin();
		std::list<float>::iterator iter_weight = weight.begin();
	
		for(;iter_track!=this->detection_model.end();)
		{
			if((*iter_track)==main_track)
			{
				++iter_track;
				++iter_weight;
			}else if(main_track->ComputeAffWithObjSubTrack(*iter_track)) //can split
			{
				new_track_list.push_back(*iter_track); // Create new obj track with this obj sub track
				new_weight_list.push_back(*iter_weight); // Also inherit weight
				iter_track = detection_model.erase(iter_track);
				iter_weight = weight.erase(iter_weight);
			}else
			{
				++iter_track;
				++iter_weight;
			}
		}
	}
	return true;
}

void ObjTrack::CreateWithNewEvidence(const ObjSeg *objSeg, std::vector<ObjSubTrack*> &subTrackVector, const cv::Rect &evidence, float weight)
{
	CreateNewSubModel(objSeg,subTrackVector,evidence,weight);
	SortWeight();
}

/* Update each model*/
bool ObjTrack::UpdateEachModel(const ObjSeg *objSeg)
{
	assert(detection_model.size() == weight.size());

	std::list<ObjSubTrack*>::iterator iter_track = detection_model.begin();
	std::list<float>::iterator iter_weight = weight.begin();

	bool remove = false;
	for(;iter_track != detection_model.end();)
	{
		ObjSubTrack *track = *iter_track; //for each objSubTrack
		track->UpdateProbability(objSeg);//Predict by dynamic based feature tracks & appearance tracking 
		if(!track->TestValidity())//should be removed
		{
			remove = true;
			track->Release(objSeg);
			/*push_free_objSubTrack(spaceAlloc,track);*/
			iter_track = detection_model.erase(iter_track);
			iter_weight = weight.erase(iter_weight);
		}else
		{
			++iter_weight;
			++iter_track;
		}
	}
	
	/*
	//Merge objSubTracks that have high overlapping in this frame
	std::list<ObjSubTrack*> objlist;
	std::list<float> weightlist;

	while(!this->detection_model.empty())
	{
		ObjSubTrack *track = this->detection_model.front();
		float weight_model = this->weight.front();
		this->detection_model.pop_front();//remove front
		this->weight.pop_front();

		cv::Rect model = track->detectionSubTrack.back().ConvertToRect();
		float max_weight = weight_model;
		ObjSubTrack *track_best = track;
		iter_track= detection_model.begin();
		iter_weight = weight.begin();

		for(;iter_track!=detection_model.end();)
		{
			ObjSubTrack *track_inner = (*iter_track);
			float weight_inner = (*iter_weight);
			cv::Rect model_inner = track_inner->detectionSubTrack.back().ConvertToRect();
			float overlap = GetOverLapPercentage(model,model_inner);
			if(overlap>0.33f)
			{
				if(weight_inner>max_weight)
				{
					max_weight = weight_inner;
					//original best track should be released
					track_best->Release();
					push_free_objSubTrack(spaceAlloc,track_best);
					track_best = track_inner;
					remove = true;
				}else
				{
					track_inner->Release();
					push_free_objSubTrack(spaceAlloc,track_inner);
					remove = true;
				}

				iter_track = this->detection_model.erase(iter_track);
				iter_weight = this->weight.erase(iter_weight);
			}else
			{
				++iter_track;
				++iter_weight;
			}
		}

		assert(track_best->isValid);
		objlist.push_back(track_best);
		weightlist.push_back(max_weight);
	}

	this->detection_model.assign(objlist.begin(),objlist.end());
	this->weight.assign(weightlist.begin(),weightlist.end());
	*/

	//If after deletion no objSubTrack, remove this objtrack
	if(detection_model.size()==0)
		return false;
	else
	{
		if(remove) // some of obj sub tracks are removed
			this->SortWeight();
		return true;
	}
}

/*Remove ins from this model*/
void ObjTrack::RemoveIns(const ObjSubTrack *ins)
{
	bool find = false;
	std::list<ObjSubTrack*>::iterator iter_track= detection_model.begin();
	std::list<float>::iterator iter_weight = weight.begin();
	for(;iter_track!=detection_model.end();)
	{
		ObjSubTrack *track = (*iter_track);
		if(track == ins)
		{
			iter_track = detection_model.erase(iter_track);
			iter_weight = weight.erase(iter_weight);
			find = true;
			break;
		}else
		{
			++iter_weight;
			++iter_track;
		}
	}
	assert(find);
}