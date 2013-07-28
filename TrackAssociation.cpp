#include "TrackAssociation.h"

bool MyCompareFunc(const AffinityVal &a1, const AffinityVal &a2);
TrackAssociation::TrackAssociation()
{
	objSeg = new ObjSeg;
}

TrackAssociation::~TrackAssociation(void)
{
}

/*Update Model by Prediction. Preprocessing of models*/
void TrackAssociation::UpdateModelsByPrediction()
{
	for(std::vector<ObjTrack*>::iterator iter = trackVector.begin(); iter != trackVector.end();)
	{
		ObjTrack *track = (*iter);

		//Predict using the estimated dynamics
		if(!track->UpdateEachModel(objSeg))//remove this objtrack
		{
			delete track;
			iter = trackVector.erase(iter);
		}else
		{
			++iter;
		}
	}

	//Refresh subvector
	int pos = 0;
	for(std::vector<ObjSubTrack*>::iterator iter = this->subTrackVector.begin(); iter != this->subTrackVector.end();)
	{
		ObjSubTrack *subTrack = (*iter);
		if(!subTrack->isValid)
		{
			//deallocate space 
			delete subTrack;
			iter = this->subTrackVector.erase(iter);
		}else
		{
			subTrack->index = pos;//Update the position
			//set possible search region for tracking lost tracks
			if (subTrack->appModel.valid && subTrack->appModel.tracker.tracking_lost)
			{
				cv::Rect searchRegion = subTrack->appModel.particle_filter.GetSearchRegion();
				subTrack->appModel.tracker.SetSearchRegion(searchRegion,subTrack->detectionSubTrack.back().ConvertToRect());
			}
			++pos;
			++iter;
		}
	}
}

/*Organize the whole process of object tracking*/
void TrackAssociation::ObjectTrackingAtKeyFrame(int frameNO, cv::Mat &image, bool external_detection)
{
	std::vector<cv::Rect> evidence;
	std::vector<float> evidence_weight;

	//Predict using feature tracks motion descriptors
	UpdateModelsByPrediction();

	if(external_detection)
	{
		PeopleDetection(frameNO, image, evidence, evidence_weight);
		//If external detection is available, perform Data Association by greedy method
		std::vector<AffinityVal> affinity_mat;
		CalculateAffinityScore(evidence,affinity_mat);
		std::vector<int> assign_match;
		std::vector<bool> evidence_used, model_used;
		FindMatch(affinity_mat,evidence,assign_match,evidence_used,model_used);
		//Tracking by particle filtering. If appearance model not available, normal method is applied
		std::list<ObjSubTrack*> new_track_list; // Some sub object track candidates may be declared as new obj tracks
		std::list<float> new_weight_list;
		//Update obj tracks with external detection (particle filtering)
		UpdateModel(objSeg,evidence,evidence_weight,assign_match,evidence_used,new_track_list,new_weight_list);
		//For evidence that neither matched or added, create new models for them
		CreateNewObjTracks(objSeg,evidence,evidence_weight,evidence_used,new_track_list,new_weight_list);
		//Clean sub track models
		CleanSubTrackVector();
		//Init appearance models for all main sub tracks
		InitAppearanceModelForMainTracks();

	}else
	{
		TrackWithoutExternalDetection(objSeg);
	}

	SuperviseTrainingForSubTracks();
	DrawDetectionModels(image,frameNO);
}

/************************************************************************/
/* supervised training for objsubTracks if no overlapping found         */
/************************************************************************/
void TrackAssociation::SuperviseTrainingForSubTracks()
{
	std::list<ObjSubTrack*> valid_track;
	for (size_t i = 0; i < subTrackVector.size(); ++i)
	{
		ObjSubTrack *subTrack = subTrackVector[i];
		if (subTrack->appModel.valid)
		{
			valid_track.push_back(subTrack);
		}
	}

	for (size_t i = 0; i<subTrackVector.size(); ++i)
	{
		ObjSubTrack *subTrack = subTrackVector[i];

		if (subTrack->appModel.valid)
		{
			const DetectionModel &model1 = subTrack->detectionSubTrack.back();
			const cv::Rect r1 = model1.ConvertToRect();

			Rect r1_rect;
			r1_rect = r1;
			Rect negative_region_rect = subTrack->appModel.tracker.getTrackingROI(r1_rect,7);
			cv::Rect negative_region = negative_region_rect.ConvertToRect();

			bool notTrain = false;
			std::list<ObjSubTrack*> negative_list; //other objTracks should be discriminated

			if (subTrack->appModel.tracker.tracking_lost)
			{
				notTrain = true;
// 				for (std::list<ObjSubTrack *>::iterator iter = valid_track.begin(); iter!=valid_track.end(); ++iter)
// 				{//push all other than itself
// 					if ((*iter)!=subTrack)
// 					{
// 						negative_list.push_back(*iter);
// 					}
// 				}
			}

			//collect neg tracks
			for (std::list<ObjSubTrack *>::iterator iter = valid_track.begin(); iter!=valid_track.end(); ++iter)
			{
				if ((*iter) != subTrack)
				{
					const DetectionModel &model2 = (*iter)->detectionSubTrack.back();
					const cv::Rect r2 = model2.ConvertToRect();

					cv::Rect intersect = r1 & r2;
					float overlap = intersect.area()/float(r1.area());
						
					if (overlap > 0.1f)
					{
						notTrain = true;
					}

					if ((*iter)->appModel.tracker.tracking_lost)
					{
						negative_list.push_back(*iter);
					}else
					{
						if ((r2 & negative_region).area()>0)
						{
							negative_list.push_back(*iter);
						}
					}
				}
			}

			//positive samples
			if (!notTrain)
			{
				if (subTrack->detectionSubTrack.back().external_detection)
				{
					subTrack->SupervisedTraining(frame_pool);
				}else
				{
					subTrack->SemiSupervisedTraining(frame_pool);
				}
			}

			subTrack->AddToNegList(negative_list);//Add to negative set (construct Nearest Neighbor Classifier)
		}
	}
}

/*Track obj sub tracks only with appearance model*/
void TrackAssociation::TrackWithoutExternalDetection(const ObjSeg *objSeg)
{
	// In this part. We check how many times the default constant velocity model is used continuously
	// If above some level. We mark the track as inactive
	std::list<ImageRepresentation*> frame_p_list;
	ConstructFramePoolPointer(frame_p_list);
	
	for(std::vector<ObjTrack*>::iterator iter_track = this->trackVector.begin(); iter_track != this->trackVector.end(); ++iter_track)
	{
		ObjTrack *obj_track = *iter_track;
		assert(obj_track->isValidStorage);
		int count = 0;
		for(std::list<ObjSubTrack*>::iterator iter_sub_track = obj_track->detection_model.begin(); iter_sub_track != obj_track->detection_model.end(); ++iter_sub_track)
		{
			ObjSubTrack *sub_track = *iter_sub_track;
			assert(sub_track->isValid);
			
			if(sub_track->main_in_track)
				++count;

			// Track use particle filtering with appearance model
			sub_track->PredictModel(objSeg, frame_p_list, false); 

// 			// Track with no supporting evidence for some time
// 			if(sub_track->total_prediction_count >= 8)
// 			{
// 				sub_track->inactive = true;	
// 			}
		}
		assert(count <= 1);
	}
}

/*Init appearance models for new main tracks and cancel appearance models non-main tracks*/
void TrackAssociation::InitAppearanceModelForMainTracks()
{
	cv::Rect frameRect(0,0,image_width,image_height);

	for(std::vector<ObjSubTrack*>::iterator iter = this->subTrackVector.begin(); iter != this->subTrackVector.end(); ++iter)
	{
		ObjSubTrack *sub_track = *iter;
		const DetectionModel &model = sub_track->detectionSubTrack.back();
		assert(sub_track->isValid);
		if(sub_track->main_in_track && sub_track->matched_len>=1)
		{
			if(!sub_track->appModel.valid)
			{
				//train app model
				sub_track->appModel.SuperviseTraining(this->frame_pool.back(), model.ConvertToRect());
			}
		}else
		{
			if(sub_track->appModel.valid)
			{
				//cancel app model for non major tracks
				sub_track->appModel.Clear();
			}
		}
	}
}

/*Create new objTrack and objSubTrack for each unused evidence*/
void TrackAssociation::CreateNewObjTracks(const ObjSeg *objSeg, const std::vector<cv::Rect> &evidence, const std::vector<float> &evidence_weight, const std::vector<bool> &evidence_used, 
	const std::list<ObjSubTrack*> &new_track_list, const std::list<float> &new_weight_list)
{
	std::list<ObjSubTrack*> list_temp;
	std::list<float> weight_temp;
	std::list<ImageRepresentation*> frame_p_list;
	ConstructFramePoolPointer(frame_p_list);

	//Add obj tracks from external detection evidence
	for(size_t i = 0; i<evidence_used.size(); ++i)
	{
		if(!evidence_used[i]) //One evidence can be only used once
		{
			//Generate a new obj track
			ObjTrack *track = new ObjTrack;
			track->CreateWithNewEvidence(objSeg,subTrackVector,evidence[i],evidence_weight[i]);
			this->trackVector.push_back(track);//add to track vector
		}
	}

	//add splitted obj tracks
	assert(new_track_list.size() == new_weight_list.size());
	std::list<ObjSubTrack*>::const_iterator iter_track = new_track_list.begin();
	std::list<float>::const_iterator iter_weight = new_weight_list.begin();

	for(;iter_track!=new_track_list.end(); ++iter_track, ++iter_weight)
	{
		ObjTrack *track = new ObjTrack;
		//Init an obj track with an existing obj subtrack
		track->CreateWithObjSubTrack(*iter_track,*iter_weight);
		this->trackVector.push_back(track);
	}
}

/*Update existing model and create submodels to handle false positives*/
void TrackAssociation::UpdateModel(const ObjSeg *objSeg, const std::vector<cv::Rect> &evidence, const std::vector<float> &weight_evidence, const std::vector<int> &assign_match, std::vector<bool> &evidence_used,
	std::list<ObjSubTrack*> &new_track_list, std::list<float> &new_weight_list)
{
	size_t subVecSize = this->subTrackVector.size();
	std::list<ImageRepresentation*> frame_p_list;
	ConstructFramePoolPointer(frame_p_list);

	//For external matches from detection
	for(size_t i = 0; i < assign_match.size(); ++i)
	{
		int index_subTrack = assign_match[i];
		if(index_subTrack != -1)
		{
			assert(index_subTrack < this->subTrackVector.size());
			ObjSubTrack *subTrack = this->subTrackVector[index_subTrack];
			assert(subTrack->isValid);
			//Tracking with external detection
			if(subTrack->UpdateModel(objSeg,frame_p_list, evidence[i]))
				subTrack->cur_weight = weight_evidence[i]; //change weight	
		}
	}

	//Track some objects with appearance model & add possible sub-models
	for(std::vector<ObjTrack*>::iterator iter = trackVector.begin(); iter!=trackVector.end();)
	{
		ObjTrack *track = (*iter);
		if(!track->ProduceResult(objSeg, frame_p_list, evidence,weight_evidence,evidence_used,subTrackVector, new_track_list, new_weight_list))//remove this objtrack
		{
			delete track;
			iter = trackVector.erase(iter);
		}else
		{
			++iter;
		}
	}
}

/*calculate spatial and appearance conformity for data association*/
void TrackAssociation::CalculateAffinityScore(const std::vector<cv::Rect> &evidence, std::vector<AffinityVal> &affinity_mat)
{
	affinity_mat.clear();
	size_t model_size = subTrackVector.size(); //# of sub objects (One object can have many sub tracks)
	size_t evidence_size = evidence.size();
	affinity_mat.reserve(model_size * evidence_size);
	//Calculation
	for(size_t i = 0;i < model_size; ++i)
	{
		ObjSubTrack *subTrack = subTrackVector[i]; //each sub object
		assert(subTrack->isValid);
		
		for(size_t j = 0; j < evidence_size; ++j)
		{
			cv::Rect rect = evidence[j]; //each external detection
			size_t index = j * model_size + i;
			float probScale = 0, probPos = 0;
			subTrack->ComputeAffinityValue(rect,probScale,probPos);
			AffinityVal aff;
			aff.index = index;
			aff.probPos = probPos;
			aff.probScale = probScale;
			affinity_mat.push_back(aff);
		}
	}
}

/*Greedy method for data association & false positive models*/
void TrackAssociation::FindMatch(std::vector<AffinityVal> &affinity_mat, const std::vector<cv::Rect> &evidence, std::vector<int> &assign_match, std::vector<bool> &evidence_used, std::vector<bool> &model_used)
{
	const float default_th = 0.25f;
	//match threshold
	const float match_pos = 0.3f;
	const float match_scale = 0.7f;
	const float match_app = 0.52f; // range from [0,1]
	//w/o app model
	const float match_pos_no_app = 0.8f;
	const float match_scale_no_app = 0.9f;

	size_t model_size = subTrackVector.size(); // for each sub object
	size_t evidence_size = evidence.size(); // external detection

	assign_match.assign(evidence_size, -1); // assign list is for each detection evidence

	model_used.assign(model_size,false);
	evidence_used.assign(evidence_size,false);
	
	for(size_t i = 0; i < affinity_mat.size(); ++i)
	{
		AffinityVal &aff = affinity_mat[i];
		float probScale = aff.probScale;
		float probPos = aff.probPos;
		int index = aff.index;
		size_t n = index/model_size;//index of evidence
		size_t m = index-n*model_size;//index of model

		ObjSubTrack *subTrack = subTrackVector[m];

		size_t missing_count = subTrack->missing_count;
		float missing_discount = 1;
		float missing_discount_scale = 1;
		// upper limit for relaxation
		if(missing_count >= 4)
		{
			missing_count = 4;
			missing_discount = logf(1+missing_count/2.0f)+1;
		}

		float scale_th;
		float pos_th;
		if (subTrack->appModel.valid)
		{
			scale_th = match_scale;
			pos_th = match_pos;
		}else
		{//for tracks without appearance model
			scale_th = match_scale_no_app;
			pos_th = match_pos_no_app;

			if (missing_count >= 1)
			{
				missing_discount = 1.09f;
			}
		}

		bool space_scale_match;
		
		//for tracked lost objs
		if (subTrack->appModel.valid && subTrack->appModel.tracker.tracking_lost)
		{
			//loose positional threshold
			bool pos_fall;
			pos_fall = ((subTrack->appModel.tracker.searchRegion & evidence[n]).area()
				/ float(evidence[n].area())) > 0.0f;
			
			space_scale_match = pos_fall && probScale >= scale_th;
		}
		else
		{
			space_scale_match = probScale >= scale_th && probPos>= pos_th/missing_discount;
		}
		
		if(space_scale_match)
		{
			bool featureSupport = subTrack->TestFeatureSupport(objSeg,objSeg->frameNO,evidence[n]);
			if (featureSupport)
			{
				if(subTrack->appModel.valid)
				{
					float probApp = subTrack->appModel.TestHistSimi(this->frame_pool.back(),evidence[n]);
					float probHarr = subTrack->appModel.TestSimi(this->frame_pool.back(),evidence[n],true);
					probHarr = (probHarr + 1)/2;// to [0,1]
					aff.proApp = probApp + probHarr;
				}else //no valid appearance 
				{
					aff.proApp = default_th; //default value
				}
			}else
			{
				aff.proApp = 0;
			}
		}else
		{
			aff.proApp = 0;
		}
	}

	//sort by appearance values
	std::sort(affinity_mat.begin(),affinity_mat.end(),MyCompareFunc);

	for (size_t i = 0; i < affinity_mat.size(); ++i)
	{
		AffinityVal &aff = affinity_mat[i];
		int index = aff.index;
		int n = index/model_size;//index of evidence
		int m = index-n*model_size;//index of model
		float appVal = aff.proApp;
		if (appVal >= default_th)//above the threshold
		{
			if (!model_used[m] && !evidence_used[n])//both unused
			{
				assign_match[n] = m;
				model_used[m] = true;
				evidence_used[n] = true;
			}
		}
	}
}

/*Test whether 'detection' comes from foreground. Returns true if it is. False otherwise.*/
bool TrackAssociation::TestForegroundDetection(const cv::Rect &detection, float th)
{
	//static const float foreground_th = 0.4f; //th percentage for a detection considered as foreground
	int num_pixels = detection.width*detection.height;
	int num_valid_th=(int)(num_pixels*th);//# of valid pixels th
	int num_invalid_th = num_pixels-num_valid_th;
	int num_valid = 0;//# of foreground pixels
	int num_invalid = 0;//# of background pixels
	for(int i = detection.x; i<detection.x+detection.width; ++i)//test all pixels in detection window
	{
		for(int j = detection.y; j<detection.y+detection.height; ++j)
		{
			if(this->objSeg->tc->DetermineForeground(j,i)) //foreground
			{
				++num_valid;
			}else //background
			{
				++num_invalid;
			}
		}
	}
	float percent = num_valid/(float)num_pixels;
	if(num_valid >= num_valid_th)
		return true;
	if(num_invalid >= num_invalid_th)
		return false;
	return false;
}

/*Draw detection sub models in one image. Note that all detection submodels are in global coordinates*/
void TrackAssociation::DrawDetectionModels(const cv::Mat &frame, int frameNO)
{
	cv::Mat image;
	frame.copyTo(image);
	//static const uchar color_crop[] = {0,255,255};

	for(std::vector<ObjTrack*>::const_iterator citer = trackVector.cbegin(); citer != trackVector.cend(); ++citer)
	{
		const ObjTrack *track = *citer;
		
		for(std::list<ObjSubTrack*>::const_iterator citer_track = track->detection_model.cbegin(); citer_track!= track->detection_model.cend(); ++citer_track)
		{
			if((*citer_track)->main_in_track) //main track
			{

				//if(!(*citer_track)->inactive)
				//if((*citer_track)->matched_len >= GROUP_FRAMES_HOLD && !(*citer_track)->inactive) //must be active tracks
				{	
					DrawRectangle(image,(*citer_track)->detectionSubTrack.back().ConvertToRect(),(*citer_track)->color);
					//DrawRectangle(image,AppFeature::CropDetection((*citer_track)->detectionSubTrack.back().ConvertToRect()),color_crop);

				}
				//(*citer_track)->PaintPariticleStates(image);
			}
		}
	}
	char file_name[50];
	sprintf(file_name,"trackResult/track%i.jpg",frameNO);
	cv::imwrite(file_name,image);
}

/*Detect people across the whole screen*/
void TrackAssociation::PeopleDetection(int frameNO, const cv::Mat &image, std::vector<cv::Rect> &evidence, std::vector<float> &weight_evidence)
{
	std::vector<cv::Rect> full_body_found;
	bool resizeImg = true;
	if (resizeImg)
	{
		cv::Size newSize(1280,960);
		cv::Mat resize_img;
		cv::resize(image,resize_img,newSize);
		this->humanDetector.DetectHuman(resize_img,full_body_found);//detect at resized image
		//map to original size
		float ratio_w = image.cols/float(newSize.width);
		float ratio_h = image.rows/float(newSize.height);
		for (size_t i = 0; i<full_body_found.size(); ++i)
		{
			cv::Rect &rect = full_body_found[i];
			float ratio_left = rect.x/float(resize_img.cols);//ratio top left
			float ratio_upper = rect.y/float(resize_img.rows);
			rect.x = int(ratio_left*image.cols + 0.5f);
			rect.y = int(ratio_upper*image.rows + 0.5f);
			rect.width = int(ratio_w*rect.width + 0.5f);
			rect.height = int(ratio_h*rect.height + 0.5f);
		}
	}else
	{
		this->humanDetector.DetectHuman(image,full_body_found);
	}

	std::vector<double> full_body_weight;
	full_body_weight.assign(full_body_found.size(),5);

	const static float fore_body_th = 0.3f; //Foreground th for full_body

	std::vector<cv::Rect> detection_list_full_body; //The list of all full_body detection
	std::vector<double> weight_list_full_body; //weight for each full body detection
	
	//First pass -- Test foreground
	std::vector<cv::Rect>::const_iterator citer_body = full_body_found.cbegin();
	std::vector<double>::const_iterator citer_body_weight = full_body_weight.cbegin();

	//reserve space	
	detection_list_full_body.reserve(full_body_found.size());
	weight_list_full_body.reserve(full_body_found.size());

	
	//body test
	for(;citer_body!=full_body_found.cend();++citer_body,++citer_body_weight)
	{
		cv::Rect detection = (*citer_body);
		bool foreground = true;
		if(this->objSeg->tc->CheckBgEnabled())
		{
			foreground = TestForegroundDetection(detection,fore_body_th);
		}
		if(foreground && detection.area() < 23100)//trick
		{
			detection_list_full_body.push_back(detection);
			weight_list_full_body.push_back((*citer_body_weight));
		}
	}

 	evidence.assign(detection_list_full_body.cbegin(), detection_list_full_body.cend());
 	weight_evidence.assign(weight_list_full_body.cbegin(),weight_list_full_body.cend());

#if	1//Draw raw detection result
	cv::Mat img2Write;
	image.copyTo(img2Write);
	char imageFileName[30];
	sprintf(imageFileName,"detect%d.jpg",frameNO);//One prediction, one result
	for (int i = 0; i<evidence.size(); ++i)
	{
		cv::rectangle(img2Write,evidence[i],cv::Scalar(255,0,0),1);
	}
	cv::imwrite(imageFileName,img2Write);
#endif

}

void TrackAssociation::CleanSubTrackVector()
{
	//Refresh subTrackVector
	int pos = 0;
	for(std::vector<ObjSubTrack*>::iterator iter = this->subTrackVector.begin(); iter != this->subTrackVector.end();)
	{
		ObjSubTrack *subTrack = (*iter);
		if(!subTrack->isValid)
		{
			delete subTrack;
			iter = this->subTrackVector.erase(iter);
		}else
		{
			if(subTrack->subTrackNO == -1) //assign an unique ID number
				subTrack->subTrackNO = ObjSubTrack::IncreaseSubTrackCount();
			subTrack->index = pos;//Update the position
			++pos;
			++iter;
		}
	}
}

/*Store frames to buffer*/
void TrackAssociation::StoreToFramePool(const cv::Mat &cur_frame, int frameNO)
{
	uchar *grayImg = SemiBoostingTracker::getGrayImage(cur_frame);
	ImageRepresentation *image = new ImageRepresentation(grayImg,Size(image_height,image_width));
	image->frameNO = frameNO;
	cur_frame.copyTo(image->frame);//store frame
	this->frame_pool.push_back(image);
	delete grayImg;
}

/*Clear frame pool for next phase's appearance checking*/
void TrackAssociation::ClearFramePool()
{
	for(std::list<ImageRepresentation*>::iterator iter = frame_pool.begin(); iter!=frame_pool.end(); ++iter)
	{
		ImageRepresentation *image = *iter;
		delete image;
	}
	this->frame_pool.clear();
}

/*Construct frame pointers for each frame in the pool*/
void TrackAssociation::ConstructFramePoolPointer(std::list<ImageRepresentation*> &frame_p_list)
{
	frame_p_list.clear();
	frame_p_list.assign(frame_pool.cbegin(),frame_pool.cend());
}

bool MyCompareFunc(const AffinityVal &a1, const AffinityVal &a2)
{
	float prob1 = a1.proApp * a1.probPos;
	float prob2 = a2.proApp * a2.probPos;
	if (prob1>prob2)
	{
		return true;
	}else if (prob1<prob2)
	{
		return false;
	}else //equal
	{
		float prob_loc1 = a1.probScale * a1.probPos;
		float prob_loc2 = a2.probScale * a2.probPos;
		return prob_loc1 > prob_loc2;
	}
}