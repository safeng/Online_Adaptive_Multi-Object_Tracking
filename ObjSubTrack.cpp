#include "ObjSubTrack.h"

int ObjSubTrack::curTrackNO = 0;
#define LEARNING_RATE 0.5f
bool FeatureNodeCompare (const FeatureNode &a, const FeatureNode &b);

ObjSubTrack::ObjSubTrack(void):isValid(true),missing_count(0U),index(-1),total_prediction_count(0U),main_in_track(false),inactive(false)
{
	std::fill(color,color+3,0u);
	subTrackNO = IncreaseSubTrackCount();
}

ObjSubTrack::~ObjSubTrack(void)
{
	assert(!isValid);
}

/************************************************************************/
/* Increase subtrackindex by one                                        */
/************************************************************************/
int ObjSubTrack::IncreaseSubTrackCount()
{
	if(curTrackNO>INT_MAX) 
		curTrackNO = 1; 
	else 
		++curTrackNO; 
	return curTrackNO;
}

/*Release the resources*/
void ObjSubTrack::Release(const ObjSeg *objSeg)
{
	std::fill(color,color+3,0u);
	inactive = false;
	main_in_track = false;
	MotionFeature *motionFeature = objSeg->GetMotionFeature();
	for (std::list<int>::const_iterator citer = confidence_feat_list.cbegin(); citer!= confidence_feat_list.cend();++citer)
	{
		int feat_no = *citer;
		FeatureTrack track = motionFeature->featureSet[feat_no];
		if(track != NULL && track->indexOfObjTrack == this->subTrackNO) //exists && matches
		{
			track->indexOfObjTrack = -1;
		}
	}

	confidence_feat_list.clear();
	detectionSubTrack.clear();
	spatialFeatTracks.clear();
	missing_count = 0u; //set parameters invalid
	total_prediction_count = 0u;
	matched_len = 0u;
	index = -1;
	subTrackNO = -1;
	cur_weight = -1;
	appModel.Clear();
	//set not valid
	isValid = false;
}

/************************************************************************/
/* Test whether the detection is supported by feature tracks            */
/************************************************************************/
bool ObjSubTrack::TestFeatureSupport(const ObjSeg *objseg, int curFrameNO, const cv::Rect &detection) const
{
	if (spatialFeatTracks.size()<30u)
	{
		return true;
	}
	float vec[2];
	EstimateVec(vec,detection,curFrameNO);

	MotionFeature *motionFeature = objseg->GetMotionFeature();
	int count = 0;
	for (std::list<int>::const_iterator citer = spatialFeatTracks.cbegin(); citer != spatialFeatTracks.cend(); ++citer)
	{
		int x = *citer;
		FeatureTrack track = motionFeature->featureSet[x];
		float cur_diff = GetVelocityDifference(track->mean_vec,vec,this->detectionSubTrack.back().width);//velocity differences
		if (cur_diff<0.2f)
		{
			count++;
		}
	}
	
	float percent = count/float(spatialFeatTracks.size());
	if (percent>0.05f)
	{
		return true;
	}else
	{
		return false;
	}
}

/************************************************************************/
/* KLT Feature similarity                                               */
/************************************************************************/
float ObjSubTrack::KLTFeatureSimi(const ObjSeg *objseg, int curFrameNO, const cv::Rect &detection) const
{
	float vec[2];
	EstimateVec(vec,detection,curFrameNO);

	MotionFeature *motionFeature = objseg->GetMotionFeature();
	int count = 0;
	for (std::list<int>::const_iterator citer = spatialFeatTracks.cbegin(); citer != spatialFeatTracks.cend(); ++citer)
	{
		int x = *citer;
		FeatureTrack track = motionFeature->featureSet[x];
		float cur_diff = GetVelocityDifference(track->mean_vec,vec,this->detectionSubTrack.back().width);//velocity differences
		if (cur_diff<0.2f)
		{
			count++;
		}
	}

	if (spatialFeatTracks.size()<=5u)
	{
		return 1;
	}else
	{
		float percent = count/float(spatialFeatTracks.size());
		return percent;
	}
}

/********************************************************************************************/
/* Generate velocities from previous reliable information if current condition is unreliable*/
/********************************************************************************************/
void ObjSubTrack::generateVecFromReliableSources(int cur_frame_no)
{
	if (this->appModel.valid && this->appModel.tracker.tracking_lost)
	{
		predicted_model = detectionSubTrack.back(); //assume static 
	
		predicted_model.start_frame_no = cur_frame_no;
		return;
	}

	std::list<std::list<DetectionModel>::reverse_iterator> riter_stack;
	std::list<DetectionModel>::reverse_iterator riter;
	for (riter = detectionSubTrack.rbegin(); riter != detectionSubTrack.rend(); ++riter)
	{
		if (riter->supported)
		{
			riter_stack.push_back(riter);
			if (riter_stack.size() >= 4)
			{
				break;
			}
		}
	}

	if (riter == detectionSubTrack.rend() && riter_stack.empty())//empty stack
	{
		riter = detectionSubTrack.rbegin();//use previous node
		riter_stack.push_back(riter);
	}

	riter = riter_stack.front();//first supported

	//calculate the average velocity 
	float velocity[2] = {0};
	int size = riter_stack.size();

	while (!riter_stack.empty())
	{
		std::list<DetectionModel>::reverse_iterator elem = riter_stack.front();

		velocity[0] += elem->vec[0];
		velocity[1] += elem->vec[1];

		riter_stack.pop_front();
	}

	velocity[0] /= size;
	velocity[1] /= size;

	int frame_gap = cur_frame_no - riter->start_frame_no;
	float movement[] = {velocity[0] * frame_gap / (MO_GROUP-1), 
		velocity[1] * frame_gap / (MO_GROUP-1)};

	predicted_model.center_x = movement[0] + riter->center_x;
	predicted_model.center_y = movement[1] + riter->center_y;
	predicted_model.width = riter->width;
	predicted_model.height = riter->height;
	predicted_model.vec[0] = velocity[0];
	predicted_model.vec[1] = velocity[1];

	predicted_model.start_frame_no = cur_frame_no;
	predicted_model.supported = false;
	predicted_model.external_detection = false;
}

/*Model prediction. Selection of feature points which represent motion of this objsubtrack*/
void ObjSubTrack::UpdateProbability(const ObjSeg *objSeg)
{
	MotionFeature *motionFeature = objSeg->GetMotionFeature();
	this->cur_weight = -1; //change to an undefined value
	bool valid_anchor = GeneratePreVecByPre(motionFeature, objSeg->frameNO); //returns whether we can predict by anchor point

// 	if (this->appModel.valid && this->appModel.tracker.tracking_lost)
// 	{
// 		generateVecFromReliableSources(objSeg->frameNO);
// 		return;
// 	}

	//Final predicted vec is based on two parts (prediction from pre and feature tracks during current key frames)
	std::list<FeatureNode> feat_no_list;

	// Select all feature tracks that satisfy spatial-temporal and motional-temporal constraints
	ComputeVecDiff(motionFeature,objSeg->mappingArray,objSeg->frameNO,feat_no_list);
	bool feat_conf = feat_no_list.size() > 4;
	//record this feat_conf to object to indicate whether this model prediction is based on feature tracks
	bool predict_safe = true;
	
	int num_feat = 0;

	if(feat_conf || valid_anchor) // anchor points are available or feature points are worth selection
	{
		num_feat = SelectFeatureTrackMotionDescriptor(feat_no_list, motionFeature, objSeg->frameNO,false);
	}

	if(num_feat < 1)
	{
		generateVecFromReliableSources(objSeg->frameNO);
	}
}

/* Tracking without external detection */
bool ObjSubTrack::PredictModel(const ObjSeg *objSeg, const std::list<ImageRepresentation*> &frame_buffer, bool external_detection)
{
	if(external_detection)
	{
		assert(this->cur_weight==-1);
		//Increase continuous # of missing external detection 
		this->missing_count ++;
	}

	// Perform particle tracking on appearance
	if(this->appModel.valid)//to be safe
	{	
		//assert(this->main_in_track);
		return ParticleFilteringWithApp(objSeg, frame_buffer);
	}else
	{
		this->detectionSubTrack.push_back(predicted_model);
		assert(!predicted_model.supported && !predicted_model.external_detection);
		//no further update on velocity model
		if (external_detection)
		{
			//for model w/o app models. missing_count == total_prediction_count
			this->total_prediction_count ++;
		}
		return false;
	}
}

/*Particle filtering with appearance*/
bool ObjSubTrack::ParticleFilteringWithApp(const ObjSeg *objSeg, const std::list<ImageRepresentation*> &frame_buffer)
{
	assert(predicted_model.start_frame_no == frame_buffer.back()->frameNO);
	assert(detectionSubTrack.back().start_frame_no == frame_buffer.front()->frameNO-1);
	float simi_predicted = appModel.TestSimi(frame_buffer.back(),predicted_model.ConvertToRect(),true);
	std::cout << "simi_motion AppCheck" << simi_predicted<<std::endl;
	float simi_predict_color = appModel.TestHistSimi(frame_buffer.back(),predicted_model.ConvertToRect());
	std::cout << "simi_hist AppCheck" << simi_predict_color<<std::endl;
	sir_filter::State max_state;
	bool success = false;
	int valid_frame_no = -1;

	if (simi_predicted < 0.0f || simi_predict_color < 0.6f || 
		!TestMostSimilar(frame_buffer.back(),predicted_model.ConvertToRect(),simi_predict_color,0))
	{
		success = AppearanceCheck(objSeg, frame_buffer, max_state,valid_frame_no);
			
		if(success)
		{
			//sir_filter::State avg_state = this->appModel.GetObjState();
			predicted_model.SetFromState(max_state);
			
			predicted_model.supported = true;//not supported (can have some problem)
			this->total_prediction_count = 0;
			this->inactive = false;
			//Reset unsupported velocity in previous frames
			assert(predicted_model.start_frame_no == objSeg->frameNO);			
			setStatesFromEvidence();
			appModel.tracker.tracking_lost = false;

			MotionFeature *motionFeature = objSeg->GetMotionFeature();
			std::list<FeatureNode> feat_no_list;
			//selects features that conform the temporal spatial, motional constraints
			ComputeVecDiff(motionFeature,objSeg->mappingArray,objSeg->frameNO, feat_no_list);
			SelectFeatureTrackMotionDescriptor(feat_no_list, motionFeature, objSeg->frameNO, true);
			StoreToConfFeatList(feat_no_list,motionFeature);
		}
		else
		{
			// tracking lost
			// generateVecFromReliableSources(objSeg->frameNO);
			if (max_state.scale_x != 0.0f && max_state.scale_y != 0.0f)
			{
				assert(valid_frame_no!=-1);
				float vec[2];//change velocity
				EstimateVec(vec,max_state,valid_frame_no);
				predicted_model.vec[0] = vec[0];
				predicted_model.vec[1] = vec[1];
				predicted_model.SetFromState(max_state);
			}
			success = false;
			appModel.tracker.tracking_lost = true;
			predicted_model.supported = false;
			//accept default state predicted state
			this->total_prediction_count ++;
		}
	}else
	{	
		success = true;
		this->total_prediction_count = 0;
		this->inactive = false;
		assert(predicted_model.start_frame_no == objSeg->frameNO);
		appModel.InitParticleFilter(predicted_model.ConvertToRect()); //reinit particle filter
		predicted_model.supported = true;
		setStatesFromEvidence();//directly accept 
		appModel.tracker.tracking_lost = false;

		MotionFeature *motionFeature = objSeg->GetMotionFeature();
		std::list<FeatureNode> feat_no_list;
		//selects features that conform the temporal spatial, motional constraints
		ComputeVecDiff(motionFeature,objSeg->mappingArray,objSeg->frameNO, feat_no_list);
		SelectFeatureTrackMotionDescriptor(feat_no_list, motionFeature, objSeg->frameNO, true);
		StoreToConfFeatList(feat_no_list,motionFeature);
	}
	//Finally, we add model to the track list
	this->detectionSubTrack.push_back(predicted_model); 
	return success;
}

void ObjSubTrack::EstimateVec(float vec[2], const sir_filter::State &max_state, int cur_frame_no) const
{
	std::list<DetectionModel>::const_reverse_iterator riter; //the last supported node in track list
	for (riter = detectionSubTrack.crbegin(); riter!=detectionSubTrack.crend(); ++riter)
	{
		if (riter->supported)
		{
			break;
		}
	}

	if (riter==detectionSubTrack.crend())
	{
		riter--;//to the first node
	}

	int frame_gap = cur_frame_no-riter->start_frame_no;
	float movement[] = {max_state.pos_x - riter->center_x, max_state.pos_y - riter->center_y};
	vec[0] = movement[0] / (frame_gap / float(MO_GROUP - 1));
	vec[1] = movement[1] / (frame_gap / float(MO_GROUP - 1));
}

void ObjSubTrack::EstimateVec(float vec[2], const cv::Rect &evidence, int cur_frame_no) const
{
	std::list<DetectionModel>::const_reverse_iterator riter; //the last supported node in track list
	for (riter = detectionSubTrack.crbegin(); riter!=detectionSubTrack.crend(); ++riter)
	{
		if (riter->supported)
		{
			break;
		}
	}

	if (riter==detectionSubTrack.crend())
	{
		riter--;//to the first node
	}

	int frame_gap = cur_frame_no-riter->start_frame_no;
	float pos_x = evidence.x+evidence.width/2.0f;
	float pos_y = evidence.y+evidence.height/2.0f;

	float movement[] = {pos_x - riter->center_x, pos_y - riter->center_y};
	vec[0] = movement[0] / (frame_gap / float(MO_GROUP - 1));
	vec[1] = movement[1] / (frame_gap / float(MO_GROUP - 1));
}

/************************************************************************/
/* Search along the detection list and set velocity according to evidence*/
/************************************************************************/
void ObjSubTrack::setStatesFromEvidence()
{
	std::list<DetectionModel>::reverse_iterator riter; //the last supported node in track list
	for (riter = detectionSubTrack.rbegin(); riter!=detectionSubTrack.rend(); ++riter)
	{
		if (riter->supported)
		{
			break;
		}
	}

	bool first_node = false;
	if (riter == detectionSubTrack.rend())
	{
		--riter;//point to the first node
		first_node = true;
	}

	int frame_gap = predicted_model.start_frame_no - riter->start_frame_no;
	float movement[] = {predicted_model.center_x - riter->center_x, predicted_model.center_y - riter->center_y};
	float vec[] = {movement[0] / (frame_gap / float(MO_GROUP - 1)), movement[1] / (frame_gap / float(MO_GROUP -1))};
	float scale_change[] = {predicted_model.width - riter->width, predicted_model.height - riter->height};
	float vec_scale[] = {scale_change[0] / (frame_gap / float(MO_GROUP - 1)), scale_change[1] / (frame_gap / float(MO_GROUP -1))};
	//set velocity
	
	for (std::list<DetectionModel>::reverse_iterator r_set = detectionSubTrack.rbegin(); r_set != riter; r_set++)
	{
		r_set->vec[0] = vec[0];
		r_set->vec[1] = vec[1];
		int gap = predicted_model.start_frame_no - r_set->start_frame_no;
		float pos[] = {predicted_model.center_x - vec[0] * gap / (MO_GROUP-1), predicted_model.center_y -vec[1]*gap / (MO_GROUP-1)};
		r_set->center_x = pos[0];
		r_set->center_y = pos[1];
		float scale[] = {predicted_model.width - vec_scale[0] * gap / (MO_GROUP-1), 
			predicted_model.height - vec_scale[1] * gap / (MO_GROUP-1)};
		r_set->width = scale[0];
		r_set->height = scale[1];
	}

	if (first_node)
	{
		riter->vec[0] = vec[0];
		riter->vec[1] = vec[1];
	}
	
	predicted_model.vec[0] = vec[0];
	predicted_model.vec[1] = vec[1];
}

void ObjSubTrack::averageScale(cv::Rect &detection)
{
	std::list<std::list<DetectionModel>::reverse_iterator> riter_stack;
	std::list<DetectionModel>::reverse_iterator riter;
	float center[] = {detection.x + detection.width/2.0f, detection.y + detection.height/2.0f};

	for (riter = detectionSubTrack.rbegin(); riter != detectionSubTrack.rend(); ++riter)
	{
		if (riter->supported)
		{
			riter_stack.push_back(riter);
			if (riter_stack.size() >= 3)//take average
			{
				break;
			}
		}
	}

	float sum_width = detection.width;
	float sum_height = detection.height;
	int size = riter_stack.size()+1;
	while (!riter_stack.empty())
	{
		std::list<DetectionModel>::reverse_iterator elem = riter_stack.front();
		sum_width+=elem->width;
		sum_height+=elem->height;
		riter_stack.pop_front();
	}

	int width = int(sum_width/size + 0.5f);
	int height = int(sum_height/size + 0.5f);

	detection.width = width;
	detection.height = height;
	detection.x = int(center[0]-detection.width/2.0f + 0.5f);
	detection.y = int(center[1]-detection.height/2.0f + 0.5f);
}

/*Update model by a detection evidence*/
bool ObjSubTrack::UpdateModel(const ObjSeg *objSeg, const std::list<ImageRepresentation*> &frame_buffer, 
	const cv::Rect &detection)
{
	cv::Rect body_evidence = detection;
	averageScale(body_evidence); // take the average scale
	bool notTrain = false;

	predicted_model.supported = true; // is supported by some actual evidence
	predicted_model.external_detection = true;
	assert(predicted_model.start_frame_no == objSeg->frameNO);
	
	this->missing_count = 0;//clear missing_count to zero
	this->total_prediction_count = 0;
	this->inactive = false;
	this->matched_len ++; //one more external detection support

	if(this->main_in_track && this->appModel.valid)
	{
		// SIR Particle filtering
		// we do not change velocity here
		sir_filter::State max_state;
		this->appModel.DetectionCheck(predicted_model.vec, body_evidence, frame_buffer, objSeg->frameNO, predicted_model.ConvertToRect(), max_state);
		//sir_filter::State avg_state_resample = this->appModel.GetObjState();
		//Set predicted_model
		predicted_model.SetFromDetection(body_evidence);
		appModel.tracker.tracking_lost = false;

	}else // no appearance model
	{
		// directly copy external detection result to predicted_model
		predicted_model.SetFromDetection(body_evidence);
	}
	
	//Set velocity
	setStatesFromEvidence();
	//update instant velocity
	MotionFeature *motionFeature = objSeg->GetMotionFeature();
	std::list<FeatureNode> feat_no_list;
	//selects features that conform the temporal spatial, motional constraints
	ComputeVecDiff(motionFeature,objSeg->mappingArray,objSeg->frameNO, feat_no_list);
	SelectFeatureTrackMotionDescriptor(feat_no_list, motionFeature, objSeg->frameNO,true);
	
	StoreToConfFeatList(feat_no_list,motionFeature);

	predicted_model.supported = true; // is supported by some actual evidence
	predicted_model.external_detection = true;

	//Finally, we add model to the track list
	this->detectionSubTrack.push_back(predicted_model);
	return true;
}

/*substitute original track with a new sub obj model from detection*/
void ObjSubTrack::ReplaceWithNewDetection(const ObjSeg *objSeg, const cv::Rect &evidence)
{
	this->missing_count = 0u;
	this->matched_len = 1u;
	this->color[0] = rand()%256;
	this->color[1] = rand()%256;
	this->color[2] = rand()%256;

	//Generate new nodes and set instant vec
	assert(this->detectionSubTrack.empty());
	//Generate new vec from all points in its bounding box
	MotionFeature *motionFeature = objSeg->GetMotionFeature();
	int numFeatures = motionFeature->numberOfFeatureTracks;
	std::list<int> feat_no_list;
	float ROI[4];

	ROI[0] = evidence.x;
	ROI[1] = evidence.y;
	ROI[2] = evidence.x + evidence.width;
	ROI[3] = evidence.y + evidence.height;
	
	for(int i = 0; i<numFeatures; ++i)
	{
		int x = objSeg->mappingArray[i];
		FeatureTrack track = motionFeature->featureSet[x];
		if(track->length < MO_GROUP)//Too short tracks
		{
			continue;
		}
		else
		{
			float pos_x = track->feature[track->length-1]->pos_x;//pos at current frame
			float pos_y = track->feature[track->length-1]->pos_y;
			if(pos_x >= ROI[0] && pos_x <= ROI[2] && pos_y >= ROI[1] && pos_y <= ROI[3])
			{
				feat_no_list.push_back(x);
			}
		}
	}

	float sum_w = 0;
	float sum_vecx = 0, sum_vecy = 0, sum_inst_vecx = 0, sum_inst_vecy = 0;
	//Compute the average vec and instant vec prepared for further prediction
	static const float longer_weight = 2.5f;
	static const float shorter_weight = 1.0f;

	for(std::list<int>::const_iterator citer = feat_no_list.cbegin(); citer != feat_no_list.cend(); ++citer)
	{
		int featureNO = (*citer);
		int length = motionFeature->featureSet[featureNO]->length;
		if(length >= NON_KEYFRAME_GAP)//longer feature
		{
			sum_w += longer_weight;
			sum_vecx += motionFeature->featureSet[featureNO]->mean_vec[0]*longer_weight;//weightesd summation
			sum_vecy += motionFeature->featureSet[featureNO]->mean_vec[1]*longer_weight;
			sum_inst_vecx += motionFeature->featureSet[featureNO]->feature[length-1]->velocity[0]*longer_weight;
			sum_inst_vecy += motionFeature->featureSet[featureNO]->feature[length-1]->velocity[1]*longer_weight;
		}else//shorter features have less weight
		{
			sum_w += shorter_weight;
			sum_vecx += motionFeature->featureSet[featureNO]->mean_vec[0]*shorter_weight;
			sum_vecy += motionFeature->featureSet[featureNO]->mean_vec[1]*shorter_weight;
			sum_inst_vecx += motionFeature->featureSet[featureNO]->feature[length-1]->velocity[0]*shorter_weight;
			sum_inst_vecy += motionFeature->featureSet[featureNO]->feature[length-1]->velocity[1]*shorter_weight;
		}

		//store to confidence list
		this->confidence_feat_list.push_back(featureNO);
		motionFeature->featureSet[featureNO]->indexOfObjTrack = this->subTrackNO;
		motionFeature->featureSet[featureNO]->num_matches = 1;
	}

	float vec[] = {sum_vecx/sum_w,sum_vecy/sum_w};//average vec during KEY_FRAME
	float inst_vec[] = {sum_inst_vecx/sum_w,sum_inst_vecy/sum_w};
	predicted_model.vec[0] = vec[0];//update instant vec
	predicted_model.vec[1] = vec[1];
	
	DetectionModel new_model;
	new_model.SetFromDetection(evidence);
	new_model.vec[0] = vec[0];
	new_model.vec[1] = vec[1];
	new_model.start_frame_no = objSeg->frameNO;

	new_model.supported = true;
	new_model.external_detection = true;//external detection
	this->detectionSubTrack.push_back(new_model);
}

/*Generate vec prediction just from pre info (nearest vec information)*/
bool ObjSubTrack::GeneratePreVecByPre(const MotionFeature *motionFeature,int frameNO)
{
	static const float th_dist_static = 0.1f;//th for static obj
	static const size_t min_size = 4u;
	bool valid_vec = false;

	// have anchoring feature points
	if(!this->confidence_feat_list.empty())
	{
		size_t len_track = this->detectionSubTrack.size();
		int num_selected = this->confidence_feat_list.size();
		size_t pre_size = this->confidence_feat_list.size();

		if(num_selected > min_size)//can be used
		{
			float th = th_dist_static;
			std::list<int> selected;
			
			float ROI[4];
			const DetectionModel &pre_model = this->detectionSubTrack.back();
			//Predict with previous knowledge (instant vec. constant vec)
			int frame_gap = frameNO - pre_model.start_frame_no;
			float frameElapsed = frame_gap / float(MO_GROUP-1);
			float cur_movement[] = {pre_model.vec[0] * frameElapsed, pre_model.vec[1] * frameElapsed};
			float predicted_location[] = {pre_model.center_x + cur_movement[0], pre_model.center_y + cur_movement[1]};//Predicted with instant vec

			ROI[0] = predicted_location[0] - pre_model.width; //predicted ROI
			ROI[1] = predicted_location[1] - pre_model.height;
			ROI[2] = predicted_location[0] + pre_model.width;
			ROI[3] = predicted_location[1] + pre_model.height;//region of interest. (cut off region)

			for(std::list<int>::iterator iter = this->confidence_feat_list.begin(); 
				iter!=this->confidence_feat_list.end();)
			{
				FeatureTrack track = motionFeature->featureSet[*iter];
				if(track != NULL && track->indexOfObjTrack == this->subTrackNO) //exists && matches
				{
					//check spatial conformity
					float pos_x = track->feature[track->length-1]->pos_x;
					float pos_y = track->feature[track->length-1]->pos_y;
					if(!(pos_x >= ROI[0] && pos_x <= ROI[2] && pos_y >= ROI[1] && pos_y <= ROI[3]))
					{
						track->indexOfObjTrack = -1;//reset to -1
						iter = this->confidence_feat_list.erase(iter); //delete when temporal spatial conformity is violated
					}else
					{
						float dist = MotionConformity(track,frameNO,predicted_model.vec); //Use the constant velocity model
						if(dist<=th)
						{
							selected.push_back(*iter);
						}
						++iter;
					}
				}else
				{
					iter = this->confidence_feat_list.erase(iter);
				}
			}

			if(selected.size() >= min_size)//existing feature tracks remain valid
			{
				int sum_w = 0;
				float sum_vecx = 0, sum_vecy = 0, sum_inst_vecx = 0, sum_inst_vecy = 0;
				// use selected features for anchoring
				for(std::list<int>::const_iterator citer = selected.cbegin();citer!=selected.cend(); ++citer)
				{
					int featureNO = *citer;
					FeatureTrack track = motionFeature->featureSet[featureNO];
					int length = track->length;
					if(length >= 3)
					{
						sum_w += track->num_matches * length;
						sum_vecx += track->mean_vec[0] * track->num_matches * length;//weighted summation
						sum_vecy += track->mean_vec[1] * track->num_matches * length;
						sum_inst_vecx += track->feature[length-1]->velocity[0] * track->num_matches * length;
						sum_inst_vecy += track->feature[length-1]->velocity[1] * track->num_matches * length;
					}
				}

				assert(sum_w!=0);
				valid_vec = true;
				float vec[] = {sum_vecx/sum_w,sum_vecy/sum_w}; 
				float inst_vec[] = {sum_inst_vecx/sum_w,sum_inst_vecy/sum_w};
				predicted_model.vec[0] = vec[0]; //update instant vec
				predicted_model.vec[1] = vec[1];
			}
		}
	}

	return valid_vec;//returns whether we have obtained a valid vec. DO NOT ADD EXTRA MODEL
}

/* When external detection is available, learn confident features and store them to confident list. (Learn prior for feature descriptors) */
void ObjSubTrack::StoreToConfFeatList(const std::list<FeatureNode> &feat_no_list, const MotionFeature *motionFeature)
{
	// Select candidates from set of feature tracks that conform temporal-spatial and temporal-motional constraints
	for(std::list<FeatureNode>::const_iterator citer = feat_no_list.cbegin(); citer!=feat_no_list.cend(); ++citer)
	{
		float dist = citer->dist;
		if(dist <= 0.1f)
		{
			int feat_no = citer->featureNO;
			FeatureTrack track = motionFeature->featureSet[feat_no];
			if(track->indexOfObjTrack == this->subTrackNO)//already has it
			{
				track->num_matches++;
			}else//newly added
			{
				this->confidence_feat_list.push_back(feat_no);
				track->indexOfObjTrack = this->subTrackNO;//store the index
				track->num_matches = 1;
			}
		}
	}
}

/*Select feature tracks as motion descriptors*/
int ObjSubTrack::SelectFeatureTrackMotionDescriptor(std::list<FeatureNode> &feat_no_list, const MotionFeature *motionFeature, int frame_no, bool instant)
{
	int num_selected = feat_no_list.size();
	// rank candidate feature tracks according to temporal motional affinity
	feat_no_list.sort(FeatureNodeCompare); 

	float sum_w = 0;
	float sum_vecx = 0, sum_vecy = 0, sum_inst_vecx = 0, sum_inst_vecy = 0;
	//Compute the average vec and instant vec prepared for further prediction

	const float longer_weight = 2.5f;
	const float shorter_weight = 1.0f;

	//top num_selected candidates
	std::list<FeatureNode>::const_iterator citer = feat_no_list.cbegin();
	int i;
	for(i = 0; i<num_selected; ++i, ++citer)
	{
		const FeatureNode &node = (*citer);
		if(node.dist >= 0.1f) //must within some th
		{
			break;
		}
		int featureNO = node.featureNO;
		FeatureTrack track = motionFeature->featureSet[featureNO];
		int length = track->length;
		int factor = 1; //weight factor
		if(track->indexOfObjTrack == this->subTrackNO)
		{
			factor = track->num_matches + 1;
		}
		if(length >= NON_KEYFRAME_GAP)//longer feature
		{
			float w = longer_weight * factor;//more weight on features already on the target
			sum_w += w;
			sum_vecx += track->mean_vec[0] * w;//weighted summation
			sum_vecy += track->mean_vec[1] * w;
			sum_inst_vecx += track->feature[length-1]->velocity[0] * w;
			sum_inst_vecy += track->feature[length-1]->velocity[1] * w;
		}else//shorter features have less weight
		{
			sum_w += shorter_weight;
			sum_vecx += track->mean_vec[0] * shorter_weight;
			sum_vecy += track->mean_vec[1] * shorter_weight;
			sum_inst_vecx += track->feature[length-1]->velocity[0] * shorter_weight;
			sum_inst_vecy += track->feature[length-1]->velocity[1] * shorter_weight;
		}
	}


	float vec[2], inst_vec[2];
	if(i>=1)
	{
		vec[0] = sum_vecx/sum_w;
		vec[1] = sum_vecy/sum_w;
		inst_vec[0] = sum_inst_vecx/sum_w;
		inst_vec[1] = sum_inst_vecy/sum_w;

		if (instant)
		{
			predicted_model.vec[0] = vec[0];
			predicted_model.vec[1] = vec[1];
			return i;
		}

	}else
	{
		return i;
	}
	
	float vec_diff = GetVelocityDifference(vec,predicted_model.vec,1.0f);

	if(vec_diff <= 2.0f)//within the distance (anchor strategy)
	{
		predicted_model.vec[0] = vec[0]; //update instant vec
		predicted_model.vec[1] = vec[1];
	}

	//Use vec from anchor point
	int frame_gap = frame_no - this->detectionSubTrack.back().start_frame_no;

	float frameElapsed = frame_gap/(float)(MO_GROUP-1);
	float cur_movement[] = {predicted_model.vec[0] * frameElapsed, predicted_model.vec[1] * frameElapsed};

	const DetectionModel &pre_model = this->detectionSubTrack.back();
	float predicted_location[] = {pre_model.center_x + cur_movement[0], pre_model.center_y + cur_movement[1]};
			
	//Set hypothesized model predicted by motion model
	predicted_model.center_x = predicted_location[0];
	predicted_model.center_y = predicted_location[1];
	//inherit scale from previous response
	predicted_model.width = pre_model.width;
	predicted_model.height = pre_model.height;
	//velocity determined by weighted voting from selected feature tracks
	predicted_model.start_frame_no = frame_no;
	// currently no external evidence or appearance evidence is used to ensure this assumption
	predicted_model.supported = false;
	predicted_model.external_detection = false;

	return i;
}

/*Object Track - Feature Tracks Motion affinity. By addressing temporal-spatial and temporal-motional similarity*/
void ObjSubTrack::ComputeVecDiff(const MotionFeature *motionFeature, const int mappingArray[], int frameNO, std::list<FeatureNode> &featureNO_list)
{
	// Find feature tracks that conform temporal-spatial constraints
	FindSpatialCandidateFeatures(frameNO, motionFeature, mappingArray);

	for(std::list<int>::const_iterator citer = spatialFeatTracks.cbegin(); citer != spatialFeatTracks.cend(); ++citer)
	{
		int x = (*citer);
		FeatureTrack track = motionFeature->featureSet[x];
		
		float distToTrack = MotionConformity(track, frameNO, predicted_model.vec);
		FeatureNode node;
		node.featureNO = x;
		node.dist = distToTrack;
		featureNO_list.push_back(node);
	}
}

/*Find candidate feature tracks that conform the spatial constraint and store them*/
void ObjSubTrack::FindSpatialCandidateFeatures(int cur_frame_no, const MotionFeature *motionFeature, const int mappingArray[])
{
	spatialFeatTracks.clear();

	int numFeatures = motionFeature->numberOfFeatureTracks;
	for(int i = 0; i < numFeatures; ++i)
	{
		int x = mappingArray[i];
		FeatureTrack track = motionFeature->featureSet[x];
		int tail_indx = track->length - 1;
	
		if(track->length >= MIN_CONSISTENT_LEN)
		{
			//check spatial-temporal conformity
			bool spatial_pass = false;
			//check from the most recent
			for(std::list<DetectionModel>::const_reverse_iterator criter = detectionSubTrack.crbegin(); criter != detectionSubTrack.crend(); ++criter)
			{
				const DetectionModel &model = *criter;
				int model_frame_no = model.start_frame_no;
				int gap = cur_frame_no - model_frame_no;
				if(tail_indx - gap < 0)
					break;
				float x = track->feature[tail_indx - gap]->pos_x;
				float y = track->feature[tail_indx - gap]->pos_y;
				float model_ROI[4];
				if (model.supported)
				{
					model_ROI[0] = model.center_x - model.width/2;
					model_ROI[1] = model.center_y - model.height/2;
					model_ROI[2] = model.center_x + model.width/2;
					model_ROI[3] = model.center_y + model.height/2;
				} 
				else
				{
					continue;
				}
				
				if(!(x >= model_ROI[0] && x <= model_ROI[2] && y >= model_ROI[1] && y <= model_ROI[3]))
				{
					spatial_pass = false;
					break;
				}else
				{
					spatial_pass = true;
				}
			}

			if(spatial_pass) // pass all spatial requirement
			{
				spatialFeatTracks.push_back(x);	
			}
		}
	}
}

float ObjSubTrack::VelocityDifferenceNoScaling(const float vec_1[2], const float vec_2[2]) const
{
	float diff = (stdfunc::pow2(vec_1[0] - vec_2[0]) + stdfunc::pow2(vec_1[1] - vec_2[1]));
	return diff;
}

/*Calculate velocity difference between two instant velocities*/
float ObjSubTrack::GetVelocityDifference(const float vec_1[2], const float vec_2[2], float scale) const
{
	float mag1 = vec_1[0] * vec_1[0] + vec_1[1] * vec_1[1];
	float mag2 = vec_2[0] * vec_2[0] + vec_2[1] * vec_2[1];
	float max_mag, min_mag;
	if(mag1 < mag2)
	{
		min_mag = mag1;
		max_mag = mag2;
	}else
	{
		min_mag = mag2;
		max_mag = mag1;
	}

	if(max_mag < 1.0f) // both small
	{
		min_mag = 1.0f;
	}

	if (min_mag > 4.0f)
	{
		min_mag = 4.0f;
	}

	float diff = (stdfunc::pow2(vec_1[0] - vec_2[0]) + stdfunc::pow2(vec_1[1] - vec_2[1])) / ((min_mag + 0.001f) * scale); // normalize by scale
	return diff;
}

/* Temporal motion affinity between feature track and objsubtrack */
float ObjSubTrack::MotionConformity(const FeatureTrack track, int frameNO, const float most_cur_vec[2]) const
{
	assert(frameNO == track->startFrame + track->length-1);

	// check most current vec
	float feat_vec[] = {track->feature[track->length-1]->velocity[0], track->feature[track->length-1]->velocity[1]}; // vec of last frame
	float cur_diff = GetVelocityDifference(feat_vec,most_cur_vec,this->detectionSubTrack.back().width);
	
	int tail_indx = track->length - 1;
	std::list<float> motion_diff; //stack to store motion difference

	//check from the most recent detection
	for(std::list<DetectionModel>::const_reverse_iterator criter = this->detectionSubTrack.crbegin(); criter != this->detectionSubTrack.crend(); ++criter)
	{
		const DetectionModel &model = *criter;
		int start_frame_no = model.start_frame_no;
		int gap = frameNO - start_frame_no;
		
		if(tail_indx - gap < 0)
		{
			break;
		}

		if (!model.supported)//only consider supported part
		{
			continue;
		}

		float diff = GetVelocityDifference(track->feature[tail_indx - gap]->velocity,model.vec,model.width);
		motion_diff.push_back(diff);
	}

	// add weight to each stage
	float sum_diff;
	if(!motion_diff.empty())
	{
		sum_diff = motion_diff.back();
		motion_diff.pop_back();
	}else
	{
		sum_diff = cur_diff;
	}

	while(!motion_diff.empty())
	{
		float diff = motion_diff.back();
		sum_diff = sum_diff * LEARNING_RATE + diff * (1 - LEARNING_RATE);
		motion_diff.pop_back();
	}
	
	float distance = (sum_diff + cur_diff)/2;
	return distance;
}

/*Test model at the back of objSubTrack*/
bool ObjSubTrack::TestValidity()
{
	const cv::Rect frameRect(0,0,image_width,image_height);
	const DetectionModel &model = this->predicted_model;
	float center_x = model.center_x;
	float center_y = model.center_y;
	
	cv::Rect predicted_rect = model.ConvertToRect();
	if ((predicted_rect & frameRect).area() < predicted_rect.area())
	{
		if (appModel.valid)
		{
			appModel.Clear();
		}
	}

	if(!model.supported && 
		(center_x<=0 || center_y<=0 || center_x>image_width || center_y>image_height))
	{
		return false;
	}else
		return true;
}

/*Compute aff with another track*/
bool ObjSubTrack::ComputeAffWithObjSubTrack(const ObjSubTrack *track) const
{
	assert(this->main_in_track && track->isValid);
	
	if(track->matched_len <= 3 || track->matched_len <= track->missing_count)
		return false;
	
	//test overlap
	cv::Rect model = this->detectionSubTrack.back().ConvertToRect();
	cv::Rect test = track->detectionSubTrack.back().ConvertToRect();
	cv::Rect intersect = model & test;
	if(intersect.area() / float(model.area()) >0.33f 
		|| intersect.area() / float(test.area()) > 0.33f)
		return false;

	if(track->matched_len >= 4 && track->cur_weight > 0)//Gain enough evidence & is supported at current frame
		return true;
	else
		return false;
}

bool ObjSubTrack::TestAddModel(const cv::Rect &evidence) const
{
	if (!this->predicted_model.supported) // only for supported  
	{
		return false;
	}
	cv::Rect full_body_model = this->predicted_model.ConvertToRect();
	cv::Rect intersect = evidence & full_body_model;

	float percent_model = intersect.area() / float(full_body_model.area());
	float percent_evidence = intersect.area() / float(evidence.area());

	//significant spatial overlap
	if(percent_model >= 0.6f || percent_evidence >= 0.6f)
	{
		return true;
	}

	return false;
}

/************************************************************************/
/* Add negative list for further judgment                              */
/************************************************************************/
void ObjSubTrack::AddToNegList(const std::list<ObjSubTrack *> &negTrackList)
{
	this->negTrackList.clear();
	this->negTrackList.assign(negTrackList.cbegin(),negTrackList.cend());
}

/************************************************************************/
/*Test whether the associated detection or tracking result is the most similar of all*/
/************************************************************************/
bool ObjSubTrack::TestMostSimilar(ImageRepresentation *image, const cv::Rect &detection, float histSimi, float gap) const
{
	cv::Mat ROI = image->frame(detection);
	float hist[AppFeature::BIN_SIZE*AppFeature::BIN_SIZE*AppFeature::BIN_SIZE];

	AppFeature::extractHistFeature(hist,ROI);

	for(std::list<ObjSubTrack*>::const_iterator iter = negTrackList.cbegin(); iter != negTrackList.cend(); ++iter)
	{
		ObjSubTrack *track = *iter;
		assert(track!=this);
		if (track->appModel.valid)
		{
			float simi = track->appModel.evalHist(hist);
			if (simi>histSimi+gap)
			{
				return false;
			}
		}
	}

	return true;
}


/*Compute the affinity value (conforming spatial, appearance, scale conformity)*/
void ObjSubTrack::ComputeAffinityValue(const cv::Rect &detection, float &scaleProb, float &posProb)
{
	float mean[] = {0,0};
	//In scale
	//float mag_missing = sqrtf(this->missing_count+1);

	// Use predicted state from motion for association
	float model_scale[] = {predicted_model.width, predicted_model.height};
	float evidence_scale[] = {detection.width, detection.height};

	//take min scale
	float min_width = model_scale[0] <= evidence_scale[0] ? model_scale[0] : evidence_scale[0];
	float min_height = model_scale[1] <= evidence_scale[1] ? model_scale[1] : evidence_scale[1];

	//sigma should be proportional to missing count
	float sigma_scale[] = {1.0f, 1.0f};
	float dist_scale[] = {fabs(model_scale[0]-evidence_scale[0])/min_width,fabs(model_scale[1]-evidence_scale[1])/min_width};
	float proScale = stdfunc::GaussianDistribution(mean,sigma_scale,2,dist_scale,false);

	//In position. Should be normalized by scale
	float model_pos[] = {predicted_model.center_x, predicted_model.center_y};
	float evidence_pos[] = {detection.x + detection.width/2.0f, detection.y + detection.height/2.0f};
	float sigma_pos[] = {1.0f, 1.0f};
	float dist[] = {fabs(model_pos[0]-evidence_pos[0])/min_width, fabs(model_pos[1]-evidence_pos[1])/min_height};
	
	float proPos = stdfunc::GaussianDistribution(mean,sigma_pos,2,dist,false);
	scaleProb = proScale;
	posProb = proPos;

}

void ObjSubTrack::PaintPariticleStates(cv::Mat &img) const
{
	if (appModel.valid)
	{
		appModel.particle_filter.PaintParticles(img,appModel.tracker.searchRegion,appModel.tracker.tracking_lost);
	}
}

/************************************************************************/
/* Supervised training in consecutive frames                            */
/************************************************************************/
void ObjSubTrack::SupervisedTraining(const std::list<ImageRepresentation *> &framePool)
{
	assert(this->appModel.valid);
	cv::Rect curModel = detectionSubTrack.back().ConvertToRect();
	std::list<DetectionModel>::reverse_iterator riter = detectionSubTrack.rbegin();
	++riter;
	assert(riter!=detectionSubTrack.crend());
	cv::Rect preModel = (*riter).ConvertToRect();
	int size = framePool.size();
	float movement[] = {(curModel.x - preModel.x)/float(size), (curModel.y - preModel.y)/float(size)};
	float scale[] = {(curModel.width - preModel.width)/float(size), (curModel.height - preModel.height)/float(size)};
	int i = 1;
	for (std::list<ImageRepresentation*>::const_iterator citer = framePool.cbegin(); citer!=framePool.cend(); ++citer, ++i)
	{
		int x = int(preModel.x + movement[0]*i + 0.5f);
		int y = int(preModel.y + movement[1]*i + 0.5f);
		int width = int(preModel.width + scale[0]*i + 0.5f);
		int height = int(preModel.height + scale[1]*i + 0.5f);
		cv::Rect model(x,y,width,height);
		appModel.SuperviseTraining(*citer,model);//supervised training
	}
}

/************************************************************************/
/* Semi-supervised training in consecutive frames                       */
/************************************************************************/
void ObjSubTrack::SemiSupervisedTraining(const std::list<ImageRepresentation *> &framePool)
{
	assert(this->appModel.valid);
	cv::Rect curModel = detectionSubTrack.back().ConvertToRect();
	std::list<DetectionModel>::reverse_iterator riter = detectionSubTrack.rbegin();
	++riter;
	assert(riter != detectionSubTrack.crend());
	cv::Rect preModel = (*riter).ConvertToRect();
	int size = framePool.size();
	float movement[] = {(curModel.x - preModel.x)/float(size), (curModel.y - preModel.y)/float(size)};
	float scale[] = {(curModel.width - preModel.width)/float(size), (curModel.height - preModel.height)/float(size)};
	int i = 1;
	for (std::list<ImageRepresentation*>::const_iterator citer = framePool.cbegin(); citer!=framePool.cend(); ++citer, ++i)
	{
		int x = int(preModel.x + movement[0]*i + 0.5f);
		int y = int(preModel.y + movement[1]*i + 0.5f);
		int width = int(preModel.width + scale[0]*i + 0.5f);
		int height = int(preModel.height + scale[1]*i + 0.5f);
		cv::Rect model(x,y,width,height);
		appModel.SemiSupervisedTraining(*citer,model);//supervised training
	}
}

bool ObjSubTrack::AppearanceCheck(const ObjSeg *objseg, std::list<ImageRepresentation*> frame_buffer, sir_filter::State &max_state, int &valid_frame_no)
{
	cv::Rect predict_bound = predicted_model.ConvertToRect();
	float scale_det = predict_bound.width / float(AppFeature::MIN_WIDTH);
	std::vector<float> weight;

	float cur_vec[] = {0, 0};//frame by frame static tracking
	float th = 0.0f;

	bool appSuccess = false;
	for(std::list<ImageRepresentation*>::iterator iter = frame_buffer.begin(); iter != frame_buffer.end(); ++iter)
	{
		if (appModel.tracker.tracking_lost)//track by exhaustive searching
		{
			cv::Rect max_rect;
			std::vector<float> posWeight;
			std::vector<cv::Rect> posRect;
			float minMargin = -0.2f;
			bool success = appModel.tracker.TrackSearchRegion(*iter,predict_bound,minMargin,max_rect,posRect,posWeight);
			float max_pos_weight = 0;
			float simi_color = 0;
			float max_weight = 0;
			//calculate joint weights
			for (size_t i = 0; i<posRect.size(); ++i)
			{
				float weight_haar = posWeight[i];
				float weight_color = appModel.TestHistSimi(*iter,posRect[i]);
				float weight_feat = KLTFeatureSimi(objseg,(*iter)->frameNO,posRect[i]);
				posWeight[i] = weight_haar * weight_color * weight_feat;
				if (posWeight[i]>max_pos_weight)
				{
					max_pos_weight = posWeight[i];
					simi_color = weight_color;
					max_weight = weight_haar;
					max_rect = posRect[i];
				}
			}

			appModel.particle_filter.DrawPosRects((*iter)->frame,posWeight,posRect);

			if (simi_color > 0.5f && 
				TestMostSimilar(*iter,max_rect,simi_color,0) 
				&& TestFeatureSupport(objseg,(*iter)->frameNO, max_rect)) //whether supported by features
			{
				appModel.tracker.tracking_lost = false;

				max_state = appModel.InitParticleFilter(max_rect); //Reinit particle filter
				valid_frame_no = (*iter)->frameNO;
				appSuccess = true;
			}else
			{
				float sig_pos = 20 * scale_det;//since we already init. Do not need sigma
				float sig_scale = 0;
				appModel.particle_filter.DrawSamplesFromStateTransition(cur_vec, 1, sig_pos, sig_scale);
				appSuccess = false;
				//return false;
			}

		}else//track by particle filtering
		{
			float minMargin = -0.2f;
			// normal frame by frame sir particle filtering
			std::vector<sir_filter::State> new_state = appModel.particle_filter.GetParticleStates();
			weight.clear();
			sir_filter::State temp_max_state;
			float max_weight = appModel.CalWeightForParticle(cv::Rect(),(*iter),new_state,weight,temp_max_state);
			float simi_color = appModel.TestHistSimi((*iter),temp_max_state.ConvertToRect());
			std::vector<sir_filter::State> posStates;
			std::vector<float> posWeight;
			appModel.particle_filter.GetPosStates(weight,minMargin,posStates,posWeight);//get pos states & weights
			
			float max_pos_weight = 0;
			//calculate joint weight
			for (size_t i = 0;i<posWeight.size();++i)
			{
				float weight_haar = posWeight[i];
				cv::Rect detection = posStates[i].ConvertToRect();
				float weight_color = appModel.TestHistSimi((*iter),detection);
				float weight_feat = KLTFeatureSimi(objseg,(*iter)->frameNO,detection);
				posWeight[i] = weight_haar * weight_color *weight_feat;

				if (posWeight[i]>max_pos_weight)
				{
					max_pos_weight = posWeight[i];
					temp_max_state = posStates[i];//record max state
					simi_color = weight_color;
					max_weight = weight_haar;
				}
			}

			appModel.particle_filter.DrawPosParticles((*iter)->frame,posWeight,posStates);

			if(simi_color > 0.4f 
				//&& TestMostSimilar(*iter,temp_max_state.ConvertToRect(),simi_color,0.1f) 
				&&TestFeatureSupport(objseg,(*iter)->frameNO,temp_max_state.ConvertToRect()))
			{
				valid_frame_no = (*iter)->frameNO;
				max_state = temp_max_state;
				appModel.particle_filter.SetMeasurementLikelihood(weight);
				appModel.particle_filter.SIRParticleFilter(); //perform rest of particle filtering

				float sig_pos = 10 * scale_det;//since we already init. Do not need sigma
				float sig_scale = 0;
				appModel.particle_filter.DrawSamplesFromStateTransition(cur_vec, 1, sig_pos, sig_scale);
				appSuccess = true;
			}else
			{
				float sig_pos = 20 * scale_det;//since we already init. Do not need sigma
				float sig_scale = 0;
				appModel.particle_filter.DrawSamplesFromStateTransition(cur_vec, 1, sig_pos, sig_scale);
				appSuccess = false;
				//return false;
			}
		}
	}

	return appSuccess;//the last result
}

DetectionModel::DetectionModel(float center_x, float center_y, float width, float height, const float vec[2], int start_frame_no)://By default it inherents vec from previous frame
	center_x(center_x),center_y(center_y), width(width), height(height),start_frame_no(start_frame_no),supported(false)
{
	this->vec[0] = vec[0]; this->vec[1] = vec[1];
}

DetectionModel::DetectionModel()
{
	start_frame_no = -1;
	supported = false;
}

/*Set state from state variable*/
void DetectionModel::SetFromState(const sir_filter::State &avg_state)
{
	this->center_x = avg_state.pos_x;
	this->center_y = avg_state.pos_y;
	this->width = avg_state.scale_x;
	this->height = avg_state.scale_y;
}

/*Set state from detection result*/
void DetectionModel::SetFromDetection(const cv::Rect &detection)
{
	float center_x = detection.x + detection.width/2.0f;
	float center_y = detection.y + detection.height/2.0f;
	this->center_x = center_x;
	this->center_y = center_y;
	this->width = detection.width;
	this->height = detection.height;
}

bool FeatureNodeCompare (const FeatureNode &a, const FeatureNode &b)
{
	return a.dist < b.dist;
}