#include "AppFeature.h"


static bool TestOutlier(const cv::Rect &bound);

AppFeature::AppFeature():valid(false)
{
	std::fill(colorHist,colorHist + BIN_SIZE*BIN_SIZE*BIN_SIZE,0);
}

AppFeature::~AppFeature(void)
{
	Clear();
}


void AppFeature::SuperviseTraining(ImageRepresentation *image, const cv::Rect &innerbound)
{
	if (!TestOutlier(innerbound))
	{
		return;
	}
	cv::Rect new_bound = CropDetection(innerbound);

	Size validROI(image_height,image_width);
	if (!valid)
	{
		
		int iteration = tracker.SupervisedTraining(image,new_bound,validROI,true);
		if (tracker.available)//tracker become available
		{
			InitParticleFilter(innerbound); //Init particle filter
			accumulateToHist(image,innerbound); //Generate histogram description
			valid = true;
		}
	}else
	{
		int iteration = tracker.SupervisedTraining(image,new_bound,validROI,false);
		accumulateToHist(image,innerbound);
	}
}

void AppFeature::SemiSupervisedTraining(ImageRepresentation *image, const cv::Rect &innerbound)
{
	if (!TestOutlier(innerbound))
	{
		return;
	}
	cv::Rect new_bound = CropDetection(innerbound);

	tracker.SemiSupervisedTraining(image,new_bound);
}

/*Clear all instances*/
void AppFeature::Clear()
{
	valid = false;
	std::fill(colorHist,colorHist+BIN_SIZE*BIN_SIZE*BIN_SIZE,0);

	for (int i = 0;i<BIN_SIZE*BIN_SIZE*BIN_SIZE;++i)
	{
		m_pos[i].setDefault();//set to default value
	}
	tracker.Clear();
}

void AppFeature::accumulateToHist(ImageRepresentation *image, const cv::Rect &bound)
{
	cv::Mat ROI = image->frame(bound);
	float hist[BIN_SIZE*BIN_SIZE*BIN_SIZE];

	extractHistFeature(hist,ROI);
	if (!valid)
	{
		for (int i =0;i<BIN_SIZE*BIN_SIZE*BIN_SIZE;++i)
		{
			m_pos[i].setValues(hist[i],1);//set estimated kalman filter
			
			m_pos[i].update(hist[i]);
			colorHist[i] = m_pos[i].getMean();
		}
	}else
	{
		for (int i = 0; i<BIN_SIZE*BIN_SIZE*BIN_SIZE; ++i)
		{
			m_pos[i].update(hist[i]);
			colorHist[i] = m_pos[i].getMean();
		}
	}
}

float AppFeature::TestHistSimi(ImageRepresentation *image, const cv::Rect &detection) const
{
	if(!TestOutlier(detection))
	{
		return -1;
	}

	cv::Rect new_bound = CropDetection(detection);

	cv::Mat ROI = image->frame(new_bound);

	float hist[BIN_SIZE*BIN_SIZE*BIN_SIZE];

	extractHistFeature(hist,ROI);

	float classify = evalHist(hist);

	return classify;
}

float AppFeature::evalHist(const float hist[BIN_SIZE*BIN_SIZE*BIN_SIZE]) const
{
	float distance_pos = WeakClassifierRGBHist::Bhattacharyya_distance(colorHist, hist, BIN_SIZE*BIN_SIZE*BIN_SIZE);
	float simi_pos = 1-distance_pos;
	return simi_pos;
}

void AppFeature::extractHistFeature(float hist[BIN_SIZE*BIN_SIZE*BIN_SIZE], const cv::Mat &ROI)
{
	std::fill(hist,hist+BIN_SIZE*BIN_SIZE*BIN_SIZE,0);

	cv::Mat Lab_ROI;
	cv::Rect crop(0.15*ROI.cols, 0.15*ROI.rows, 0.7*ROI.cols, 0.7*ROI.rows);

	cv::Mat shrinkROI = ROI(crop);

	cv::cvtColor(shrinkROI,Lab_ROI,CV_BGR2Lab);

	uchar gap = 256/BIN_SIZE;

	float h_w = Lab_ROI.cols/2.0f;
	float h_h = Lab_ROI.rows/2.0f;
	float sum_location = 0;

	for (int i = 0; i<Lab_ROI.cols; ++i)
	{
		for (int j = 0; j < Lab_ROI.rows; ++j)
		{
			cv::Vec3b val = Lab_ROI.at<cv::Vec3b>(j,i);
			uchar L = val.val[0];
			uchar a = val.val[1];
			uchar b = val.val[2];
			int indexB = L / gap;
			int indexG = a / gap;
			int indexR = b / gap;
			//weight of this location 
			float location = stdfunc::pow2((i-h_w)/h_w) + stdfunc::pow2((j-h_h)/h_h);
			if (location<=1)
			{
				float weight = 0.75f*(1-location);
				hist[(indexG*BIN_SIZE+indexR)*BIN_SIZE+indexB] += weight;
// 				hist[indexB] += weight;
// 				hist[indexG + BIN_SIZE] += weight;
// 				hist[indexR + BIN_SIZE*2] += weight;
				sum_location += weight;
			}
		}
	}

	for (int i = 0; i < BIN_SIZE*BIN_SIZE*BIN_SIZE; ++i)
	{
		hist[i]/=sum_location;
	}

}

/************************************************************************/
/* Crop body detection to a refined region                              */
/************************************************************************/
cv::Rect AppFeature::CropDetection(const cv::Rect &original)
{
	//crop accord to the inner setting of tracker (mainly upper)
	cv::Rect new_bound;
	new_bound.width = int(original.width/2.0f + 0.5f);
	new_bound.height = int(original.height + 0.5f);
	new_bound.y = int(original.y + 0.5f);
	new_bound.x = int(original.x + original.width/4.0f + 0.5f);

	return original;
}

/*

cv::Rect AppFeature::CropDetection(const cv::Rect &original)
{
//crop accord to the inner setting of tracker (mainly upper)
cv::Rect new_bound;
new_bound.width = int(original.width/2.0f + 0.5f);
new_bound.height = int(original.height*0.35f +0.5f);
new_bound.y = int(original.y + original.height * 0.19f + 0.5f);
new_bound.x = int(original.x + original.width/4.0f + 0.5f);

return new_bound;
}

*/

sir_filter::State AppFeature::InitParticleFilter(const sir_filter::State &state)
{
	cv::Rect new_bound(state.pos_x-state.scale_x/2.0f,state.pos_y-state.scale_y/2.0f,state.scale_x,state.scale_y);
	return InitParticleFilter(new_bound);
}

/*Init particle filter*/
sir_filter::State AppFeature::InitParticleFilter(const cv::Rect &new_bound)
{
	float aspect = MIN_WIDTH / float(MIN_HEIGHT);
	// Draw samples from initial detection 
	float scale_det = new_bound.width / float(MIN_WIDTH);
	float sigma = 14 * scale_det; // initial pos sigma for drawing samples
	float sigma_scale = 0; // scale sigma
	float center[] = {new_bound.x + new_bound.width/2.0f, new_bound.y + new_bound.height/2.0f};
	std::vector<sir_filter::State> state;
	std::vector<float> weight;
	state.reserve(sir_filter::SIRFilter::N_PARTICLE);
	weight.reserve(sir_filter::SIRFilter::N_PARTICLE);

	float sum_w = 0;
	float max_weight = 0;
	sir_filter::State max_state;
	//draw std randn noise centered at bounding box center
	for(int i = 0; i < sir_filter::SIRFilter::N_PARTICLE; ++i)
	{
		float noise_x = stdfunc::randn(0,sigma);
		float noise_y = stdfunc::randn(0,sigma);
		sir_filter::State sample_state;
		sample_state.pos_x = center[0] + noise_x;
		sample_state.pos_y = center[1] + noise_y;

		float noise_scale_w = stdfunc::randn(0,sigma_scale);
		float noise_scale_h = noise_scale_w / aspect;
		sample_state.scale_x = new_bound.width + noise_scale_w;
		sample_state.scale_y = new_bound.height + noise_scale_h;

		cv::Rect sample_bound(int(sample_state.pos_x-sample_state.scale_x/2+0.5f), int(sample_state.pos_y - sample_state.scale_y/2+0.5f) ,
			int(sample_state.scale_x + 0.5f), int(sample_state.scale_y + 0.5f));

		float w = GetOverLapPercentage(sample_bound,new_bound);
		state.push_back(sample_state);
		weight.push_back(w);
		sum_w += w;
		if (w > max_weight)
		{
			max_weight = w;
			max_state = sample_state;
		}
	}

	//normalize weight
	for(std::vector<float>::iterator iter = weight.begin(); iter != weight.end(); ++iter)
	{
		(*iter) /= sum_w;
	}

	particle_filter.SetInitState(state,weight);
	return max_state;
}

float AppFeature::TestSimi(ImageRepresentation *image, const cv::Rect &detection, bool use_negative)
{
	if(!TestOutlier(detection))
	{
		return 0;
	}

	cv::Rect new_bound = CropDetection(detection);
	
	float classify =  tracker.Evaluate(image,new_bound,use_negative);
	return classify;
}

/*Particle filtering when detection responses are associated*/
void AppFeature::DetectionCheck(const float vec[2], const cv::Rect &detection, const std::list<ImageRepresentation*> &frame_buffer, int frame_no, const cv::Rect &predict_bound, sir_filter::State &max_state)
{
	std::vector<sir_filter::State> state_org = particle_filter.GetParticleStates();
	float scale_det = predict_bound.width / float(MIN_WIDTH);
	float sig_pos = 6 * scale_det;
	float sig_scale = 0;
	int frame_gap = frame_buffer.size();

	particle_filter.DrawSamplesFromStateTransition(vec, frame_gap, sig_pos, sig_scale);
	std::vector<sir_filter::State> proposed_state = particle_filter.GetParticleStates();
	//Calculate weight for each transited particle
	std::vector<float> weight;
	float max_weight = CalWeightForParticle(detection,NULL,proposed_state,weight,max_state);
	//Whether reinit
	max_state = InitParticleFilter(detection);
}

/*Calculate weight for each particle*/
float AppFeature::CalWeightForParticle(const cv::Rect &detection, ImageRepresentation *image, const std::vector<sir_filter::State> &particles, std::vector<float> &weight, sir_filter::State &max_state)
{
	weight.reserve(sir_filter::SIRFilter::N_PARTICLE);
	float max_weight = 0;
	if(detection.area() == 0) //no detection 
	{
		assert(image!=NULL);
		for(std::vector<sir_filter::State>::const_iterator citer = particles.cbegin(); citer != particles.cend(); ++citer)
		{
			const sir_filter::State &s = (*citer);
			cv::Rect candidate(int(s.pos_x - s.scale_x/2 + 0.5f), int(s.pos_y - s.scale_y/2 + 0.5f), int(s.scale_x + 0.5f), int(s.scale_y + 0.5f));
			
			float w = TestSimi(image,candidate,false); // w ranges from [0, 1]
			
			if(max_weight < w)
			{
				max_weight = w;
				max_state = s;
			}
			weight.push_back(w);
		}
	}else //have a detection, weight is determined by associated detection result
	{
		for(std::vector<sir_filter::State>::const_iterator citer = particles.cbegin(); citer != particles.cend(); ++citer)
		{
			const sir_filter::State &s = (*citer);
			cv::Rect candidate(int(s.pos_x - s.scale_x/2 + 0.5f), int(s.pos_y - s.scale_y/2 + 0.5f), int(s.scale_x + 0.5f), int(s.scale_y + 0.5f));
			
			float w = GetOverLapPercentage(detection, candidate);
			if(max_weight < w)
			{
				max_weight = w;
				max_state = s;
			}

			weight.push_back(w);
		}
	}

	return max_weight;
}

/*Get object's current hidden state*/
sir_filter::State AppFeature::GetObjState()
{
	sir_filter::State avg_state = particle_filter.GetAvgState();
	return avg_state;
}

/*Test whether bound is inside the image*/
static bool TestOutlier(const cv::Rect &bound)
{
	if(bound.x <0 || bound.y<0 || bound.br().x >= image_width || bound.br().y >= image_height)
	{
		return false;//out
	}
	if(bound.width<=0 || bound.height<=0)
		return false;

	return true;//inside image frame
}