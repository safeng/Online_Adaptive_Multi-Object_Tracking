#include "SIRFilter.h"
#include "trackingParam.h"
#include <cassert>
#include <cstdlib>
#include <iostream>

using namespace sir_filter;

State::State():pos_x(0),pos_y(0),scale_x(0),scale_y(0)
{
}

cv::Rect State::ConvertToRect() const
{
	cv:: Rect state;
	state.x = int(this->pos_x - this->scale_x/2 + 0.5f);
	state.y = int(this->pos_y - this->scale_y/2 + 0.5f);
	state.width = int(this->scale_x+0.5f);
	state.height = int(this->scale_y+0.5f);
	return state;
}

void State::Clear()
{
	this->pos_x = 0;
	this->pos_y = 0;
	this->scale_x = 0;
	this->scale_y = 0;
}

void State::operator+= (const State &s)
{
	this->pos_x += s.pos_x;
	this->pos_y += s.pos_y;
	this->scale_x += s.scale_x;
	this->scale_y += s.scale_y;
}

State State::operator+ (const State &s)
{
	State state;
	state.pos_x = this->pos_x + s.pos_x;
	state.pos_y = this->pos_y + s.pos_y;
	state.scale_x = this->scale_x + s.scale_x;
	state.scale_y = this->scale_y + s.scale_y;
	return state;
}

State State::operator/ (float numerator)
{
	State state;
	state.pos_x = this->pos_x / numerator;
	state.pos_y = this->pos_y / numerator;
	state.scale_x = this->scale_x / numerator;
	state.scale_y = this->scale_y / numerator;

	return state;
}

State State::operator* (float multiplier)
{
	State state;
	state.pos_x = this->pos_x * multiplier;
	state.pos_y = this->pos_y * multiplier;
	state.scale_x = this->scale_x * multiplier;
	state.scale_y = this->scale_y * multiplier;
	
	return state;
}

State &State::operator= (const cv::Rect &rect)
{
	this->pos_x = rect.x + rect.width/2.0f;
	this->pos_y = rect.y + rect.height/2.0f;
	this->scale_x = rect.width;
	this->scale_y = rect.height;
	return *this;
}


const int SIRFilter::N_PARTICLE = 200;

SIRFilter::SIRFilter(void)
{
}


SIRFilter::~SIRFilter(void)
{
}

/*Systematic resampling*/
void SIRFilter::SystematicResampling()
{
	assert(state.size() == N_PARTICLE && state.size() == weight.size());
	// new samples and new weight
	// after resampling, weights are reset to 1/Ns
	std::vector<State> newState (N_PARTICLE);

	std::vector<float> cdf(N_PARTICLE,0); // init cdf
	cdf[0] = weight[0]; // use weight[0]
	for(int i = 1; i < N_PARTICLE; ++i) // construct cdf (from the second)
	{
		cdf[i] = cdf[i-1] + weight[i];
	}

	float u0 = rand() / float(RAND_MAX); // draw a starting point [0,1]
	assert(u0>=0 && u0<=1);
	u0 = u0 / N_PARTICLE;
	int i = 0; // start at the bottom of cdf

	for(int j = 0; j < N_PARTICLE; ++j)
	{
		float u;
		u = u0 + j/float(N_PARTICLE);
		while (u>cdf[i])
		{
			++i;
		}

		//assign new state
		newState[j] = state[i];
	}

	//reset weight & new state
	weight.assign(weight.size(),1.0f/N_PARTICLE);
	state.assign(newState.cbegin(),newState.cend());
}

/*Normalize weight*/
void SIRFilter::WeightNormalize(float sum)
{
	assert(sum>0);
	for(std::vector<float>::iterator iter = weight.begin(); iter != weight.end(); ++iter)
	{
		(*iter) = (*iter)/sum;
	}
}

/*Set initial states of particles*/
void SIRFilter::SetInitState(const std::vector<State> &state, const std::vector<float> &weight)
{
	this->state.assign(state.cbegin(), state.cend());
	this->weight.assign(weight.cbegin(),weight.cend());
}

/*Draw samples from previous state by transition function*/
void SIRFilter::DrawSamplesFromStateTransition(const float vec[2], int frame_gap, float sig_pos, float sig_scale)
{
	// sigma may vary from different stages in tracking 
	assert(state.size() == N_PARTICLE && state.size() == weight.size());

	float frame_elapsed = frame_gap / float(MO_GROUP-1);
	float movement[] = {vec[0] * frame_elapsed, vec[1] * frame_elapsed};

	for(std::vector<State>::iterator iter = state.begin(); iter != state.end(); ++iter)
	{
		State &s_x = *iter; //one particle
		float noise_pos = stdfunc::randn(0,sig_pos);
		s_x.pos_x += noise_pos + movement[0];
		noise_pos = stdfunc::randn(0,sig_pos);
		s_x.pos_y += noise_pos + movement[1];

		// scale still holds the aspect ratio
		float aspect_ratio = s_x.scale_x / s_x.scale_y;
		float noise_scale_x = stdfunc::randn(0,sig_scale);
		float noise_scale_y = noise_scale_x / aspect_ratio;

		s_x.scale_x += noise_scale_x;
		s_x.scale_y += noise_scale_y;
	}
}

/*Set the new weights for each transited particle*/
void SIRFilter::SetMeasurementLikelihood(const std::vector<float> &weight)
{
	assert(weight.size() == this->weight.size());
	this->weight.assign(weight.cbegin(),weight.cend());
}

/*SIR Particle Filter*/
void SIRFilter::SIRParticleFilter()
{
	//assume that state transition step is completed

	//Calculate total weight
	float sum = 0;
	for(std::vector<float>::const_iterator citer = weight.cbegin(); citer != weight.cend(); ++citer)
	{
		sum += (*citer);
	}

	if (sum == 0)
	{
		assert(false);
	}
	//Normalize
	WeightNormalize(sum);

	SystematicResampling();

}

/*Get the average particle for estimation*/
State SIRFilter::GetAvgState()
{
	State state;
	for(int i = 0; i < N_PARTICLE; ++i)
	{
		state += this->state[i] * this->weight[i];
	}

	return state;
}

/*Set particle states*/
void SIRFilter::SetStates(const std::vector<State> &state)
{
	assert(state.size() == this->state.size());
	this->state.assign(state.cbegin(),state.cend());
}

/*Get States of all particles*/
const std::vector<State> &SIRFilter::GetParticleStates()
{
	return this->state;
}

void SIRFilter::RetrieveWeights(std::vector<float> &weight_container)
{
	weight_container.assign(weight.cbegin(),weight.cend());
}

/************************************************************************/
/* Set appearance search region from particle states                    */
/************************************************************************/
cv::Rect SIRFilter::GetSearchRegion()
{
	int min_x = INT_MAX, max_x = 0;
	int min_y = INT_MAX, max_y = 0;
	for (size_t i = 0; i<state.size(); ++i)
	{
		const State &s = state[i];
		if (s.pos_x<min_x)
		{
			min_x = s.pos_x;
		}
		if (s.pos_x>max_x)
		{
			max_x = s.pos_x;
		}
		if (s.pos_y<min_y)
		{
			min_y = s.pos_y;
		}
		if (s.pos_y>max_y)
		{
			max_y = s.pos_y;
		}
	}
	cv::Rect searchRegion;
	searchRegion.x = min_x;
	searchRegion.y = min_y;
	searchRegion.width = max_x-min_x;
	searchRegion.height = max_y-min_y;
	assert(searchRegion.width>0 && searchRegion.height>0);
	return searchRegion;
}

void SIRFilter::PaintParticles(cv::Mat &img, const cv::Rect &searchRegion,  bool lost) const
{
	uchar color[3];
	if (lost)
	{
		//red
		color[0] = 255;
		color[1] = 0;
		color[2] = 0;
		//also paint searching region
		cv::rectangle(img,searchRegion,cv::Scalar(255,0,0),2);
		
	}else
	{
		//green
		color[0] = 0;
		color[1] = 255;
		color[2] = 0;
	}
	int dot_size = 1;
	for(int i = 0; i<N_PARTICLE ; ++i)
	{
		int x = int(state[i].pos_x+0.5f);
		int y = int(state[i].pos_y+0.5f);
		for(int xx = x-dot_size; xx <= x+dot_size; ++xx)
		{
			if(xx>=0 && xx<img.cols)
			{
				for(int yy = y-dot_size; yy <=y+dot_size; ++yy)
				{
					if(yy>=0 && yy<img.rows)
					{
						cv::Vec3b &val = img.at<cv::Vec3b>(yy,xx);
						val.val[0] = color[2];
						val.val[1] = color[1];
						val.val[2] = color[0];
					}
				}
			}
		}
	}
}

void SIRFilter::DrawPosParticles(const cv::Mat &frame, const std::vector<float> &new_weight, const std::vector<State> &posStates) const
{
	cv::Mat img;
	frame.copyTo(img);
	uchar color[3] = {0};
	float min_weight = FLT_MAX;
	float max_weight = -FLT_MAX;
	for (size_t i = 0;i<new_weight.size();++i)
	{
		if (new_weight[i]>max_weight)
		{
			max_weight = new_weight[i];
		}
		if (new_weight[i]<min_weight)
		{
			min_weight = new_weight[i];
		}
	}

	float gap = max_weight - min_weight;
	int dot_size = 1;
	for (size_t i = 0; i<posStates.size(); ++i)
	{

		float w = (new_weight[i]-min_weight)/gap; //normalize to [0,1]
		color[0] = uchar(255u*w);
		color[2] = uchar(255u*(1-w));

		int x = int(posStates[i].pos_x+0.5f);
		int y = int(posStates[i].pos_y+0.5f);
		for(int xx = x-dot_size; xx <= x+dot_size; ++xx)
		{
			if(xx>=0 && xx<img.cols)
			{
				for(int yy = y-dot_size; yy <=y+dot_size; ++yy)
				{
					if(yy>=0 && yy<img.rows)
					{
						cv::Vec3b &val = img.at<cv::Vec3b>(yy,xx);
						val.val[0] = color[2];
						val.val[1] = color[1];
						val.val[2] = color[0];
					}
				}
			}
		}
	}

	cv::imwrite("PosParticles.jpg",img);
}


void SIRFilter::DrawPosRects(const cv::Mat &img, const std::vector<float> &new_weight, const std::vector<cv::Rect> &posRects) const
{
	std::vector<State> posStates;
	posStates.reserve(posRects.size());
	for (size_t i = 0; i<posRects.size(); ++i)
	{
		State s;
		s = posRects[i];
		posStates.push_back(s);
	}

	DrawPosParticles(img,new_weight,posStates);
}

/************************************************************************/
/* Get positive states                                                  */
/************************************************************************/
void SIRFilter::GetPosStates(const std::vector<float> &new_weight, float minMargin, std::vector<State> &posStates, std::vector<float> &posWeight) const
{
	posStates.reserve(new_weight.size()/2);
	posWeight.reserve(posStates.size());

	for (size_t i = 0;i<new_weight.size();++i)
	{
		if (new_weight[i]>minMargin)
		{
			posStates.push_back(state[i]);
			posWeight.push_back(new_weight[i]-minMargin);
		}
	}
}