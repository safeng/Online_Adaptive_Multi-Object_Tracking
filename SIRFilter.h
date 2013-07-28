#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

namespace sir_filter
{
	/*The state of an object*/
	struct State
	{
		// position
		float pos_x;
		float pos_y;
		// scale
		float scale_x;
		float scale_y;

		State();
		void Clear();
		cv::Rect ConvertToRect() const;
		void operator += (const State &s);
		State operator +(const State &s);
		State operator /(float norminator);
		State operator *(float multiplier);
		State &operator = (const cv::Rect &rect);
	};

	/*sampling importance resampling filter*/
	class SIRFilter
	{
	private:
		std::vector<State> state; // state of particles
		std::vector<float> weight; // weight of particles
		void SystematicResampling(); //systematic resampling
		void WeightNormalize(float sum); //normalize weight

	public:
		static const int N_PARTICLE; // # of particles per iteration
		cv::Rect GetSearchRegion();
		void SetInitState(const std::vector<State> &state, const std::vector<float> &weight); // Set initial state and weight
		void DrawSamplesFromStateTransition(const float vec[2], int frame_gap, float sig_pos, float sig_scale); //Draw samples from state transition function
		const std::vector<sir_filter::State> &GetParticleStates(); //Get states of particles (X)
		void SetMeasurementLikelihood(const std::vector<float> &weight); // Set weight for each particles (P(X|Z))
		void SetStates(const std::vector<sir_filter::State> &state);
		void SIRParticleFilter();
		State GetAvgState(); //Get the average state
		void RetrieveWeights(std::vector<float> &weight_container);
		void PaintParticles(cv::Mat &img, const cv::Rect &searchRegion, bool lost) const;
		void DrawPosParticles(const cv::Mat &img, const std::vector<float> &new_weight, const std::vector<State> &posStates) const;
		void DrawPosRects(const cv::Mat &img, const std::vector<float> &new_weight, const std::vector<cv::Rect> &posRects) const;
		void GetPosStates(const std::vector<float> &new_weight, float minMargin, std::vector<State> &posStates, std::vector<float> &posWeight) const;
		SIRFilter(void);
		~SIRFilter(void);
	};
}