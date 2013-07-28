#pragma once
#include "WeakClassifier.h"
#include "EstimatedGaussDistribution.h"

class WeakClassifierRGBHist : public WeakClassifier
{
private:
	enum{
		BIN_SIZE = 4, //bin size per channel
	};
	EstimatedGaussDistribution m_pos[BIN_SIZE*3]; //a kalman filtered gaussian for each bin
	EstimatedGaussDistribution m_neg[BIN_SIZE*3];
	float *histPos, *histNeg;
	bool first_pos,first_neg;
	float left_scale,upper_scale;
	Rect ROI; //Region of interest
	Size initPatchSize;
	void generateRandomClassifier(Size patchSize);
	void extractFeatures(const cv::Mat &img, float *hist);
	int predict(const float *hist);
	bool evalFeatures(ImageRepresentation *image, Rect ROI, float *hist);
public:
	WeakClassifierRGBHist(Size patchSize);
	virtual ~WeakClassifierRGBHist(void);

	bool update(ImageRepresentation* image, Rect ROI, int target); 

	int eval(ImageRepresentation* image, Rect ROI); 

	float getValue(ImageRepresentation* image, Rect ROI);

	int getType(){return 2;} //indicate color histogram feature

	void print();

	static float normalizedCorrelation(const float *hist1, const float *hist2, unsigned int N);
	static float Bhattacharyya_distance(const float *hist1, const float *hist2, unsigned int N);
};

