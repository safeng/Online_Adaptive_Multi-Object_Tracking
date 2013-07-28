/*This class does HoG human upper body detection using the detector read from file*/
#include <opencv2\opencv.hpp>
#include <opencv\cv.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>
#include <string>
#include <list>

#define SAMPLE_X 32U
#define SAMPLE_Y 40U

#pragma once
/*Class to store the HoG parameters*/
class Arg
{
private:
	cv::Size win_size;//size of the window
	cv::Size block_size;//size of the block
	cv::Size cell_size;//size of the cell
	cv::Size block_stride;//stride of the block
	cv::Size padding;// size of the padding at detection
	int nbins;//Number of beans of histogram
	double win_sigma;//sigma of Gaussian smoothing window
	double threshold_L2hys;//threshold for normalization
	bool gamma_correction;//whether to need gamma_correction or not
	int nlevels;//Maximum number of detection window increases
	cv::Size win_stride;//Window stride to generate descriptors for one training sample
	double hit_threshold;//Threshold for distance between features and SVM classifying plane
	double group_threshold;//Coefficient to regulate the similarity threshold. Group multiple rectangles at some objects
	double scale0;//Coefficient of the detection window increase
public:
	friend class HoGDetection;
	Arg(const cv::Size &win_size = cv::Size(SAMPLE_X,SAMPLE_Y) , const cv::Size &block_size = cv::Size(16,16), const cv::Size &block_stride = cv::Size(8,8), const cv::Size &cell_size = cv::Size(8,8), const cv::Size &win_stride = cv::Size(8,8),
		int nbins = 9, double win_sigma = -1, double threshold_L2hys = 0.2,
		bool gamma_correction = true, int nlevels = cv::HOGDescriptor::DEFAULT_NLEVELS, const cv::Size &padding = cv::Size(0,0),
		double hit_threshold = 0.7, double group_threshold = 2.0, double scale0 = 1.05)//Can be initialized with the default value
	:win_size(win_size), block_size(block_size), block_stride(block_stride), cell_size(cell_size), win_stride(win_stride),
	nbins(nbins), win_sigma(win_sigma), threshold_L2hys(threshold_L2hys),
	gamma_correction(gamma_correction),nlevels(nlevels), padding(padding),
	hit_threshold(hit_threshold),group_threshold(group_threshold),scale0(scale0){}
};

class HoGDetection
{
private:

	cv::HOGDescriptor full_body;//full body HoG detector
	std::vector<cv::Rect> found_full_body;
	std::vector<cv::Rect> found_full_body_filtered;
	std::vector<double> weight_full_body;
	std::vector<double> weight_full_body_filtered;

	cv::HOGDescriptor *hog;//hog descriptor with predefined parameters
	const Arg arg;//HoG parameters
	std::vector<float> detector; //svm classifier trained by svmlight (init in constructor)
	std::vector<cv::Rect> found; //vectors to store classification result in rectangles (also serve as the final result)
	std::vector<cv::Rect> found_filtered; //filtered results
	std::vector<double> weight; //the weight of hit
	std::vector<double> weight_filtered;//filtered weight
	std::vector<double> scores; //final scores of each detection
	std::vector<double> prediction_matching_scores; //The prediction matching scores for each detection
	std::vector<double> prediction_matching_scores_final;//prediction matching score corresponding to final detections
	cv::Rect prediction; //Only one prediction for each detection
	double matching_score;//The matching score of tracking.
	const std::string vectorFileName;//fileName of the svm vector
	void LoadDetectorFromFile();//Load svm vector from file
	void ComputeScores(const cv::Mat &img, bool init, double &normalization_term, bool isPredictTrustable = true);//compute scores for all detections
	void VisualizeMatrix(const cv::Mat &image, const cv::Mat &result, const std::string &fileName, const std::list<cv::Rect> &maxima_detection);
	void HillClimbing(const cv::Mat &density, const std::list<cv::Point> &initials, float threshold, std::list<cv::Point> &maxima);//Finding the local maxima
	void GetFinalDetection(bool init, const std::list<cv::Point> &maxima, std::list<cv::Rect> &final_detection, const cv::Mat &image, bool isPredictionTrustable = true);//From a point to a rect

public:
	HoGDetection(const std::string &vectorFileName);//We use built-in default parameters for HoG
	~HoGDetection(void);
	HoGDetection(const Arg &arg,const std::string &vectorFileName);//Init with a parameter object
	void detect(const cv::Mat& image);//C++ interface

	const std::vector<cv::Rect> &GetFoundFiltered() const { return found_filtered; }
	const std::vector<double> &GetWeightFiltered() const { return weight_filtered; }
	const std::vector<cv::Rect> &GetFullBodyFound() const { return found_full_body_filtered; }
	const std::vector<double> &GetFullBodyWeight() const { return weight_full_body_filtered; }

	void ComputePostProb(bool init, const cv::Mat &image, bool isPredictTrustable = true,
		bool enable_visualization = false, const std::string &fileName = std::string(), size_t missing_frames = 0);//compute the posterior probability
	void drawResult(const cv::Mat& image, const std::string &fileName);//C++ interface to save img to disk
	void drawResult(cv::Mat &img);
	void SetBoundingBox(const cv::Mat &image, cv::Mat &sub_img, int relative_pos[2] ,const cv::Rect &ROI);
	void SetPredictions(const cv::Rect &prediction, double matching_score);//Set predictions (this function is called after data association)
};