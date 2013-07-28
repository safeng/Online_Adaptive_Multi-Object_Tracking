#pragma once
#include "PeopleDetector/Pedestrian.h"
class PeopleDetector
{
private:
	DetectionScanner scanner;
	void PostProcess(std::vector<CRect>& result,const int combine_min);
	void RemoveCoveredRectangles(std::vector<CRect>& result);
	void LoadCascade();

public:
	static const int HUMAN_height;
	static const int HUMAN_width;
	static const int HUMAN_xdiv;
	static const int HUMAN_ydiv;
	static const int EXT;

	PeopleDetector(void);
	~PeopleDetector(void);
	int DetectHuman(const cv::Mat &image, std::vector<cv::Rect> &detection);
};