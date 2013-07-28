#pragma once
#include "OpenCVImage.h"
#include "trackingParam.h"
#include <list>
#include <fstream>

#define GROUP_FRAMES_HOLD 3
//#define WRITE_TO_FILE 

/*To manage frames*/
class ImageBuffer
{
private:
	int pointer;
	void IncreasePointer();
public:
	cv::Mat image[NON_KEYFRAME_GAP];
	ImageBuffer(void):pointer(0){}
	void LoadImageBuffer(const cv::Mat &image);
};

struct DetectionBuffer
{
	int subTrackNO;
	cv::Rect detection[NON_KEYFRAME_GAP];
};

class FrameBuffer
{
private:
#ifdef WRITE_TO_FILE
	std::ifstream inFileStream;//in File stream
	int pre_num;
	int truth_num;
	int match_num;
	float acc_num;
	int total_num;
#endif
	std::list<DetectionBuffer> detectionBuffer[GROUP_FRAMES_HOLD-1];
	int cur_exterpolate_p;//Pointer to current frames need to be exterpolated
	int cur_load_p;//pointer to current frames need to be loaded
	int cur_overwrite_p;//pointer to the current overlap block
	int cur_frameNO;//current frameNO
	cv::Mat cur_img;//most current image

	DetectionBuffer *FindDetectionBuffer(int pointer, int subTrackNO);

#ifdef WRITE_TO_FILE
	void TestRecall(const std::list<cv::Rect> &detection_list, int frameNO, int &num_truth, int &num_match, float &acc_num);
#endif
public:
	void LoadKeyFrameImage();
	void LoadImageFrame(const cv::Mat &img);
	void LoadMostCurImage(const cv::Mat &image);
	inline void SetCurFrameNO(int frameno) { this->cur_frameNO = frameno; }
	void WriteImageInBatch();//each time write NON_KEYFRAME_GAP images
	void IncreaseExPointer();
	void IncreaseLoadPointer();
	ImageBuffer buffer[GROUP_FRAMES_HOLD];//Frame buffer
	void Exterporlate(const cv::Rect &body, const float vec[2]);
	void Interporlate(const std::list<cv::Rect> &rect_list, int subTrackNO);
	FrameBuffer(void);
	~FrameBuffer(void);
};