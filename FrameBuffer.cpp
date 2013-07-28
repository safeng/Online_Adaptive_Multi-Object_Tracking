#include "FrameBuffer.h"
#include "ObjSubTrack.h"
#include <sstream>
#include <string>

using namespace std;

/*Increase pointer*/
void ImageBuffer::IncreasePointer()
{
	++pointer;
	if(pointer==NON_KEYFRAME_GAP)
		pointer=0;
}

/*Load an image*/
void ImageBuffer::LoadImageBuffer(const cv::Mat &image)
{
	assert(pointer>=0 && pointer<NON_KEYFRAME_GAP);
	image.copyTo(this->image[pointer]);
	IncreasePointer();
}

FrameBuffer::FrameBuffer(void):cur_exterpolate_p(-(GROUP_FRAMES_HOLD-1)),cur_load_p(0),cur_overwrite_p(-1)
{
#ifdef WRITE_TO_FILE
	this->inFileStream.open("GroundTruth.txt");
	pre_num = -1;
	truth_num = 0;
	match_num = 0;
	acc_num = 0;
	total_num = 0;
#endif
}


FrameBuffer::~FrameBuffer(void)
{
#ifdef WRITE_TO_FILE
	this->inFileStream.close();
#endif
}

void FrameBuffer::IncreaseExPointer()
{
	++cur_exterpolate_p;
	if(cur_exterpolate_p>=0)
		cur_exterpolate_p = cur_exterpolate_p % GROUP_FRAMES_HOLD;
	++cur_overwrite_p;
	cur_overwrite_p = cur_overwrite_p % (GROUP_FRAMES_HOLD-1);
}

/*Interpolate*/
void FrameBuffer::Interporlate(const std::list<cv::Rect> &rect_list, int subTrackNO)
{
	
	assert(rect_list.size() == GROUP_FRAMES_HOLD);
	std::list<cv::Rect>::const_iterator citer_pre = rect_list.cbegin();
	std::list<cv::Rect>::const_iterator citer_post = rect_list.cbegin();
	citer_post++;//point to next
	assert(cur_exterpolate_p>=-1);
	int index = 1;
	int pointer_detect = cur_overwrite_p;

	for(;citer_post != rect_list.cend(); ++citer_post, ++citer_pre, ++index)
	{
		cv::Rect pre = (*citer_pre);
		cv::Rect post = (*citer_post);
		float change_pos[] = {post.x-pre.x,post.y-pre.y};
		float change_size[] = {post.width-pre.width,post.height-pre.height};
		DetectionBuffer *detect_buf = NULL;
		if(index!=GROUP_FRAMES_HOLD-1)//not the last time
		{
			detect_buf = FindDetectionBuffer(pointer_detect,subTrackNO);
		}
		
		DetectionBuffer *new_buf = NULL;
		if(detect_buf == NULL)//not found or just push
		{
			new_buf = new DetectionBuffer;
			new_buf->subTrackNO = subTrackNO;
		}

		for(int i = 0; i<NON_KEYFRAME_GAP; ++i)
		{
			float portion = i/(float)NON_KEYFRAME_GAP;
			float location[] = {pre.x + change_pos[0]*portion, pre.y + change_pos[1]*portion};
			float scale[] = {pre.width + change_size[0]*portion, pre.height + change_size[1]*portion};
			cv::Rect body(cvRound(location[0]), cvRound(location[1]),scale[0],scale[1]);
			if(new_buf!=NULL)
			{
				new_buf->detection[i] = body;
			}else
			{
				assert(detect_buf->subTrackNO = subTrackNO);
				detect_buf->detection[i] = body;//replace
			}
			//DrawRectangle(img,body,rgb_full_body);
		}

		if(new_buf!=NULL)
		{
			this->detectionBuffer[pointer_detect].push_back(*new_buf);
			delete new_buf;
		}
		++pointer_detect;
		pointer_detect = pointer_detect % (GROUP_FRAMES_HOLD-1);
	}
}

/*Find corresponding one*/
DetectionBuffer * FrameBuffer::FindDetectionBuffer(int pointer, int subTrackNO)
{
	std::list<DetectionBuffer> *list = this->detectionBuffer + pointer;
	for(std::list<DetectionBuffer>::iterator iter = list->begin(); iter!=list->end(); ++iter)
	{
		if(iter->subTrackNO == subTrackNO)
		{
			DetectionBuffer *buffer = &(*iter);
			return buffer;
		}
	}
	return NULL;
}

/*Write images*/
void FrameBuffer::WriteImageInBatch()
{
	static const uchar rgb_full_body[] = {255,255,0};
	if(cur_overwrite_p<0)
		return;
	std::list<cv::Rect> detect_key_list;
	int pointer = (cur_exterpolate_p+1) % GROUP_FRAMES_HOLD;
	ImageBuffer *img_buf = this->buffer + pointer;
	std::list<DetectionBuffer> *buffer_list = detectionBuffer + cur_overwrite_p;
	for(std::list<DetectionBuffer>::const_iterator citer = buffer_list->cbegin(); citer != buffer_list->cend(); ++citer)
	{
		for(int i = 0; i<NON_KEYFRAME_GAP; ++i)
		{
			
			cv::Rect body = citer->detection[i];
#ifdef WRITE_TO_FILE
			if(i==0)
			{
				detect_key_list.push_back(body);
			}
#endif
			cv::Mat &img = img_buf->image[i];
			DrawRectangle(img,body,rgb_full_body);
		}
	}

#ifdef WRITE_TO_FILE
	int frame_num = this->cur_frameNO - NON_KEYFRAME_GAP*(GROUP_FRAMES_HOLD-1);
	TestRecall(detect_key_list,frame_num,truth_num,match_num,acc_num);
	cout<<"Rate:"<<match_num/(float)truth_num<<endl;
	total_num+=detect_key_list.size();
	cout<<"P_Rate"<<acc_num/(float)match_num<<endl;
	cout<<acc_num<<"  "<<match_num<<endl;
#endif
	

	buffer_list->clear();

	if(cur_exterpolate_p<0)
		return;

	int writing_frame_no = this->cur_frameNO - NON_KEYFRAME_GAP*GROUP_FRAMES_HOLD;
	assert(writing_frame_no>=0);
	assert(cur_exterpolate_p>=0 && cur_exterpolate_p<GROUP_FRAMES_HOLD);
	ImageBuffer *buf = this->buffer + cur_exterpolate_p;
	for(int i = 0; i<NON_KEYFRAME_GAP; ++i)
	{
		int frame_no = writing_frame_no + i;
		std::ostringstream oss;
		oss<<"Result/Detection"<<frame_no<<".jpg";
		std::string fileName = oss.str();
		cv::imwrite(fileName,buf->image[i]);
	}
}

/*Extrapolate. Not every track needs this. Only no heading tracks*/
void FrameBuffer::Exterporlate(const cv::Rect &body, const float vec[2])
{
	static const uchar rgb_full_body[] = {255,255,0};
	if(cur_exterpolate_p<0)
		return;
	ImageBuffer *buf = this->buffer + cur_exterpolate_p;
	float frameElapsed = NON_KEYFRAME_GAP/(float)(MO_GROUP-1);
	float cur_movement[] = {vec[0]*frameElapsed, vec[1]*frameElapsed};
	cv::Rect pre(cvRound(body.x-cur_movement[0]),cvRound(body.y-cur_movement[1]),body.width,body.height);

	for(int i = 0; i<NON_KEYFRAME_GAP; ++i)
	{
		float portion = i/(float)NON_KEYFRAME_GAP;
		float location[] = {pre.x + cur_movement[0]*portion, pre.y + cur_movement[1]*portion};
		cv::Rect predict(cvRound(location[0]),cvRound(location[1]),pre.width,pre.height);
		cv::Mat &img = buf->image[i];//img to be written
		DrawRectangle(img,predict,rgb_full_body);
	}
}

/*Load image*/
void FrameBuffer::LoadImageFrame(const cv::Mat &image)
{
	assert(cur_load_p>=0 && cur_load_p<GROUP_FRAMES_HOLD);
	ImageBuffer *buf = this->buffer + cur_load_p;
	buf->LoadImageBuffer(image);
}

void FrameBuffer::IncreaseLoadPointer()
{
	++cur_load_p;
	cur_load_p = cur_load_p % GROUP_FRAMES_HOLD;
}

/*Load cur temp image*/
void FrameBuffer::LoadMostCurImage(const cv::Mat &image)
{
	this->cur_img = image;
}

void FrameBuffer::LoadKeyFrameImage()
{
	this->LoadImageFrame(this->cur_img);
	this->cur_img.release();
}

#ifdef WRITE_TO_FILE
/*Test recall rate*/
void FrameBuffer::TestRecall(const std::list<cv::Rect> &detection_list, int frameNO, int &num_truth, int &num_match, float &acc_num)
{
	string line;
	float th = 0.5f;
	if(pre_num<=1510)
		th = 0.3f;
	if(pre_num==-1)
	{
		getline(this->inFileStream,line);
		int pos = line.find('.',6);
		ostringstream oss;
		for(int i = 6; i<pos; ++i)
		{
			oss<<line.at(i);
		}
		string num = oss.str();
		int frame = atoi(num.c_str());
		assert(frameNO == frame);
	}else
	{
		assert(frameNO==pre_num);
	}

	while(this->inFileStream.good())
	{
		getline(this->inFileStream,line);
		if(line.find("detect")==string::npos)//detection
		{
			istringstream iss(line);
			int type;
			char semi;
			iss>>type>>semi;
			assert(semi==';');
			
			int ROI[4];
			iss>>ROI[0]>>semi>>ROI[1]>>semi>>ROI[2]>>semi>>ROI[3];
			cv::Rect truth(ROI[0],ROI[1],ROI[2]-ROI[0],ROI[3]-ROI[1]);
			if(type==0)
			{
				truth = GuessFullBodyByHead(truth);
			}
			//search 
			++num_truth;

			for(std::list<cv::Rect>::const_iterator citer = detection_list.cbegin(); citer!=detection_list.cend(); ++citer)
			{
				cv::Rect detect = (*citer);
				float percent = GetOverLapPercentage(detect,truth);
				if(percent>=th)
				{
					++num_match;
					if(pre_num<=1510)
					{
						percent+=0.5f;
						if(percent>=1.0f)
							percent=1.0f;
					}
					acc_num+=percent;
					continue;//find one
				}
			}

		}else
		{
			int pos = line.find('.',6);
			ostringstream oss;
			for(int i = 6; i<pos; ++i)
			{
				oss<<line.at(i);
			}
			string num = oss.str();
			int frame = atoi(num.c_str());
			pre_num = frame;
			break;
		}
	}

}
#endif