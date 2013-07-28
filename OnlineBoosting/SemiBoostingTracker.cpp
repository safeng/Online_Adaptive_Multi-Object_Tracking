#include "SemiBoostingTracker.h"

SemiBoostingTracker::SemiBoostingTracker(void):available(false),tracking_lost(true),classifier(NULL),classifierOff(NULL),detector(NULL)
{
}


SemiBoostingTracker::~SemiBoostingTracker(void)
{
	delete classifier;
	delete classifierOff;
	delete detector;
}

void SemiBoostingTracker::SetSearchRegion(const cv::Rect &search_org, const cv::Rect &trackedPatch)
{
	Rect tracked_patch;
	tracked_patch = trackedPatch;

	//range of scale
	Rect searchedRegion_up = getTrackingROI(tracked_patch,3.5f);
	Rect searchedRegion_down = getTrackingROI(tracked_patch,2.0f);

	cv::Rect region_up,region_down;
	region_up = searchedRegion_up.ConvertToRect();
	region_down = searchedRegion_down.ConvertToRect();

	searchRegion = region_up & (search_org | region_down);
	searchRegion.width = searchRegion.height;
}

/************************************************************************/
/* Given tracked inner region returns the extended region to get samples 
/************************************************************************/
Rect SemiBoostingTracker::getTrackingROI(Rect trackedPatch, float searchFactor)
{
	Rect searchRegion;

	searchRegion = trackedPatch*(searchFactor);
	//check
	if (searchRegion.upper+searchRegion.height > validROI.height)
		searchRegion.height = validROI.height-searchRegion.upper;
	if (searchRegion.left+searchRegion.width > validROI.width)
		searchRegion.width = validROI.width-searchRegion.left;

	return searchRegion;
}

/************************************************************************/
/*Get uchar* internal data structure for cur frame                                                                    
/************************************************************************/
uchar* SemiBoostingTracker::getGrayImage(const cv::Mat &frame)
{
	cv::Mat greyImg;
	cv::cvtColor(frame,greyImg,CV_RGB2GRAY);
	IplImage ipl_gray = greyImg; //no data copying
	int rows = ipl_gray.height;
	int cols = ipl_gray.width;
	int iplCols= ipl_gray.widthStep;

	unsigned char *dataCh = new unsigned char[rows*cols];
	unsigned char *buffer = reinterpret_cast<unsigned char*>(ipl_gray.imageData);

	for(int i=0; i<rows; i++)
	{
		memcpy(dataCh+i*cols, buffer+i*iplCols, sizeof(unsigned char) * cols);
	}
	return dataCh;
}

/************************************************************************/
/* Init tracker for the first time. Perform one-shot training           */
/************************************************************************/
int SemiBoostingTracker::initTracking(ImageRepresentation *image, Rect trackedPatch, Size validROI, Patches *trackingPatches)
{
	//trackedPatch is the actual size of detection. Cropped to predefined aspect ratio
	this->validROI = validROI;

	if(!checkValidTrackPatch(trackedPatch))
		return -1;

	this->available = true;
	tracking_lost = false;

	Size trackedSize;
	trackedSize = trackedPatch;
	classifier = new StrongClassifierStandardSemi(NUM_SELECTORS,NUM_WEAK_CLASSIFIERS,trackedSize,true,ITERATION_INIT); //init with default size
	classifierOff = new StrongClassifierStandardSemi(NUM_SELECTORS,NUM_WEAK_CLASSIFIERS,trackedSize,true,ITERATION_INIT);
	detector = new Detector(classifier); //init detector
	Rect trackingROI = getTrackingROI(trackedPatch, 3.0f);
	Size trackedPatchSize;
	
	trackedPatchSize = trackedPatch;

	bool new_patches = false;
	if (trackingPatches == NULL)
	{
		trackingPatches = new PatchesRegularScan(trackingROI, this->validROI, trackedPatchSize, 0.01f);
		new_patches = true;
	}
	
	//one-shot init
	int iterationInit = 4;
	for (int curInitStep = 0; curInitStep < iterationInit; curInitStep++)
	{
		classifier->updateSemi (image, trackingPatches->getSpecialRect ("UpperLeft"), -1);
		classifier->updateSemi (image, trackedPatch, 1);
		classifier->updateSemi (image, trackingPatches->getSpecialRect ("UpperRight"), -1);
		classifier->updateSemi (image, trackedPatch, 1);
		classifier->updateSemi (image, trackingPatches->getSpecialRect ("LowerLeft"), -1);
		classifier->updateSemi (image, trackedPatch, 1);
		classifier->updateSemi (image, trackingPatches->getSpecialRect ("LowerRight"), -1);
		classifier->updateSemi (image, trackedPatch, 1);
		classifier->updateSemi (image,trackingPatches->getSpecialRect ("UpperMiddle"), -1);
		classifier->updateSemi (image, trackedPatch, 1);
		classifier->updateSemi (image,trackingPatches->getSpecialRect ("LowerMiddle"), -1);
		classifier->updateSemi (image, trackedPatch, 1);
		classifier->updateSemi (image,trackingPatches->getSpecialRect ("MiddleLeft"), -1);
		classifier->updateSemi (image, trackedPatch, 1);
		classifier->updateSemi (image,trackingPatches->getSpecialRect ("MiddleRight"), -1);
		classifier->updateSemi (image, trackedPatch, 1);
	}

	//one (first) shot learning
	for (int curInitStep = 0; curInitStep < iterationInit; curInitStep++)
	{
		classifierOff->updateSemi (image, trackedPatch, 1);
		classifierOff->updateSemi (image, trackingPatches->getSpecialRect ("UpperLeft"), -1);
		classifierOff->updateSemi (image, trackedPatch, 1);
		classifierOff->updateSemi (image, trackingPatches->getSpecialRect ("UpperRight"), -1);
		classifierOff->updateSemi (image, trackedPatch, 1);
		classifierOff->updateSemi (image, trackingPatches->getSpecialRect ("LowerLeft"), -1);
		classifierOff->updateSemi (image, trackedPatch, 1);
		classifierOff->updateSemi (image, trackingPatches->getSpecialRect ("LowerRight"), -1);

		classifierOff->updateSemi (image,trackingPatches->getSpecialRect ("UpperMiddle"), -1);
		classifierOff->updateSemi (image, trackedPatch, 1);
		classifierOff->updateSemi (image,trackingPatches->getSpecialRect ("LowerMiddle"), -1);
		classifierOff->updateSemi (image, trackedPatch, 1);
		classifierOff->updateSemi (image,trackingPatches->getSpecialRect ("MiddleLeft"), -1);
		classifierOff->updateSemi (image, trackedPatch, 1);
		classifierOff->updateSemi (image,trackingPatches->getSpecialRect ("MiddleRight"), -1);
		classifierOff->updateSemi (image, trackedPatch, 1);
	}

	if (new_patches)
	{
		delete trackingPatches;
	}

	return iterationInit;
}

/************************************************************************/
/* Clear tracking context                                               */
/************************************************************************/
void SemiBoostingTracker::Clear()
{
	available = false;
	tracking_lost = true;
	validROI = Rect();
	delete classifier;
	delete classifierOff;
	delete detector;
	classifier = NULL;
	classifierOff = NULL;
	detector = NULL;
}

/************************************************************************/
/* Given a detection result, output classifier's result                 */
/************************************************************************/
float SemiBoostingTracker::Evaluate(ImageRepresentation *image, const cv::Rect &region, bool use_negative)
{
	assert(available);
	
	Rect ROI;
	ROI = region;
	//check
	if (ROI.upper<=0)
	{
		ROI.upper = 0;
	}

	if (ROI.left<=0)
	{
		ROI.left = 0;
	}

	if (ROI.left + ROI.width > validROI.width + validROI.left)
	{
		ROI.width = validROI.width + validROI.left - ROI.left;
	}

	if (ROI.upper + ROI.height > validROI.height + validROI.upper)
	{
		ROI.height = validROI.height + validROI.upper - ROI.upper;
	}

	float confidence;
	{
		float val = 1/classifier->getSumAlpha();
		confidence = classifier->eval(image,ROI)*val;
	}
	
	if (confidence<0 && !use_negative)
	{
		confidence = 0;
	}

	return confidence;
}

/************************************************************************/
/* Update recognizer from off-line trained detection result             */
/************************************************************************/
int SemiBoostingTracker::updateOn(ImageRepresentation *image, Rect detectionPatch)
{
	Rect trackingROI = getTrackingROI(detectionPatch, 3.0f);
	Size trackedPatchSize;

	trackedPatchSize = detectionPatch;
	Patches* trackingPatches = new PatchesRegularScan(trackingROI, validROI, trackedPatchSize, 0.01f);
	int iteration = 1;
	for (int i = 0;i<iteration;++i)
	{
		classifierOff->updateSemi(image, trackingPatches->getSpecialRect ("UpperLeft"), -1);
		classifierOff->updateSemi(image, detectionPatch, 1);
		classifierOff->updateSemi(image, detectionPatch, 1); 
		classifierOff->updateSemi(image, trackingPatches->getSpecialRect ("UpperRight"), -1);
		classifierOff->updateSemi(image, detectionPatch, 1);
		classifierOff->updateSemi(image, detectionPatch, 1); 
		classifierOff->updateSemi(image, trackingPatches->getSpecialRect ("LowerLeft"), -1);
		classifierOff->updateSemi(image, detectionPatch, 1);
		classifierOff->updateSemi(image, detectionPatch, 1); 
		classifierOff->updateSemi(image, trackingPatches->getSpecialRect ("LowerRight"), -1);
		classifierOff->updateSemi(image, detectionPatch, 1);
		classifierOff->updateSemi(image, detectionPatch, 1); 

		classifierOff->updateSemi (image,trackingPatches->getSpecialRect ("UpperMiddle"), -1);
		classifierOff->updateSemi (image, detectionPatch, 1);
		classifierOff->updateSemi (image, detectionPatch, 1); 
		classifierOff->updateSemi (image,trackingPatches->getSpecialRect ("LowerMiddle"), -1);
		classifierOff->updateSemi (image, detectionPatch, 1);
		classifierOff->updateSemi (image, detectionPatch, 1); 
		classifierOff->updateSemi (image,trackingPatches->getSpecialRect ("MiddleLeft"), -1);
		classifierOff->updateSemi (image, detectionPatch, 1);
		classifierOff->updateSemi (image, detectionPatch, 1); 
		classifierOff->updateSemi (image,trackingPatches->getSpecialRect ("MiddleRight"), -1);
		classifierOff->updateSemi (image, detectionPatch, 1);
		classifierOff->updateSemi (image, detectionPatch, 1); 
	}
	
	for (int i = 0;i<iteration;++i)
	{
		
		classifier->updateSemi(image, detectionPatch, 1);
		classifier->updateSemi(image, detectionPatch, 1); 
		classifier->updateSemi(image, trackingPatches->getSpecialRect ("UpperLeft"), -1);
		
		classifier->updateSemi(image, detectionPatch, 1);
		classifier->updateSemi(image, detectionPatch, 1); 
		classifier->updateSemi(image, trackingPatches->getSpecialRect ("UpperRight"), -1);
		
		classifier->updateSemi(image, detectionPatch, 1);
		classifier->updateSemi(image, detectionPatch, 1); 
		classifier->updateSemi(image, trackingPatches->getSpecialRect ("LowerLeft"), -1);
		
		classifier->updateSemi(image, detectionPatch, 1);
		classifier->updateSemi(image, detectionPatch, 1); 
		classifier->updateSemi(image, trackingPatches->getSpecialRect ("LowerRight"), -1);

		classifier->updateSemi (image,trackingPatches->getSpecialRect ("UpperMiddle"), -1);
		classifier->updateSemi (image, detectionPatch, 1);
		classifier->updateSemi (image, detectionPatch, 1);

		classifier->updateSemi (image,trackingPatches->getSpecialRect ("LowerMiddle"), -1);
		classifier->updateSemi (image, detectionPatch, 1);
		classifier->updateSemi (image, detectionPatch, 1);

		classifier->updateSemi (image,trackingPatches->getSpecialRect ("MiddleLeft"), -1);
		classifier->updateSemi (image, detectionPatch, 1);
		classifier->updateSemi (image, detectionPatch, 1);

		classifier->updateSemi (image,trackingPatches->getSpecialRect ("MiddleRight"), -1);
		classifier->updateSemi (image, detectionPatch, 1);
		classifier->updateSemi (image, detectionPatch, 1);
	}
	

// 	float confidence = classifierOff->eval(image,detectionPatch)/classifierOff->getSumAlpha();
// 	if (confidence < 0.7f) //below some threshold
// 	{
// 		//reinit new trackers
// 		//copy validROI
// 		Size trackedSize;
// 		trackedSize = validROI;
// 		reInit(image,detectionPatch,trackedSize,trackingPatches);
// 	}

	delete trackingPatches;

	return iteration;
}

bool SemiBoostingTracker::TrackSearchRegion(ImageRepresentation *image, const cv::Rect &trackedPatch, float minMargin, 
	cv::Rect &result, std::vector<cv::Rect> &posRect, std::vector<float> &posWeight)
{
	assert(tracking_lost);
	Rect tracked_patch;
	tracked_patch = trackedPatch;
	Size trackingRectSize;
	trackingRectSize = tracked_patch;
	
	Rect searchingRegion;
	searchingRegion = searchRegion;

	Patches *trackingPatches = new PatchesRegularScan(searchingRegion,validROI,trackingRectSize,0.9f);
	//image->setNewROI(searchedRegion);
	detector->classifySmooth(image,trackingPatches,minMargin);

	if (detector->getNumDetections() <=0 )
	{
		return false;
	}

	Rect rect_result = trackingPatches->getRect (detector->getPatchIdxOfBestDetection());
	result.x = rect_result.left;
	result.y = rect_result.upper;
	result.width = rect_result.width;
	result.height = rect_result.height;
	float confidence = detector->getConfidenceOfBestDetection ();

	//Store all the detections
	int numDetection = detector->getNumDetections();
	posWeight.reserve(numDetection);
	posRect.reserve(numDetection);

	for (int i = 0; i<numDetection; ++i)
	{
		Rect d = trackingPatches->getRect(detector->getPatchIdxOfDetection(i));
		float w = detector->getConfidenceOfDetection(i) - minMargin;//convert to > 0
		posWeight.push_back(w);
		posRect.push_back(d.ConvertToRect());
	}

	//draw picture to show results
	//DrawConfidenceMap(image->frame(searchRegion),searchingRegion,trackingPatches);

	delete trackingPatches;
	return true;
}

/************************************************************************/
/* Draw confidence map for patches                                      */
/************************************************************************/
void SemiBoostingTracker::DrawConfidenceMap(const cv::Mat &ROI, const Rect searchRegion, Patches *patches)
{
	cv::Mat imgToWrite;
	ROI.copyTo(imgToWrite);
	int num = patches->getNum();
	float *conf = new float[num];
	float min_conf = FLT_MAX;
	float max_conf = -FLT_MAX;

	for (int i = 0;i<num;++i)
	{
		float confidence = detector->getConfidence(i);
		conf[i] = confidence;
		if (confidence>max_conf)
		{
			max_conf = confidence;
		}
		if (confidence<min_conf)
		{
			min_conf = confidence;
		}
	}

	//map to [0,1]
	float gap_neg, gap_pos;
	gap_neg = gap_pos = 1;
	if (max_conf > 0)
	{
		gap_neg = -min_conf;
		gap_pos = max_conf;
	}else
	{
		gap_neg = max_conf - min_conf;
	}

	int x = searchRegion.left;
	int y = searchRegion.upper;
	int stepCol = patches->getStepCol();
	int stepRow = patches->getStepRow();

	for (int i = 0;i<num;++i)
	{
		Rect patch = patches->getRect(i);
		uchar color[3] = {0u};
		if (conf[i]>0)//positive
		{
			conf[i] = (conf[i] - min_conf)/gap_pos;
			color[2] = uchar(255u/2*conf[i])+255u/2;//[255/2,255]
			color[0] = uchar(255u/2*(1-conf[i]));//[0,255/2]
		}else//negative
		{
			conf[i] = (conf[i] - min_conf)/gap_neg;
			color[0] = uchar(255u/2*(1-conf[i]))+255u/2;//[255/2,255]
			color[2] = uchar(255u/2*conf[i]);//[0,255u/2]
		}
		
		int center_x = patch.left - x + patch.width/2;
		int center_y = patch.upper - y + patch.height/2;
		for (int xx = center_x-stepCol/2;xx<=center_x+stepCol/2;++xx)
		{
			for (int yy = center_y-stepRow/2;yy<=center_y+stepRow/2;++yy)
			{
				cv::Vec3b &val = imgToWrite.at<cv::Vec3b>(yy,xx);
				val.val[0] = uchar(val.val[0]*0.5 + color[0]*0.5);
				val.val[1] = uchar(val.val[1]*0.5 + color[1]*0.5);
				val.val[2] = uchar(val.val[2]*0.5 + color[2]*0.5);
			}
		}
	}
	cv::imwrite("confMap.jpg",imgToWrite);
	delete[] conf;
}

/************************************************************************/
/* Check validity of tracked patch                                      */
/************************************************************************/
bool SemiBoostingTracker::checkValidTrackPatch(Rect trackedPatch)
{
	if (trackedPatch.upper<validROI.upper || trackedPatch.left<validROI.left || 
		trackedPatch.width + trackedPatch.left > validROI.width + validROI.left ||
		trackedPatch.height + trackedPatch.upper > validROI.height + validROI.upper)
	{
		return false;
	}else
	{
		return true;
	}
}

/************************************************************************/
/* Reinit the trackers. When prior classifier is confident but wrong    */
/************************************************************************/
void SemiBoostingTracker::reInit(ImageRepresentation *image, Rect trackedPatch, Size validROI, Patches *patches)
{
	Clear();
	initTracking(image,trackedPatch, validROI, patches);
}

/************************************************************************/
/* Perform supervised training. Train both prior and tracker            */
/************************************************************************/
int SemiBoostingTracker::SupervisedTraining(ImageRepresentation *image, const cv::Rect &region, Size validROI, bool one_shot)
{
	Rect trackedPatch;
	trackedPatch = region;
	//check self-validity of target region
	if (!one_shot && !checkValidTrackPatch(trackedPatch))
	{
		return -1;
	}
	int iteration;
	if (one_shot)
	{
		assert(!available);
		iteration = initTracking(image, trackedPatch, validROI,NULL);
	}else
	{
		assert(available);
		iteration = updateOn(image,trackedPatch);
	}
	return iteration;
}

/************************************************************************/
/* Perform semi-supervised training guided by prior.Perform update when the max confidence is above some threshold*/
/************************************************************************/
void SemiBoostingTracker::SemiSupervisedTraining(ImageRepresentation *image, const cv::Rect &region)
{
	assert(available);
	Rect trackedPatch;
	trackedPatch = region;
	//check self-validity of target region
	if (!checkValidTrackPatch(trackedPatch))
	{
		return;
	}

	Rect trackingROI = getTrackingROI(trackedPatch, 3.0f);
	Size trackedPatchSize;

	trackedPatchSize = trackedPatch;
	Patches* trackingPatches = new PatchesRegularScan(trackingROI, validROI, trackedPatchSize, 0.01f);

	float off, priorConfidence;//Learn the surround environment and patch itself

	Rect tmp;

	tmp = trackingPatches->getSpecialRect ("UpperLeft");
	off = classifierOff->eval(image, tmp)/classifierOff->getSumAlpha();
	classifier->updateSemi (image, tmp, off);

	priorConfidence = classifierOff->eval(image, trackedPatch)/classifierOff->getSumAlpha();
	classifier->updateSemi (image, trackedPatch, priorConfidence);

	tmp = trackingPatches->getSpecialRect ("LowerLeft");
	off = classifierOff->eval(image, tmp)/classifierOff->getSumAlpha();
	classifier->updateSemi (image, tmp, off);

	priorConfidence = classifierOff->eval(image, trackedPatch)/classifierOff->getSumAlpha();
	classifier->updateSemi (image, trackedPatch, priorConfidence);

	tmp = trackingPatches->getSpecialRect ("UpperRight");
	off = classifierOff->eval(image, tmp)/classifierOff->getSumAlpha();
	classifier->updateSemi (image, tmp, off);	

	priorConfidence = classifierOff->eval(image, trackedPatch)/classifierOff->getSumAlpha();
	classifier->updateSemi (image, trackedPatch, priorConfidence);

	tmp = trackingPatches->getSpecialRect ("LowerRight");
	off = classifierOff->eval(image, tmp)/classifierOff->getSumAlpha();
	classifier->updateSemi (image, tmp, off);

	priorConfidence = classifierOff->eval(image, trackedPatch)/classifierOff->getSumAlpha();
	classifier->updateSemi (image, trackedPatch, priorConfidence);

	tmp = trackingPatches->getSpecialRect("UpperMiddle");
	off = classifierOff->eval(image, tmp)/classifierOff->getSumAlpha();
	classifier->updateSemi (image, tmp, off);

	priorConfidence = classifierOff->eval(image, trackedPatch)/classifierOff->getSumAlpha();
	classifier->updateSemi (image, trackedPatch, priorConfidence);

	tmp = trackingPatches->getSpecialRect("LowerMiddle");
	off = classifierOff->eval(image, tmp)/classifierOff->getSumAlpha();
	classifier->updateSemi (image, tmp, off);

	priorConfidence = classifierOff->eval(image, trackedPatch)/classifierOff->getSumAlpha();
	classifier->updateSemi (image, trackedPatch, priorConfidence);

	tmp = trackingPatches->getSpecialRect("MiddleLeft");
	off = classifierOff->eval(image, tmp)/classifierOff->getSumAlpha();
	classifier->updateSemi (image, tmp, off);

	priorConfidence = classifierOff->eval(image, trackedPatch)/classifierOff->getSumAlpha();
	classifier->updateSemi (image, trackedPatch, priorConfidence);

	tmp = trackingPatches->getSpecialRect("MiddleRight");
	off = classifierOff->eval(image, tmp)/classifierOff->getSumAlpha();
	classifier->updateSemi (image, tmp, off);

	priorConfidence = classifierOff->eval(image, trackedPatch)/classifierOff->getSumAlpha();
	classifier->updateSemi (image, trackedPatch, priorConfidence);

	delete trackingPatches;
}

void SemiBoostingTracker::SupervisedNegative(ImageRepresentation *image, const cv::Rect &region)
{
	Rect negative;
	negative = region;
	//check self-validity of target region
	if (!checkValidTrackPatch(negative))
	{
		return;
	}
	for (int i = 0; i < 2; ++i)
	{
		classifierOff->update(image, negative, -1);
		classifier->updateSemi(image, negative, -1);
	}
}

/************************************************************************/
/* Set image representation as current frame (general integral image)   */
/************************************************************************/
void SemiBoostingTracker::SetImageRepresentation(ImageRepresentation *image, const cv::Mat &frame)
{
	assert(image!=NULL);
	uchar *grayImg = getGrayImage(frame);
	image->setNewImage(grayImg);
	delete grayImg;
}