/*A program to test the whole process*/
#include "TrackAssociation.h"
#include <ctime>
#define RESIZE 0
#define START_FRAME 1
using namespace std;

void Tracking();

int image_width, image_height;

int main()
{
	srand((unsigned int)time(NULL));
	//long totaltime;
	//clock_t start,finish;
	//start=clock();
	Tracking();
	//finish=clock();
	//totaltime=finish-start;
	//printf("Total time elapsed: %ldms\n%gms per frame!\n",totaltime,(float)totaltime/NUMFRAMES);
	system("pause");
	return EXIT_SUCCESS;
}

void Tracking()
{
	TrackAssociation * trackAssociation = new TrackAssociation();//To global variable
	//Some instances we used in object tracking and are declared as extern in related files
	ObjSeg *objseg = trackAssociation->GetObjSeg();
	
	cv::Mat img1,img2;//two images
	vector<cv::Mat> pyr1,pyr2;//two pyramid images
	cv::Mat ROI;
	ROI.create(img1.rows,img1.cols,cv::DataType<uchar>::type);

	char framedata[100];
	KLT klt;//KLT tracking
	KLT_TrackingContext tc = &klt;
	KLT_FeatureList fl;

	objseg->tc = tc;
	//sprintf(framedata,"../../HOG Train/dataStd/image0.jpg");
	sprintf(framedata,"../View_001/frame0.jpg");
	img1 = cv::imread(std::string(framedata));

#if RESIZE
	cv::Mat img1_temp;
	cv::resize(img1,img1_temp,cv::Size(1280,960));
	img1.release();
	img1 = img1_temp;
#endif

	trackAssociation->StoreToFramePool(img1,0);

	//Set the frame size (global variables)
	image_width = img1.cols;
	image_height = img1.rows;
	objseg->ResetOccupationMap();

	//Just init for the first frame
	fl = KLTCreateFeatureList(NUMFLFEATURES);
	//Feature point detection stage, features detected at whole screen; Feature tracking stage, features detected at border
	
	//tc->UseBackgroundModel(img1);//enable background model

	tc->ProcessNxtImg(img1,pyr1); //load first pyramid image

	objseg->SetFrameNO(START_FRAME-1); //starting frame
	
	//tracking features across whole frame
	tc->SelectGoodFeaturesToTrack(img1, fl, ROI); 

	int length = KLTCountRemainingFeatures(fl);

	for(int i=0;i<length;i++)
	{
		objseg->GetMotionFeature()->AddNewTrack(fl->feature[i], i, START_FRAME-1);// At the 0th starting frame
	}

#if WRITETEST
	sprintf(framedata,"track%d.txt",0);
	objseg->WriteTrackToFile(framedata);
#endif

	std::list<int> misslist;

	bool justTracking = true; //just tracking existing features
	bool appearanceChecking = false; // appearance tracking 
	bool external_detection = false; // tracking with external detection
	int detectionFrameNO(0); //last clustering frame NO
	 
	for(int i = START_FRAME; i < NUMFRAMES; i++)
	{
		printf("frame %d\n",i);

		objseg->SetFrameNO(i);

		//sprintf(framedata, "../../HOG Train/dataStd/image%d.jpg", i);
		sprintf(framedata,"../View_001/frame%d.jpg",i);
		img2 = cv::imread(std::string(framedata));

#if RESIZE
		cv::Mat img2_temp;
		cv::resize(img2,img2_temp,cv::Size(1280,960));
		img2.release();
		img2 = img2_temp;
#endif

		tc->ProcessNxtImg(img2,pyr2); //construct pyramid image for the next image

		trackAssociation->StoreToFramePool(img2,i);

		//track features at img2 (next image)
		tc->TrackFeatures(pyr1,pyr2,fl);
		
		// Reset occupation map to avoid duplicates in the same location
		objseg->ResetOccupationMap();
		// According to the tracking result, delete feature tracks or append new feature to the list
		objseg->Delete_Append_FeatureTrackSet(fl, misslist);

		int originalLength = fl->nFeatures;
		int diffFrame = i - detectionFrameNO;

		if(diffFrame == 2 || diffFrame==5)
		{
			justTracking = false; // feature point redetection
		}
		
		if(diffFrame == 3 || diffFrame == 6 || i % NON_KEYFRAME_GAP == 0)
		{
			appearanceChecking = true; // perform appearance checking
			if(i % NON_KEYFRAME_GAP == 0)
			{
				external_detection = true;
				detectionFrameNO = i;
			}
		}

		//Perform feature points re-detection
		fl = objseg->ReplaceFeatureList(img2, fl, misslist, MIN_DIST, justTracking);
		justTracking = true;

#if WRITETEST
		char fl_file[50];
		sprintf(fl_file,"featureList/fl%d.jpg",i);
		fl->WriteToImage(fl_file,img2);
#endif

		objseg->Add_FeatureTrackSet(fl,originalLength,misslist);
		if(appearanceChecking)
		{
			//buffer->LoadMostCurImage(img2);
			//buffer->IncreaseLoadPointer();//Increase load pointer

			objseg -> CalAvgVecBwtFrames(trackAssociation->GetPoolSize());

			trackAssociation -> ObjectTrackingAtKeyFrame(i, img2, external_detection);

			// system state clearance
			external_detection = false;
			objseg -> ResetMappingArray();//Called after we have created clusters
			trackAssociation -> ClearFramePool();
			appearanceChecking = false;
		}

		if(i % SHRINKFEATURE==0) //To shrink the feature list to save search time
			ShrinkFeatureList(fl);
		misslist.clear(); 
		pyr2.clear();

#if WRITETEST
		sprintf(framedata,"track%d.txt",i);
		objseg->WriteTrackToFile(framedata);
#endif

	}
	KLTFreeFeatureList(fl);
	delete trackAssociation;
}