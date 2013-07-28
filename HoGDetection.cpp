#include "HoGDetection.h"
#include "stdfunc.h"
using namespace cv;
using namespace std;
inline void ModifyImage(Mat &image, float value, size_t i, size_t j, float alpha, const Vec3b &bottom, const Vec3b &middle, const Vec3b &top);
HoGDetection::HoGDetection(const string &vectorFileName):
vectorFileName(vectorFileName)
{
	hog = new HOGDescriptor(arg.win_size, arg.block_size, arg.block_stride, arg.cell_size,
		arg.nbins, 1, arg.win_sigma, HOGDescriptor::L2Hys, arg.threshold_L2hys, arg.gamma_correction, arg.nlevels);
	LoadDetectorFromFile();
	hog->setSVMDetector(detector);
	full_body.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
}

HoGDetection::HoGDetection(const Arg &args,const string &vectorFileName):
vectorFileName(vectorFileName),arg(args)
{
	hog = new HOGDescriptor(arg.win_size, arg.block_size, arg.block_stride, arg.cell_size,
		arg.nbins, 1, arg.win_sigma, HOGDescriptor::L2Hys, arg.threshold_L2hys, arg.gamma_correction, arg.nlevels);
	LoadDetectorFromFile();
	hog->setSVMDetector(detector);

	full_body.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

}

HoGDetection::~HoGDetection(void)
{
	delete hog;
}

/*Load Detector from file*/
void HoGDetection::LoadDetectorFromFile()
{
	printf("Loading detector from %s...\n",vectorFileName.c_str());
	assert(!vectorFileName.empty());//must not be empty
	ifstream ifs(vectorFileName);
	if(ifs.fail())
	{
		char err_msg[50];
		sprintf(err_msg,"Cannot open file %s\n",vectorFileName);
		cvError(-1, __FUNCTION__, err_msg, __FILE__, __LINE__);
		system("pause");
		exit(-1);
	}
	int curLine(0);//current Line number
	//svm vector begins from the third line
	int featureIndex;
	while(ifs.good()&&++curLine<=2)
	{
		string line;
		getline(ifs,line);
		istringstream iss(line);//string stream binded
		string word;
		while(iss>>word)//loaded to word till end
		{
			if(word.compare("#")==0||word.compare("")==0)
				break;//end of line or file
			else
				featureIndex = atoi(word.c_str());
		}
	}
	detector.reserve(featureIndex+1);//reserve the space
	int count(0);//count of current feature
	while(ifs.good())
	{
		string line;
		getline(ifs,line);//indeed, there is only one line left
		istringstream iss(line);
		string word;
		while(iss>>word)//iterate all the words
		{
			if(word.compare("#")==0||word.compare("")==0)
				break;//end of line
			detector.push_back(atof(word.c_str()));//store it to descriptor
			count++;
		}
		assert(count<=featureIndex+1);
	}
	ifs.close();//release resource
	printf("Detector loading done!\n");
}


/*Multisacle detection at image*/
void HoGDetection::detect(const Mat &image)
{
	found.clear();
	found_filtered.clear();
	weight.clear();
	weight_filtered.clear();
	assert(!image.empty());//clear the original result
	hog->detectMultiScale(image,found,weight,arg.hit_threshold,arg.win_stride,
			arg.padding,arg.scale0,arg.group_threshold);//Multi-scale detection, note the padding around


	found_full_body.clear();
	found_full_body_filtered.clear();
	full_body.detectMultiScale(image,found_full_body,weight_full_body,-0.2,arg.win_stride,
			arg.padding,1.05,2);

	size_t a,b;
	for(a = 0; a < found_full_body.size(); a++ )
	{
		Rect r = found_full_body[a];
		double w = weight_full_body[a];
		for(b = 0; b < found_full_body.size(); b++ )
			if( b != a && (r & found_full_body[b]) == r)
				break;
		    if( b == found_full_body.size() )
			{
			    found_full_body_filtered.push_back(r);
				weight_full_body_filtered.push_back(w);
			}
	}

	for( a = 0; a < found_full_body_filtered.size(); a++ )
	{
		Rect &r = found_full_body_filtered[a];
		// the HOG detector returns slightly larger rectangles than the real objects.
		// so we slightly shrink the rectangles to get a nicer output.
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
	}

	//Filter out some results
	size_t i, j;
	for( i = 0; i < found.size(); i++ )
	{
		Rect r = found[i];
		double w = weight[i];
		for( j = 0; j < found.size(); j++ )
			if( j != i && (r & found[j]) == r)
				break;
		    if( j == found.size() )
			{
			    found_filtered.push_back(r);
				weight_filtered.push_back(w);
			}
	}

	for( i = 0; i < found_filtered.size(); i++ )
	{
		Rect &r = found_filtered[i];
		// the HOG detector returns slightly larger rectangles than the real objects.
		// so we slightly shrink the rectangles to get a nicer output.
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
	}
}

void HoGDetection::drawResult(cv::Mat &img)
{
	for(size_t i = 0; i < found_filtered.size(); i++ )
	{
		Rect r = found_filtered[i];
		rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 1);
		//ostringstream os;
		//os<<scores[i];
		//putText(img_to_show,os.str(),Point(r->tl().x,(r->tl().y+r->br().y)/2),FONT_HERSHEY_PLAIN,1.0,Scalar(0,255,255));//show text
	}


	for(size_t i = 0; i < found_full_body_filtered.size(); i++ )
	{
		Rect r = found_full_body_filtered[i];
		rectangle(img, r.tl(), r.br(), cv::Scalar(255,255,0), 1);
		//ostringstream os;
		//os<<scores[i];
		//putText(img_to_show,os.str(),Point(r->tl().x,(r->tl().y+r->br().y)/2),FONT_HERSHEY_PLAIN,1.0,Scalar(0,255,255));//show text
	}
}

void HoGDetection::drawResult(const Mat &image,const string &fileName)//'image' is the cropped sub-image 
{
	Mat img_to_show;
	image.copyTo(img_to_show);
	for(size_t i = 0; i < found_filtered.size(); i++ )
	{
		Rect r = found_filtered[i];
		// the HOG detector returns slightly larger rectangles than the real objects.
		// so we slightly shrink the rectangles to get a nicer output.
		rectangle(img_to_show, r.tl(), r.br(), cv::Scalar(0,255,0), 1);
		//ostringstream os;
		//os<<scores[i];
		//putText(img_to_show,os.str(),Point(r->tl().x,(r->tl().y+r->br().y)/2),FONT_HERSHEY_PLAIN,1.0,Scalar(0,255,255));//show text
	}


	for(size_t i = 0; i < found_full_body_filtered.size(); i++ )
	{
		Rect r = found_full_body_filtered[i];
		// the HOG detector returns slightly larger rectangles than the real objects.
		// so we slightly shrink the rectangles to get a nicer output.
		rectangle(img_to_show, r.tl(), r.br(), cv::Scalar(255,255,0), 1);
		//ostringstream os;
		//os<<scores[i];
		//putText(img_to_show,os.str(),Point(r->tl().x,(r->tl().y+r->br().y)/2),FONT_HERSHEY_PLAIN,1.0,Scalar(0,255,255));//show text
	}

	if(!imwrite(fileName,img_to_show))//write the image to disk
	{
		char err_msg[50];
		sprintf(err_msg,"Cannot write image %s\n",fileName);
		cvError(-1, __FUNCTION__, err_msg, __FILE__, __LINE__);
		system("pause");
		exit(-1);
	}
}

void HoGDetection::ComputeScores(const Mat &img, bool init, double &normalization_term, bool isPredictTrustable)//boundingBox is the original area without being enlarged
{
	scores.clear();
	prediction_matching_scores.clear();
	normalization_term=0;//sum of all the confidence score to allow normalization
	if(init)//init
	{
		const static double alpha_init=5;//how much we rely on initial detection 
		//const static double scale_mean=0.09;//relative to the original size of bounding box (likely to be a whole person image)
		const static double scale_std=0.5;//how much we believe the scale info
		for(size_t i=0;i<found_filtered.size();++i)
		{
			Rect r = found_filtered[i];
			double w = weight_filtered[i];//weight of ith detection
			//for init case, confidence of each detection is also proportional to its scale
			//double width_norm=r.width/(double)img.cols;
			//double height_norm=r.height/(double)img.rows;
			//double scale_value=width_norm*height_norm;
			//scale_value=log(scale_mean/scale_value);//punish more on larger scale detection
			const double mean=0;
			//Ideally, we want our scale to be 'scale_mean' of original bounding box
			//double proScale=stdfunc::GaussianDistribution(&mean,&scale_std,1,&scale_value);
			double score_i=alpha_init*w;
			scores.push_back(score_i);//we have the score of each detection
			normalization_term+=score_i;
			//Note the final value should be normalized by summation of all the weight
		}
	}else//We have predictions
	{
		assert(prediction.area()!=0);
		const static double alpha_predict=10;//how much we rely on detection
		const static double scale_std_predict=0.5;//how much we believe scale info
		const static double norm_scale[]={1.0,1.0};
		const static double norm_scale_std[]={scale_std_predict,scale_std_predict};
		const static double mean_dist = 0;
		const static double std_dist = 1.0;

		//The scale should be similar with the prediction
		double center_prediction[]={(prediction.tl().x+prediction.br().x)/2.0,(prediction.tl().y+prediction.br().y)/2.0};
		double predict_scale=prediction.width<prediction.height?prediction.width:prediction.height;//smaller one of prediction size
		for(size_t i=0;i<found_filtered.size();++i)
		{
			Rect r = found_filtered[i];
			double w = weight_filtered[i];//weight of ith detection
			w=w*w*3;//mapped to higher
			double scale_i[]={r.width/(double)prediction.width,r.height/(double)prediction.height};//scale of ith detection
			double proScale=1;
			//we consider scale factor
			proScale=stdfunc::GaussianDistribution(norm_scale,norm_scale_std,2,scale_i,false);
			//consider location factor (center distance normalized by min of scale)
			double center_r[]={(r.tl().x+r.br().x)/2.0,(r.tl().y+r.br().y)/2.0};//center of detection
			double found_scale = r.width<r.height ? r.width:r.height;//min of found
			double norm_scale = predict_scale<found_scale?predict_scale:found_scale;
			double dist2predict = sqrt(((center_r[0]-center_prediction[0])/norm_scale)*((center_r[0]-center_prediction[0])/norm_scale)+
				((center_r[1]-center_prediction[1])/norm_scale)*((center_r[1]-center_prediction[1])/norm_scale));
			double proDist=stdfunc::GaussianDistribution(&mean_dist,&std_dist,1,&dist2predict,false);//pro location
			double predict_matching = proScale*proDist;
			double score_i = alpha_predict*w*predict_matching;
			prediction_matching_scores.push_back(predict_matching);//store the matching score to prediction to judge which is which
			scores.push_back(score_i);
			normalization_term+=score_i;
		}
	}
}

/*Extract subimage by boundingBox of motion cluster. Detection region. But all detection should fall into ROI*/
void HoGDetection::SetBoundingBox(const Mat &image, Mat &sub_img, int relative_pos[2], const Rect &ROI)
{
	size_t width_b = ROI.width;
	size_t height_b = ROI.height;
	Point center_b(ROI.x+width_b/2,ROI.y+height_b/2);//center of ROI

	while(width_b<arg.win_size.width*4)//still smaller than the min requirement
		width_b += arg.win_size.width;//enlarge by 2
	while(height_b<arg.win_size.height*6)
		height_b += arg.win_size.height;

	int xmin_b=center_b.x-width_b/2;
	if(xmin_b<0)
		xmin_b=0;
	int ymin_b=center_b.y-height_b/2;
	if(ymin_b<0)
		ymin_b=0;
	int xmax_b=center_b.x+width_b/2;
	if(xmax_b>=image.cols)
		xmax_b=image.cols-1;
	int ymax_b=center_b.y+height_b/2;
	if(ymax_b>=image.rows)
		ymax_b=image.rows-1;
	relative_pos[0] = xmin_b;
	relative_pos[1] = ymin_b;//left-top corner position in global image
	Point center_final(cvRound((xmin_b+xmax_b)/2.0),cvRound((ymin_b+ymax_b)/2.0));
	Size size_final(xmax_b-xmin_b,ymax_b-ymin_b);
	getRectSubPix(image,size_final,center_final,sub_img);//extract the sub-window
}

/*Set predicted result and its matching score
Called when data association is done*/
void HoGDetection::SetPredictions(const Rect &prediction, double matching_score)
{
	this->prediction = prediction;//copy
	this->matching_score=matching_score;
}

/*Generate posterior probability*/
void HoGDetection::ComputePostProb(bool init, const Mat &image, bool isPredictTrustable, 
	bool enable_visualization, const string &fileName, size_t missing_frames)
{
	const static double std_dist=1.0;
	Mat result(image.rows,image.cols,DataType<float>::type,Scalar(0));//result matrix for visualization, filled with 0
	float max_score=-1;
	float min_score=FLT_MAX;
	const static float threshold = 0.9f;
	double pos_mean[2];
	/*我们先visualize看效果
	each element of matrix should fall into [0,1]
	*/
	if(init && found_filtered.size()!=0 || !isPredictTrustable)//init condition & have detection//In this case we do not have any hint
	{
		const static double gama_pos = 5;//how much we rely on positional priori
		const static double pos_std[]={1.5,0.5};//std of distance to priori
		double norm_term;//normalization term
		//Compute confidence for each detection
		ComputeScores(image,true,norm_term);
		//additionally compute confidence for prior location
		//User defined linear function to estimate position of head
		double ratio=image.cols/(double)image.rows;
		double percentage_y;
		percentage_y = 0.2;

		pos_mean[0]=0.5*image.cols;
		pos_mean[1]=percentage_y*image.rows;//prefered relative position of the center
		double mean=0,std=1;
		norm_term += gama_pos;
		double norm_factor=image.rows<image.cols?image.rows:image.cols;
		for(size_t i=0; i<image.cols; ++i)
		{
			for(size_t j=0; j<image.rows; ++j)
			{
				float score_ij(0);//score at (i,j) pixel
				for(size_t k=0; k<found_filtered.size(); ++k)//for each there is pulse
				{
					double center_k[]={found_filtered[k].x+found_filtered[k].width/2.0,found_filtered[k].y+found_filtered[k].height/2.0};//center of kth detection
					//distance is normalized by the image size
					double dist=sqrt(((i-center_k[0])/norm_factor)*((i-center_k[0])/norm_factor)+
						((j-center_k[1])/norm_factor)*((j-center_k[1])/norm_factor));//E-distance to the center of detection
					const double mean=0;//zero-mean
					double dis_k=stdfunc::GaussianDistribution(&mean,&std_dist,1,&dist);
					float scoreij_on_k=(float)(scores[k]*dis_k);//our confidence on the detection multiplied by Gaussian distance
					score_ij+=scoreij_on_k;
				}
				
				//double dis2prior=sqrt(((i-pos_mean[0])/norm_factor)*((i-pos_mean[0])/norm_factor)
					//+((j-pos_mean[1])/norm_factor)*((j-pos_mean[1])/norm_factor));//distance to the prior location
				const double mean=0;
				double dist_x=fabs((i-pos_mean[0])/norm_factor);//distance in x 
				double dist_y=fabs((j-pos_mean[1])/norm_factor);//distance in y

				double dis_px=stdfunc::GaussianDistribution(&mean,pos_std,1,&dist_x);//we consider y is more important than x
				double dis_py=stdfunc::GaussianDistribution(&mean,pos_std+1,1,&dist_y);
				score_ij+=(float)(gama_pos*dis_px*dis_py);
				float normalized_score_ij=score_ij/(float)norm_term;//normalized term
				if(normalized_score_ij<min_score)
					min_score=normalized_score_ij;
				if(normalized_score_ij>max_score)
					max_score=normalized_score_ij;
				result.at<float>(j,i)=normalized_score_ij;//set the value (jth row, ith col)
			}
		}

		//do min-max normalization
		for(size_t i=0;i<image.cols;++i)
		{
			for(size_t j=0;j<image.rows;++j)
			{
				float value=result.at<float>(j,i);
				float new_value=(value-min_score)/(max_score-min_score);
				result.at<float>(j,i)=new_value;
			}
		}
	}else if(!init&&prediction.area()!=0)//we have predictions
	{
		double norm_term;//normalization term
		//Compute confidence for each detection
		ComputeScores(image,false,norm_term);
		const static double beta_predict = 2;//how much we rely on prediction
		double std_pre=1.5;
		if(!isPredictTrustable)//larger std
			std_pre=2;
		const double std_predict=std_pre*(missing_frames+1);//std should be proportional to # of missing frames
		
		double pro_prediction=beta_predict*matching_score*matching_score;//confidence on predicted result should be proportional to tracking matching score
		norm_term+=pro_prediction;
		double norm_factor=image.rows<image.cols?image.rows:image.cols;
		for(size_t i=0;i<image.cols;++i)//all the pixels in detection region
		{
			for(size_t j=0;j<image.rows;++j)
			{
				float score_ij(0);
				for(size_t k=0; k<found_filtered.size(); ++k)//for each detection there is pulse
				{
					double center_k[]={found_filtered[k].x+found_filtered[k].width/2.0,found_filtered[k].y+found_filtered[k].height/2.0};//center of kth detection
					//distance is normalized by the image size
					double dist=sqrt(((i-center_k[0])/norm_factor)*((i-center_k[0])/norm_factor)+
						((j-center_k[1])/norm_factor)*((j-center_k[1])/norm_factor));//E-distance to the center of detection
					const double mean=0;//zero-mean
					double dis_k = stdfunc::GaussianDistribution(&mean,&std_dist,1,&dist);
					float scoreij_on_k=(float)(scores[k]*dis_k);//our confidence on the detection multiplied by Gaussian distance
					score_ij+=scoreij_on_k;
				}
				double center_predict[]={prediction.x+prediction.width/2.0,prediction.y+prediction.height/2.0};
				//distance is normalized by size of cropped region
				double dist_predict=sqrt(((i-center_predict[0])/norm_factor)*((i-center_predict[0])/norm_factor)+
					((j-center_predict[1])/norm_factor)*((j-center_predict[1])/norm_factor));
				const double mean=0;
				double dist_p=stdfunc::GaussianDistribution(&mean,&std_predict,1,&dist_predict);
				score_ij+=(float)(pro_prediction*dist_p);//confidence on prediction multiplied by distance
				float normalized_score_ij=score_ij/(float)norm_term;//normalized term
				if(normalized_score_ij<min_score)
					min_score=normalized_score_ij;
				if(normalized_score_ij>max_score)
					max_score=normalized_score_ij;
				result.at<float>(j,i)=normalized_score_ij;//set the value
			}
		}

		//do min-max normalization
		for(size_t i=0;i<image.cols;++i)
		{
			for(size_t j=0;j<image.rows;++j)
			{
				float value=result.at<float>(j,i);
				float new_value=(value-min_score)/(max_score-min_score);
				result.at<float>(j,i)=new_value;
			}
		}
	}
	
	list<Point> initials, maxima;
	if(init&&found_filtered.size()!=0)
	{
		//initial points are detections & predictions
		for(vector<Rect>::const_iterator citer=found_filtered.cbegin();citer!=found_filtered.cend();citer++)
		{
			Rect r= (*citer);
			Point p(r.x+r.width/2,r.y+r.height/2);
			initials.push_back(p);
		}
		initials.push_back(Point(pos_mean[0],pos_mean[1]));
		HillClimbing(result,initials,threshold,maxima);
		//The following code transform a point to a rect

	}else if(!init&&prediction.area()!=0)//prediction stage
	{
		//for each detection & prediction
		for(vector<Rect>::const_iterator citer=found_filtered.cbegin();citer!=found_filtered.cend();citer++)
		{
			Rect r= (*citer);
			Point p(r.x+r.width/2,r.y+r.height/2);
			initials.push_back(p);
		}
		Point p(prediction.x+prediction.width/2,prediction.y+prediction.height/2);
		initials.push_back(p);
		HillClimbing(result,initials,threshold,maxima);
	}

	//GetFinalDetection(init,maxima,final_detection,image,isPredictTrustable);

	//After normal process, we retain detections that owns a body detection associated with
	if(enable_visualization)//visualize the result
	{
		//VisualizeMatrix(image,result,fileName,final_detection);
	}
}

/*visualize the matrix on the image*/
void HoGDetection::VisualizeMatrix(const Mat &image, const Mat &result, const string &fileName, const list<Rect> &maxima_detection)
{
	assert(image.rows==result.rows&&image.cols==result.cols);
	Mat img_to_show;
	image.copyTo(img_to_show);
	const static float alpha=0.5f;
	const static Vec3b bottom(255,0,0);//0 == total blue
	const static Vec3b middle(0,255,0);//0.5 == total green
	const static Vec3b top(0,0,255);//1 == total red
	for(size_t i=0;i<image.cols;++i)
	{
		for(size_t j=0;j<image.rows;++j)
		{
			float res=result.at<float>(j,i);//jth row. ith col
			ModifyImage(img_to_show,res,i,j,alpha,bottom,middle,top);
		}
	}

	for(list<Rect>::const_iterator citer=maxima_detection.cbegin();citer!=maxima_detection.cend();++citer)
	{
		rectangle(img_to_show, citer->tl(), citer->br(), cv::Scalar(255,255,0), 1);
	}

	if(!imwrite(fileName,img_to_show))
	{
		char err_msg[50];
		sprintf(err_msg,"Cannot write visulization image %s\n",fileName);
		cvError(-1, __FUNCTION__, err_msg, __FILE__, __LINE__);
		system("pause");
		exit(-1);
	}
}

/*Add color mapping according to the value to the image at (i,j)*/
inline void ModifyImage(Mat &image, float value, size_t i, size_t j, float alpha, const Vec3b &bottom, const Vec3b &middle, const Vec3b &top)
{
	const static Vec3b inter(0,255,255);
	assert(value>=0&&value<=1);//value must fall into [0 1]
	assert(alpha>=0&&alpha<=1);//alpha must fall into [0,1]
	assert(i<image.cols&&j<image.rows);
	Vec3b &intensity = image.at<Vec3b>(j,i);//retrieve the reference to the value (jth row, ith col)
	uchar blue = intensity.val[0];
	uchar green = intensity.val[1];
	uchar red = intensity.val[2];
	//get the value color by interporlation
	uchar value_blue,value_green,value_red;
	if(value>0.5)//since we focus on high-value area we further divide it 
	{
		value=(value-0.5)/0.5;
		if(value>0.5)
		{
			value=(value-0.5)/0.5;
			value_blue=(uchar)((1-value)*inter.val[0]+value*top.val[0]);
			value_green=(uchar)((1-value)*inter.val[1]+value*top.val[1]);
			value_red=(uchar)((1-value)*inter.val[2]+value*top.val[2]);
		}else
		{
			value=value/0.5;
			value_blue=(uchar)((1-value)*middle.val[0]+value*inter.val[0]);
			value_green=(uchar)((1-value)*middle.val[1]+value*inter.val[1]);
			value_red=(uchar)((1-value)*middle.val[2]+value*inter.val[2]);
		}
		
	}else
	{
		value=value/0.5;
		value_blue=(uchar)((1-value)*bottom.val[0]+value*middle.val[0]);
		value_green=(uchar)((1-value)*bottom.val[1]+value*middle.val[1]);
		value_red=(uchar)((1-value)*bottom.val[2]+value*middle.val[2]);
	}
	//blend the value with the image original intensity
	uchar new_blue=(uchar)(alpha*value_blue+(1-alpha)*blue);
	uchar new_green=(uchar)(alpha*value_green+(1-alpha)*green);
	uchar new_red=(uchar)(alpha*value_red+(1-alpha)*red);
	intensity.val[0]=new_blue;
	intensity.val[1]=new_green;
	intensity.val[2]=new_red;//modify
}

/*Local search algorithm to find local maxima*/
void HoGDetection::HillClimbing(const Mat &density, const list<Point> &initials, float threshold, list<Point> &maxima)
{
	const static int min_diff = 1;//search stops it does not improve
	const static size_t step = 1u;//step of each movement
	list<Point> temp_maxima;
	list<float> value;
	//we apply the naive HillClimbing algorithm and filter the final results by thresholding and combination
	//Usually we have mutlple initials. For each detections and predictions
	for(list<Point>::const_iterator citer=initials.cbegin();citer!=initials.cend();++citer)
	{
		//for each initial estimate
		int cur_x=citer->x;
		int cur_y=citer->y;
		if(cur_x<0||cur_x>=density.cols||cur_y<0||cur_y>=density.rows)//invalid estimates
			continue;
		int neigh_x, neigh_y, best_x,best_y;
		float up,down,left,right,best_value;
		float cur_value;
		do{
			//compare all the neighbours
			neigh_x=cur_x;
			neigh_y=cur_y-step;//up
			if(neigh_y<0)
				up=0;
			else
				up=density.at<float>(neigh_y,neigh_x);//neigh_y th row, neigh_x th column
			neigh_y=cur_y+step;//down
			if(neigh_y>=density.cols)
				down=0;
			else
				down=density.at<float>(neigh_y,neigh_x);

			if(up>=down)
			{
				best_value=up;
				best_x=cur_x;
				best_y=cur_y-step;
			}else
			{
				best_value=down;
				best_x=cur_x;
				best_y=cur_y+step;
			}

			neigh_x=cur_x-step;//left
			neigh_y=cur_y;
			if(neigh_x<0)
				left=0;
			else
				left=density.at<float>(neigh_y,neigh_x);

			if(left>=best_value)
			{
				best_value=left;
				best_x=cur_x-step;
				best_y=cur_y;
			}

			neigh_x=cur_x+step;//right
			if(neigh_x>=density.rows)
				right=0;
			else
				right=density.at<float>(neigh_y,neigh_x);
			//find the max of four neighbours
			if(right>=best_value)
			{
				best_value=right;
				best_x=cur_x+step;
				best_y=cur_y;
			}
			//compare with the current value
			cur_value=density.at<float>(cur_y,cur_x);
			if(best_value>cur_value)
			{
				cur_x=best_x;
				cur_y=best_y;
				cur_value=best_value;
			}else
			{
				break;//break with no violation
			}

		}while(true);
		temp_maxima.push_back(Point(cur_x,cur_y));
		value.push_back(cur_value);
	}
	
	//Filter the reuslts by thresholding and combination
	while(!temp_maxima.empty())
	{
		float dens_value=value.front();
		if(dens_value>threshold)//enough ensurance
		{
			Point max=temp_maxima.front();
			//combine the similar values
			bool find = false;
			for(list<Point>::const_iterator citer=maxima.cbegin();citer!=maxima.cend();++citer)//search for previous similar result
			{
				Point compare=(*citer);
				if(abs(compare.x-max.x)<=min_diff&&abs(compare.y-max.y)<=min_diff)//within a threshold
				{
					find=true;
					break;
				}
			}
			if(!find)
				maxima.push_back(max);
		}
		temp_maxima.pop_front();
		value.pop_front();
	}
}

/*Transform a point to a rect*/
void HoGDetection::GetFinalDetection(bool init, const list<Point> &maxima, list<Rect> &final_detection, const cv::Mat &image, bool isPredictTrustable)
{
	if(init)
	{
		const double dist_thresh = image.rows/3.0;
		//Get scale from nearest detection result but we use the detection result if proper
		vector<Rect>::const_iterator citer_f;
		int x,y;
		for(list<Point>::const_iterator citer_p=maxima.cbegin(); citer_p!=maxima.cend(); ++citer_p)
		{
			int min_dist=INT_MAX;
			int width=-1,height=-1;//derived width and height
			for(citer_f=found_filtered.cbegin();citer_f!=found_filtered.cend();++citer_f)
			{
				int center_x=citer_f->x+citer_f->width/2;
				int center_y=citer_f->y+citer_f->height/2;
				int dist=abs(center_x-citer_p->x)+abs(center_y-citer_p->y);
				if(dist < min_dist)
				{
					min_dist = dist;
					width = citer_f->width;
					height = citer_f->height;
					x = citer_f->x;
					y = citer_f->y;
				}
			}
			if(min_dist<dist_thresh)//found a proper detection and use the detection as our result
			{
				Rect r(x,y,width,height);
				//remove dulpicates
				bool add=true;
				for(list<Rect>::const_iterator citer=final_detection.cbegin();citer!=final_detection.cend();++citer)
				{
					if((r&(*citer))==r)//replicate
					{
						add=false;
						break;
					}
				}
				if(add)
					final_detection.push_back(r);
			}
		}

	}else//prediction
	{
		//average of nearest detection result with prediction
		vector<Rect>::const_iterator citer_f;
		int center_predict[]={prediction.x+prediction.width/2,prediction.y+prediction.height/2};//center of prediction
		for(list<Point>::const_iterator citer_p=maxima.cbegin(); citer_p!=maxima.cend(); ++citer_p)
		{
			int min_dist=INT_MAX;
			int width=-1,height=-1;//derived width and height
			int x,y;//final detection position
			for(citer_f=found_filtered.cbegin();citer_f!=found_filtered.cend();++citer_f)
			{
				int center_x=citer_f->x+citer_f->width/2;
				int center_y=citer_f->y+citer_f->height/2;
				int dist=abs(center_x-citer_p->x)+abs(center_y-citer_p->y);
				if(dist<min_dist&&citer_f->contains(*citer_p))//must contain the maxima
				{
					min_dist = dist;
					width = citer_f->width;
					height = citer_f->height;
					x = citer_f->x;
					y = citer_f->y;
				}
			}

			if(width!=-1)//have nearest detection, use the detection result
			{
				//compare with prediction. if prediction is more accurate, use prediction (the longer we successfuuly predict. more confidence we place on it)
				int dist_to_predict = abs(center_predict[0]-citer_p->x)+abs(center_predict[1]-citer_p->y);
				if(dist_to_predict<min_dist)//prediction is nearer to maxima
				{//use the predicted result
					width=prediction.width;
					height=prediction.height;
					x = prediction.x;
					y = prediction.y;
				}
			}
			else//use the prediction
			{
				width=prediction.width;
				height=prediction.height;
				x = prediction.x;//directly use prediction//local maxima is biased to noise
				y = prediction.y;
			}
			Rect r(x,y,width,height);
			//remove dulpicates
			bool add=true;
			for(list<Rect>::const_iterator citer=final_detection.cbegin();citer!=final_detection.cend();++citer)
			{
				if((r&(*citer))==r)//replicate
				{
					add=false;
						break;
				}
			}
			if(add)
				final_detection.push_back(r);
		}
	}
}