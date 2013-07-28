#include "WeakClassifierRGBHist.h"

WeakClassifierRGBHist::WeakClassifierRGBHist(Size patchSize) : histPos(NULL), histNeg(NULL)
{
	generateRandomClassifier(patchSize);
	initPatchSize = patchSize;
	left_scale = ROI.left/float(initPatchSize.width);
	upper_scale = ROI.upper/float(initPatchSize.height);
}


WeakClassifierRGBHist::~WeakClassifierRGBHist(void)
{
	delete[] histPos;
	delete[] histNeg;
}

void WeakClassifierRGBHist::generateRandomClassifier(Size patchSize)
{
	Size minSize(3,3);
	int minArea = 9;
	bool valid = false;
	while(!valid)
	{
		//choses position and scale
		ROI.upper = rand()%(patchSize.height);
		ROI.left = rand()%(patchSize.width);

		ROI.width = (int) ((1-sqrt(1-(float)rand()/RAND_MAX))*patchSize.width);//1-sqrt(1-x)
		ROI.height = (int) ((1-sqrt(1-(float)rand()/RAND_MAX))*patchSize.height);

		if (ROI.upper + ROI.height >= patchSize.height ||
			ROI.left + ROI.width >= patchSize.width)
			continue;
		int area = ROI.height*ROI.width;
		if (area < minArea)
			continue;
		valid = true;
	}
	histNeg = new float[BIN_SIZE*3];
	histPos = new float[BIN_SIZE*3];
	std::fill(histPos,histPos+BIN_SIZE*3,0);
	std::fill(histNeg,histNeg+BIN_SIZE*3,0);
	first_pos = true;
	first_neg = true;
}

bool WeakClassifierRGBHist::evalFeatures(ImageRepresentation *image, Rect ROI, float *hist)
{
	if ( ROI.left<0 || ROI.upper <0 || ROI.width + ROI.left >= image->frame.cols || ROI.height + ROI.upper >= image->frame.rows)
	{
		return false;
	}
	
	float m_scale_width = ROI.width/float(initPatchSize.width);
	float m_scale_height = ROI.height/float(initPatchSize.height);
	float new_ROI_width = this->ROI.width*m_scale_width;
	float new_ROI_height = this->ROI.height*m_scale_height;
	float new_ROI_left = ROI.width*left_scale + ROI.left;
	float new_ROI_upper = ROI.height*upper_scale + ROI.upper;

	cv::Rect new_ROI(new_ROI_left,new_ROI_upper,new_ROI_width,new_ROI_height);
	if (new_ROI.area()<=9)
	{
		return false;
	}
	cv::Mat croppedImg = image->frame(new_ROI);
	//cv::resize(croppedImg,resizeImg,cv::Size(initPatchSize.width,initPatchSize.height));
	// extract features

	extractFeatures(croppedImg,hist);
	return true;
}

bool WeakClassifierRGBHist::update(ImageRepresentation* image, Rect ROI, int target)
{	
	float hist[BIN_SIZE*3];
	bool valid = evalFeatures(image,ROI,hist);
	if (!valid)
	{
		return true;
	}

	//incremental training
	if (target==1)
	{
		if (first_pos)
		{
			for (int i =0; i<BIN_SIZE*3; ++i)
			{
				histPos[i] = hist[i];
				m_pos[i].setValues(hist[i],1);
			}
			first_pos = false;
		}else
		{
			for (int i =0; i<BIN_SIZE*3; ++i)
			{
				m_pos[i].update(hist[i]);
				histPos[i] = m_pos[i].getMean();
			}
		}
		
	}else
	{
		if (first_neg)
		{
			for (int i =0; i<BIN_SIZE*3; ++i)
			{
				histNeg[i] = hist[i];
				m_neg[i].setValues(hist[i],1);
			}
			first_neg = false;
		}else
		{
			for (int i =0; i<BIN_SIZE*3; ++i)
			{
				m_neg[i].update(hist[i]);
				histNeg[i]=m_neg[i].getMean();
			}
		}
		
	}

	//evaluate after training
	return (predict(hist) != target);
}

void WeakClassifierRGBHist::extractFeatures(const cv::Mat &img, float *hist)
{
	
	std::fill(hist,hist+BIN_SIZE*3,0);
	uchar gap = 256 / BIN_SIZE;
	//assert(img.cols == initPatchSize.width && img.rows == initPatchSize.height);
	int numPix = 0;
	for (int i = 0; i<img.cols; ++i)
	{
		for (int j = 0; j < img.rows; ++j)
		{
			cv::Vec3b val = img.at<cv::Vec3b>(j,i);
			uchar b = val.val[0];
			uchar g = val.val[1];
			uchar r = val.val[2];
			int indexB = b / gap;
			int indexG = g / gap;
			int indexR = r / gap;
			hist[indexB] ++;
			hist[indexG + BIN_SIZE] ++;
			hist[indexR + BIN_SIZE * 2] ++;
			numPix++;
		}
	}

	for (int i = 0; i<BIN_SIZE*3; ++i)
	{
		hist[i]/= numPix;
	}
}

/************************************************************************/
/* Normalized correlation distance between two histograms               */
/************************************************************************/
float WeakClassifierRGBHist::normalizedCorrelation(const float *hist1, const float *hist2, unsigned int N)
{
	if (hist1 == NULL || hist2 == NULL)
	{
		return 0;
	}
	float avg = 1/float(N);
	float numerator = 0;
	float sig1 = 0;
	float sig2 = 0;
	for (int i = 0; i<N; ++i)
	{
		float diff1 = hist1[i]-avg;
		float diff2 = hist2[i]-avg;

		numerator += (diff1)*(diff2);
		sig1 += diff1*diff1;
		sig2 += diff2*diff2;
	}

	float distance = numerator / sqrt(sig1*sig2);
	return distance;
}

/************************************************************************/
/*Bhattacharyya_distance of two histograms                              */
/************************************************************************/
float WeakClassifierRGBHist::Bhattacharyya_distance(const float *hist1, const float *hist2, unsigned int N)
{
	assert(hist1!=NULL && hist2!=NULL);

	float sum = 0;
	float sum_1 = 0, sum_2 = 0;
	for (int i = 0; i<N; ++i)
	{
		assert(hist1[i]*hist2[i]>=0);
		sum += sqrt(hist1[i]*hist2[i]);
		sum_1 += hist1[i];
		sum_2 += hist2[i];
	}

	float coeff = 1/sqrt(sum_1*sum_2);
	float distance = sqrt(1-coeff*sum);
	return distance;
}

int WeakClassifierRGBHist::predict(const float *hist)
{
	float dist2Pos = Bhattacharyya_distance(histPos, hist, BIN_SIZE*3);
	float dist2Neg = Bhattacharyya_distance(histNeg, hist ,BIN_SIZE*3);
	if (dist2Pos>=dist2Neg)
	{
		return -1;
	}else
	{
		return 1;
	}
}

int WeakClassifierRGBHist::eval(ImageRepresentation* image, Rect ROI)
{
	float hist[BIN_SIZE*3];
	bool valid = evalFeatures(image,ROI,hist);
	if (!valid)
	{
		return 0;
	}

	return predict(hist);
}

float WeakClassifierRGBHist::getValue(ImageRepresentation* image, Rect ROI)
{
	assert(false);
	return 0;
}

void WeakClassifierRGBHist::print()
{
	for (int i = 0; i<BIN_SIZE*3; ++i)
	{
		std::cout<<"HistPos["<<i<<"]="<<histPos[i]<<"  "<<"HistNeg["<<i<<"]"<<histNeg[i]<<std::endl;
	}
	std::cout<<std::endl;
}