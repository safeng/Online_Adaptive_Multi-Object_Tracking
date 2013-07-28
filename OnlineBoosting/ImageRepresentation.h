#pragma once
#include <memory>
#include <cmath>
#include <cstdint>
#include "Regions.h"
#include "OS_specific.h"

class ImageRepresentation  
{
public:

	ImageRepresentation(unsigned char* image, Size imagSize);
	ImageRepresentation(unsigned char* image, Size imagSize, Rect imageROI);
	void defaultInit(unsigned char* image, Size imageSize);
	virtual ~ImageRepresentation();

	int getSum(Rect imageROI);
	float getMean(Rect imagROI);
	unsigned int getValue(Point2D position);
	Size getImageSize(void){return m_imageSize;};
	Rect getImageROI(void){return m_ROI;};
	void setNewImage(unsigned char* image);
	void setNewROI(Rect ROI);
	void setNewImageSize( Rect ROI );
	void setNewImageAndROI(unsigned char* image, Rect ROI);
	float getVariance(Rect imageROI);
	long getSqSum(Rect imageROI);
	bool getUseVariance(){return m_useVariance;};
	void setUseVariance(bool useVariance){ this->m_useVariance = useVariance; };
	cv::Mat frame; //RGB image
	int frameNO;
private:

	bool m_useVariance;
	void createIntegralsOfROI(unsigned char* image); 

	Size m_imageSize;
	__uint32* intImage;//integral image 
	__uint64* intSqImage;//integral sq image
	Rect m_ROI;
	Point2D m_offset;
};
