/*In this header file, we define all our parameters for tracking*/
#pragma once
#include "stdfunc.h"
/************************Parameters for the video********************************/
#define NUMFRAMES 795// The total number of frames in our video
#define NUMFLFEATURES 400// The number of initial features
#define SHRINKFEATURE 25// The frame to shrink the featurelist

#define NON_KEYFRAME_GAP 3 //The frame gap between the consecutive non-key frames (should not be very large otherwise we cannot detect many objects)
#define MAX_OBJ_TACKSET_SIZE 500 // Define the max size of object tracks
#define MAX_OBJSUBTRACKSET_SIZE 100 //The capacity of each detection tracks
#define NUMTRACKS 5500// Number of initial feature tracks
#define NUMFEATURES 500// Inital number of features per track
#define MO_GROUP 2 //We consider & compare robust motion features when tracks' length are above this threshold
#define MIN_CONSISTENT_LEN MO_GROUP+2 //Min. length of a consistent feature track
#define NUMINITFEATURES 200//Assume a constant
#define WRITETEST 0//switch for whether we should write intermediate results
#define MIN_DIST 3

extern int image_width, image_height;//Global variable