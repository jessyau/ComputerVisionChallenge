#include <iostream>
#include <stdio.h>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


using namespace cv;
using namespace std;

bool selectObject = false;
Rect selection;
Point origin;
int trackObject = 0;
int selected = 0;
int k = 0;
Mat image, image_gray, imageClone, imageROI;

namespace
{
    // windows and trackbars name
    const std::string windowName = "Hough Circle Detection Demo";
    const std::string cannyThresholdTrackbarName = "Canny threshold";
    const std::string accumulatorThresholdTrackbarName = "Accumulator Threshold";
    const std::string usage = "Usage : tutorial_HoughCircle_Demo <path_to_input_image>\n";

    // initial and max values of the parameters of interests.
    const int cannyThresholdInitialValue = 200;
    const int accumulatorThresholdInitialValue = 50;
    const int maxAccumulatorThreshold = 200;
    const int maxCannyThreshold = 255;

    void HoughDetection(const Mat& src_gray, const Mat& src_display, int cannyThreshold, int accumulatorThreshold)
    {
        // will hold the results of the detection
        std::vector<Vec3f> circles;
        // runs the actual detection
        HoughCircles( src_gray, circles, HOUGH_GRADIENT, 1, src_gray.rows/8, cannyThreshold, accumulatorThreshold, 0, 0 );

        // clone the colour, input image for displaying purposes
        Mat display = src_display.clone();
        for( size_t i = 0; i < circles.size(); i++ )
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            // circle center
            circle( display, center, 3, Scalar(0,255,0), -1, 8, 0 );
            // circle outline
            circle( display, center, radius, Scalar(0,0,255), 3, 8, 0 );
        }

        // shows the results
        imshow( windowName, display);
    }
}

int main (void)
{
    
    // Starts webcam and services
    VideoCapture cap(0);
    Mat frame;
    //namedWindow( "Camera Feed", 0 );
    
    //declare and initialize both parameters that are subjects to change
    int cannyThreshold = cannyThresholdInitialValue;
    int accumulatorThreshold = accumulatorThresholdInitialValue;

    // create the main window, and attach the trackbars
    namedWindow( windowName, WINDOW_AUTOSIZE );
    createTrackbar(cannyThresholdTrackbarName, windowName, &cannyThreshold,maxCannyThreshold);
    createTrackbar(accumulatorThresholdTrackbarName, windowName, &accumulatorThreshold, maxAccumulatorThreshold);

    // Infinite looooooop to loop through camera frames
    for(;;)
    {
        cap >> image;   // Stores camera frame into Mat image
        if( image.empty() ) break;

        //Hough Transform code
        // Convert it to gray
        cvtColor( image, image_gray, COLOR_BGR2GRAY );

        // Reduce the noise so we avoid false circle detection
        GaussianBlur( image_gray, image_gray, Size(9, 9), 2, 2 );

        // those paramaters cannot be =0
        // so we must check here
        cannyThreshold = std::max(cannyThreshold, 1);
        accumulatorThreshold = std::max(accumulatorThreshold, 1);

        //runs the detection, and update the display
        HoughDetection(image_gray, image, cannyThreshold, accumulatorThreshold);
        
        k = waitKey(10);
        
        if( k == 27 )   // Exits when ESC is pressed
            break;
    }  
    
    return 0;    
}