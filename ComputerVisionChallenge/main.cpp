#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;


bool selectObject = false;
Rect selection;
Point origin;
int trackObject = 0;
int selected = 0;
int k = 0;
Mat image, imageClone, imageROI;

// This function gets called whenever there is a mouse event -- any real execution is limited by one object selection event unless specified by user to redo selection by pressing the space bar
static void onMouse( int event, int x, int y, int, void* )
{
    if( trackObject == 0 )
    {
        // Calculates ROI measurements and filters selection area
        if( selectObject )
        {
            selection.x = MIN(x, origin.x);
            selection.y = MIN(y, origin.y);
            selection.width = std::abs(x - origin.x);
            selection.height = std::abs(y - origin.y);
            
            selection &= Rect(0, 0, image.cols, image.rows);
        }

        switch( event )
        {
            // Stores points of mouseclick (into a point) and selection points (into a rect)
            case EVENT_LBUTTONDOWN:
                origin = Point(x,y);
                selection = Rect(x,y,0,0);
                selectObject = true;
                break;
            case EVENT_LBUTTONUP:
                selectObject = false;
                
                if( selection.width > 0 && selection.height > 0 )
                {
                    trackObject = -1;
                    imageROI = imageClone(selection);    // "Snapshots" ROI & displays snapshot
                    imshow("ROI", imageROI);
                }
                break;
        }
    }
}


int main (void)
{
    
    // Starts webcam and services
    VideoCapture cap(0);
    Mat frame;
    namedWindow( "Camera Feed", 0 );
    setMouseCallback( "Camera Feed", onMouse, 0 );
    
    // Infinite looooooop to loop through camera frames
    for(;;)
    {
        cap >> image;
        if( image.empty() ) break;

        
        if( selectObject && selection.width > 0 && selection.height > 0 )
        {

            imageClone = image.clone();
            rectangle(image, selection, Scalar(255, 0, 0), 2, 8, 0); // Draws rectangle around ROI
            
            printf("%d %d %d %d\n", selection.x, selection.y, selection.width, selection.height);
        }

        imshow( "Camera Feed", image );
        
        k = waitKey(10);
        
        if( k == 27 )   // Exits when ESC is pressed
            break;
        else if( k == 32)   // Redo ROI selection when Space is pressed
        {
            destroyWindow("ROI");
            trackObject = 0;
        }
        
    }  
    
    return 0;    
}