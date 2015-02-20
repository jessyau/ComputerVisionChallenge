/* In order to make SURF/SIFT work, you will need to install opencv_contrib which is only compatible with
 * OpenCV 3.0.0 and up. opencv_contrib includes the library xfeatures2d used in this code. 
 * Find it at: https://github.com/Itseez/opencv_contrib
 */

#include <iostream>
#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;


bool selectObject = false;
Rect selection;
Point origin;
int trackObject = 0;
int selected = 0;
int k = 0;
Mat image, imageClone, imageROI;

const int LOOP_NUM = 10;
const int GOOD_PTS_MAX = 40;
const float GOOD_PORTION = 0.15f;

struct SURFDetector
{
    Ptr<Feature2D> surf;
    SURFDetector(double hessian = 800.0)
    {
        surf = SURF::create(hessian);
    }
    template<class T>
    void operator()(const T& in, const T& mask, std::vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
    {
        surf->detectAndCompute(in, mask, pts, descriptors, useProvided);
    }
};

template<class KPMatcher>
struct SURFMatcher
{
    KPMatcher matcher;
    template<class T>
    void match(const T& in1, const T& in2, std::vector<cv::DMatch>& matches)
    {
        matcher.match(in1, in2, matches);
    }
};

static Mat drawGoodMatches(
                           const Mat& img1,
                           const Mat& img2,
                           const std::vector<KeyPoint>& keypoints1,
                           const std::vector<KeyPoint>& keypoints2,
                           std::vector<DMatch>& matches,
                           std::vector<Point2f>& scene_corners_
                           )
{
    //-- Sort matches and preserve top 10% matches
    std::sort(matches.begin(), matches.end());
    std::vector< DMatch > good_matches;
    double minDist = matches.front().distance;
    double maxDist = matches.back().distance;
    
    const int ptsPairs = std::min(GOOD_PTS_MAX, (int)(matches.size() * GOOD_PORTION));
    for( int i = 0; i < ptsPairs; i++ )
    {
        good_matches.push_back( matches[i] );
    }
    std::cout << "\nMax distance: " << maxDist << std::endl;
    std::cout << "Min distance: " << minDist << std::endl;
    
    std::cout << "Calculating homography using " << ptsPairs << " point pairs." << std::endl;
    
    // drawing the results
    Mat img_matches;
    
    drawMatches( img1, keypoints1, img2, keypoints2,
                good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS  );
    
    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
    }
    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point(0,0);
    obj_corners[1] = Point( img1.cols, 0 );
    obj_corners[2] = Point( img1.cols, img1.rows );
    obj_corners[3] = Point( 0, img1.rows );
    std::vector<Point2f> scene_corners(4);
    
    Mat H = findHomography( obj, scene, RANSAC );
    perspectiveTransform( obj_corners, scene_corners, H);
    
    scene_corners_ = scene_corners;
    
    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_matches,
         scene_corners[0] + Point2f( (float)img1.cols, 0), scene_corners[1] + Point2f( (float)img1.cols, 0),
         Scalar( 0, 255, 0), 2, LINE_AA );
    line( img_matches,
         scene_corners[1] + Point2f( (float)img1.cols, 0), scene_corners[2] + Point2f( (float)img1.cols, 0),
         Scalar( 0, 255, 0), 2, LINE_AA );
    line( img_matches,
         scene_corners[2] + Point2f( (float)img1.cols, 0), scene_corners[3] + Point2f( (float)img1.cols, 0),
         Scalar( 0, 255, 0), 2, LINE_AA );
    line( img_matches,
         scene_corners[3] + Point2f( (float)img1.cols, 0), scene_corners[0] + Point2f( (float)img1.cols, 0),
         Scalar( 0, 255, 0), 2, LINE_AA );
    return img_matches;
}

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
        cap >> image;   // Stores camera frame into Mat image
        if( image.empty() ) break;

        
        if( selectObject && selection.width > 0 && selection.height > 0 )
        {

            imageClone = image.clone();
            rectangle(image, selection, Scalar(255, 0, 0), 2, 8, 0); // Draws rectangle around ROI
            
            printf("%d %d %d %d\n", selection.x, selection.y, selection.width, selection.height);
        }
        
        if( !imageROI.empty())
        {
            // declare input/output
            std::vector<KeyPoint> keypoints1, keypoints2;
            std::vector<DMatch> matches;
            
            UMat _descriptors1, _descriptors2;
            Mat descriptors1 = _descriptors1.getMat(ACCESS_RW),
            descriptors2 = _descriptors2.getMat(ACCESS_RW);
            
            // instantiate detectors/matchers
            SURFDetector surf;
            
            SURFMatcher<BFMatcher> matcher;
            
            
            for (int i = 0; i <= LOOP_NUM; i++)
            {
                surf(image, Mat(), keypoints1, descriptors1);
                surf(imageROI, Mat(), keypoints2, descriptors2);
                matcher.match(descriptors1, descriptors2, matches);
            }

            std::cout << "FOUND " << keypoints1.size() << " keypoints on first image" << std::endl;
            std::cout << "FOUND " << keypoints2.size() << " keypoints on second image" << std::endl;
            
            
            std::vector<Point2f> corner;
            Mat img_matches = drawGoodMatches(image, imageROI, keypoints1, keypoints2, matches, corner);
            
            //-- Show detected matches
            
            imshow("Camera Feed", img_matches);
        } else imshow( "Camera Feed", image );
        
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