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

//const int LOOP_NUM = 10;
const int GOOD_PTS_MAX = 30;
const float GOOD_PORTION = 0.1f;


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
    std::cout << "Good Matches Size: " << good_matches.size() << std::endl;
    
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
    
    if (!H.empty())
    {
        perspectiveTransform( obj_corners, scene_corners, H);
        
        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        line( img_matches,
             scene_corners[0] + Point2f(img1.cols, 0), scene_corners[1] + Point2f(img1.cols, 0),
             Scalar( 0, 255, 0), 2, LINE_AA );
        line( img_matches,
             scene_corners[1] + Point2f(img1.cols, 0), scene_corners[2] + Point2f(img1.cols, 0),
             Scalar( 0, 255, 0), 2, LINE_AA );
        line( img_matches,
             scene_corners[2] + Point2f(img1.cols, 0), scene_corners[3] + Point2f(img1.cols, 0),
             Scalar( 0, 255, 0), 2, LINE_AA );
        line( img_matches,
             scene_corners[3] + Point2f(img1.cols, 0), scene_corners[0] + Point2f(img1.cols, 0),
             Scalar( 0, 255, 0), 2, LINE_AA );
    } else std::cout << "findHomography failed " << minDist << std::endl;
    
    return img_matches;
}



int main (void)
{
    
    // Starts webcam and services
    VideoCapture cap(0);
    Mat frame;
    namedWindow( "Camera Feed", WINDOW_AUTOSIZE );
//    setMouseCallback( "Camera Feed", onMouse, 0 );
    
    // declare input/output
    std::vector<KeyPoint> keypoints1, keypoints2;
    
    cv::Ptr<Feature2D> f2d = xfeatures2d::SURF::create();
    Mat img_matches;
    
    imageROI = imread( "/Users/Jessica/Documents/CompVi/CompVi/sample.jpeg", CV_LOAD_IMAGE_GRAYSCALE );
    resize(imageROI, imageROI, Size(imageROI.cols/2, imageROI.rows/2));
    
    if( !imageROI.data )
    {
        std::cout<< "Error reading object " << std::endl;
        return -1;
    }
    
    // Infinite looooooop to loop through camera frames
    for(;;)
    {
        cap >> frame;   // Stores camera frame into Mat image
        if( frame.empty() ) break;
        
        // Grayscales and resize camera frame
        cvtColor(frame, image, CV_RGB2GRAY);
        resize(image, image, Size(image.cols/2, image.rows/2));
        
        // Detect the keypoints:
        f2d->detect( image, keypoints1 );
        f2d->detect( imageROI, keypoints2 );
        
        // Calculate descriptors (feature vectors)
        Mat descriptors_1, descriptors_2;
        f2d->compute( image, keypoints1, descriptors_1 );
        f2d->compute( imageROI, keypoints2, descriptors_2 );
        
        // Matching descriptor vectors using BFMatcher :
        std::vector<DMatch> matches;
        FlannBasedMatcher matcher;
        matcher.match( descriptors_1, descriptors_2, matches );
            
        std::vector<Point2f> corner;
        if (matches.size() > 0 && keypoints1.size() > 0 && keypoints2.size() > 0)
        {
            std::cout << "Found Matches" << std::endl;
            img_matches = drawGoodMatches(image, imageROI, keypoints1, keypoints2, matches, corner);
        }
        
        //-- Show detected matches
        if ( !img_matches.empty() )
            imshow("Results", img_matches);
        
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