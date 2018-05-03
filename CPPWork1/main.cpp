

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <utility>
#include <iostream>
//#include <stdio.h>

using namespace cv;
using namespace std;

RNG rng(12345);
Scalar MY_RED (0, 0, 255);
Scalar MY_BLUE (255, 0, 0);
Scalar MY_GREEN (0, 255, 0);
Scalar MY_PURPLE (255, 0, 255);
Scalar GUIDE_DOT(255,255,0);
Point TEST_POINT(120,120);

const int RES_X = 640, RES_Y = 480;
const int MIN_HUE = 68, MAX_HUE = 180;
const int MIN_SAT = 140, MAX_SAT = 255;
const int MIN_VAL = 0, MAX_VAL = 255;

typedef std::vector<Point> contour_type;

const double
MIN_AREA = 0.001, MAX_AREA = 1000000,
MIN_WIDTH = 0, MAX_WIDTH = 100000, //rectangle width
MIN_HEIGHT = 50, MAX_HEIGHT = 100000, //rectangle height
MIN_RECT_RAT = 0.1, MAX_RECT_RAT = 10, //rect height / rect width
MIN_AREA_RAT = 0.85, MAX_AREA_RAT = 100,
BLUR = 25; //cvxhull area / contour area



inline int getHue ( Mat &img, int r, int c) {
    return img.at< Vec3b>(r, c)[0];
}

inline int getSat ( Mat &img, int r, int c) {
    return img.at< Vec3b>(r, c)[1];
}

inline int getVal ( Mat &img, int r, int c) {
    return img.at< Vec3b>(r, c)[2];
}

bool is_valid (contour_type &contour) {
    bool valid = true; //start out assuming its valid, disprove this later
    
    //find bounding rect & convex hull
     Rect rect =  boundingRect(contour);
    contour_type hull;
    cv::convexHull(contour, hull);
    
    double totalArea = (RES_X * RES_Y);
    
    //calculate relevant ratios & values
    double area =  contourArea(contour) / totalArea;
    //double perim =  arcLength(hull, true);
    
    double convex_area =  contourArea(hull) / totalArea;
    
    double width = rect.width, height = rect.height;
    
    double area_rat = area / convex_area;
    double rect_rat = height / width;

    
    //check ratios & values for validity
    if (area < MIN_AREA || area > MAX_AREA) valid = false;
    if (area_rat < MIN_AREA_RAT || area_rat > MAX_AREA_RAT) valid = false;
    if (rect_rat < MIN_RECT_RAT || rect_rat > MAX_RECT_RAT) valid = false;
    if (width < MIN_WIDTH || width > MAX_WIDTH) valid = false;
    if (height < MIN_HEIGHT || height > MAX_HEIGHT) valid = false;
    
    if (valid) {
        cout << "HEIGHT: ";
        cout <<height <<endl;
    }
    
    //valid = true; //for the sake of testing assume all contours are valid.
    return valid;
}

void calculate(const Mat &bgr, Mat &processedImage){
    //blur the image
    blur(bgr, processedImage, Size(BLUR, BLUR));
    Mat hsvMat;
    //convert to hsv
    cvtColor(processedImage, hsvMat, COLOR_BGR2HSV);
    
    //store HSV values at a given test point to send back
    int hue = getHue(hsvMat, TEST_POINT.x, TEST_POINT.y);
    int sat = getSat(hsvMat, TEST_POINT.x, TEST_POINT.y);
    int val = getVal(hsvMat, TEST_POINT.x, TEST_POINT.y);
    
    //threshold on green (light ring color)
    Mat greenThreshed;
    inRange(hsvMat,
                Scalar(MIN_HUE, MIN_SAT, MIN_VAL),
                Scalar(MAX_HUE, MAX_SAT, MAX_VAL),
                greenThreshed);
    
    processedImage = greenThreshed.clone();
    threshold (processedImage, processedImage, 0, 255, THRESH_BINARY);
    cvtColor(processedImage, processedImage, CV_GRAY2BGR);
    
    imshow("Processed", processedImage);

    //processedImage = bgr.clone();
    
    //drawPoint (processedImage, TEST_POINT, GUIDE_DOT);
    
    //contour detection
    vector<contour_type> contours;
    vector<Vec4i> hierarchy; //throwaway, needed for function
    try {
        findContours (greenThreshed, contours, hierarchy,
                          RETR_TREE, CHAIN_APPROX_SIMPLE);
    }
    catch (...) { //TODO: change this to the error that occurs when there are no contours
        cout << "No contours";
    }
    
    
    /*if (contours.size() < 1) { //definitely did not find
     cout << "No contours";
     }*/
    
    //store the convex hulls of any valid contours
    vector<contour_type> valid_contour_hulls;
    for (int i = 0; i < (int)contours.size(); i++) {
        contour_type contour = contours[i];
        if (is_valid (contour)) {
            contour_type hull;
            cv::convexHull(contour, hull);
            valid_contour_hulls.push_back(hull);
        }
    }
    
    int numContours = valid_contour_hulls.size();
    printf ("Num contours: %d\n", numContours);
    
    if (numContours < 1) { //definitely did not find
        //printf("DID NOT FIND CONTOUR");
    }
    
    //find the largest contour in the image
    contour_type largest;
    double largestArea = 0;
    for (int i = 0; i < numContours; i++){
        double curArea = contourArea(valid_contour_hulls[i], true);
        if (curArea > largestArea){
            largestArea = curArea;
            largest = valid_contour_hulls[i];
        }
    }
    
    //get the points of corners
    vector<Point> all_points;
    all_points.insert (all_points.end(), largest.begin(), largest.end());
    
    //find which corner is which
    Point ul (1000, 1000), ur (0, 1000), ll (1000, 0), lr (0, 0);
    for (int i = 0; i < (int)all_points.size(); i++) {
        int sum = all_points[i].x + all_points[i].y;
        int dif = all_points[i].x - all_points[i].y;
        
        if (sum < ul.x + ul.y) {
            ul = all_points[i];
        }
        
        if (sum > lr.x + lr.y) {
            lr = all_points[i];
        }
        
        if (dif < ll.x - ll.y) {
            ll = all_points[i];
        }
        
        if (dif > ur.x - ur.y) {
            ur = all_points[i];
        }
    }
    
    //find the center of mass of the largest contour
    Moments centerMass = moments(largest, true);
    double centerX = (centerMass.m10) / (centerMass.m00);
    double centerY = (centerMass.m01) / (centerMass.m00);
    Point center (centerX, centerY);
    
    vector<contour_type> largestArr;
    largestArr.push_back(largest);
    //drawContours(processedImage, largestArr , 0, MY_GREEN, 2);
    
    double top_width = ur.x - ul.x;
    double bottom_width = lr.x - ll.x;
    double left_height = ll.y - ul.y;
    double right_height = lr.y - ur.y;
    
    double angle = (centerX * top_width);
        if (numContours >= 1) {
            cout << "CENTER X: ";
            cout << centerX << endl;
            cout << "CENTER Y: ";
            cout <<centerY <<endl;
    }
    
    //create the results package
    
    /* copyPointData (ul, res.ul);
     copyPointData (ur, res.ur);
     copyPointData (ll, res.ll);
     copyPointData (lr, res.lr);
     copyPointData (center, res.midPoint);
     
     res.upperWidth = top_width;
     res.lowerWidth = bottom_width;
     res.leftHeight = left_height;
     res.rightHeight = right_height;
     
     res.sampleHue = hue;
     res.sampleSat = sat;
     res.sampleVal = val;
     
     drawOnImage (processedImage, res);*/
    //return res;
}

int process(VideoCapture& capture) {
    int n = 0;
    char filename[200];
    string window_name = "video | q or esc to quit";
    cout << "press space to save a picture. q or esc to quit" << endl;
    namedWindow(window_name, WINDOW_KEEPRATIO); //resizable window;
    Mat frame;
        
    for (;;) {
        capture >> frame;
        if (frame.empty())
            break;
        
        imshow(window_name, frame);
        char key = (char)waitKey(30);
        calculate(frame, frame);
        switch (key) {
            case 'q':
            case 'Q':
            case 27: //escape key
                return 0;
            case ' ': //Save an image
                sprintf(filename,"filename%.3d.jpg",n++);
                imwrite(filename,frame);
                cout << "Saved " << filename << endl;
                break;
            default:
                break;
        }
    }
    return 0;
}




int main(int ac, char** av) {
    string arg = av[1];
    VideoCapture capture(arg); //try to open string, this will attempt to open it as a video file or image sequence
    if (!capture.isOpened()) //if this fails, try to open as a video camera, through the use of an integer param
        capture.open(atoi(arg.c_str()));
    if (!capture.isOpened()) {
        cerr << "Failed to open the video device, video file or image sequence!\n" << endl;
        return 1;
    }
    return process(capture);
}














