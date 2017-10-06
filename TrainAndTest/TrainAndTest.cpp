// TrainAndTest.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<sstream>

// global variables ///////////////////////////////////////////////////////////////////////////////
const int MIN_CONTOUR_AREA = 100;

const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

///////////////////////////////////////////////////////////////////////////////////////////////////
class ContourWithData {
public:
    // member variables ///////////////////////////////////////////////////////////////////////////
    std::vector<cv::Point> ptContour;           // contour
    cv::Rect boundingRect;                      // bounding rect for contour
    float fltArea;                              // area of contour

                                                ///////////////////////////////////////////////////////////////////////////////////////////////
    bool checkIfContourIsValid() {                              // obviously in a production grade program
        if (fltArea < MIN_CONTOUR_AREA) return false;           // we would have a much more robust function for 
        return true;                                            // identifying if a contour is valid !!
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {      // this function allows us to sort
        return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);                                                   // the contours from left to right
    }

};

///////////////////////////////////////////////////////////////////////////////////////////////////
int main() {
    std::vector<ContourWithData> allContoursWithData;           // declare empty vectors,
    std::vector<ContourWithData> validContoursWithData;         // we will fill these shortly

                                                                // read in training classifications ///////////////////////////////////////////////////

    cv::Mat matClassificationInts;      // we will read the classification numbers into this variable as though it is a vector

    cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);        // open the classifications file

    if (fsClassifications.isOpened() == false) {                                                    // if the file was not opened successfully
        std::cout << "error, unable to open training classifications file, exiting program\n\n";    // show error message
        return(0);                                                                                  // and exit program
    }

    fsClassifications["classifications"] >> matClassificationInts;      // read classifications section into Mat classifications variable
    fsClassifications.release();                                        // close the classifications file

                                                                        // read in training images ////////////////////////////////////////////////////////////

    cv::Mat matTrainingImagesAsFlattenedFloats;         // we will read multiple images into this single image variable as though it is a vector

    cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);          // open the training images file

    if (fsTrainingImages.isOpened() == false) {                                                 // if the file was not opened successfully
        std::cout << "error, unable to open training images file, exiting program\n\n";         // show error message
        return(0);                                                                              // and exit program
    }

    fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;           // read images section into Mat training images variable
    fsTrainingImages.release();                                                 // close the traning images file

                                                                                // train //////////////////////////////////////////////////////////////////////////////

    cv::Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());            // instantiate the KNN object

                                                                                // finally we get to the call to train, note that both parameters have to be of type Mat (a single Mat)
                                                                                // even though in reality they are multiple images / numbers
    kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);

    // test ///////////////////////////////////////////////////////////////////////////////

    cv::Mat matTestingNumbers = cv::imread("test1.png");            // read in the test numbers image

    if (matTestingNumbers.empty()) {                                // if unable to open image
        std::cout << "error: image not read from file\n\n";         // show error message on command line
        return(0);                                                  // and exit program
    }

    cv::Mat matGrayscale;           //
    cv::Mat matBlurred;             // declare more image variables
    cv::Mat matThresh;              //
    cv::Mat matThreshCopy;          //

    cv::cvtColor(matTestingNumbers, matGrayscale, CV_BGR2GRAY);         // convert to grayscale

                                                                        // blur
    cv::GaussianBlur(matGrayscale,              // input image
        matBlurred,                // output image
        cv::Size(5, 5),            // smoothing window width and height in pixels
        0);                        // sigma value, determines how much the image will be blurred, zero makes function choose the sigma value

                                   // filter image from grayscale to black and white
    cv::adaptiveThreshold(matBlurred,                           // input image
        matThresh,                            // output image
        255,                                  // make pixels that pass the threshold full white
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,       // use gaussian rather than mean, seems to give better results
        cv::THRESH_BINARY_INV,                // invert so foreground will be white, background will be black
        11,                                   // size of a pixel neighborhood used to calculate threshold value
        2);                                   // constant subtracted from the mean or weighted mean

    matThreshCopy = matThresh.clone();              // make a copy of the thresh image, this in necessary b/c findContours modifies the image

    std::vector<std::vector<cv::Point> > ptContours;        // declare a vector for the contours
    std::vector<cv::Vec4i> v4iHierarchy;                    // declare a vector for the hierarchy (we won't use this in this program but this may be helpful for reference)

    cv::findContours(matThreshCopy,             // input image, make sure to use a copy since the function will modify this image in the course of finding contours
        ptContours,                             // output contours
        v4iHierarchy,                           // output hierarchy
        cv::RETR_EXTERNAL,                      // retrieve the outermost contours only
        cv::CHAIN_APPROX_SIMPLE);               // compress horizontal, vertical, and diagonal segments and leave only their end points

    for (int i = 0; i < ptContours.size(); i++) {               // for each contour
        ContourWithData contourWithData;                                                    // instantiate a contour with data object
        contourWithData.ptContour = ptContours[i];                                          // assign contour to contour with data
        contourWithData.boundingRect = cv::boundingRect(contourWithData.ptContour);         // get the bounding rect
        contourWithData.fltArea = cv::contourArea(contourWithData.ptContour);               // calculate the contour area
        allContoursWithData.push_back(contourWithData);                                     // add contour with data object to list of all contours with data
    }

    for (int i = 0; i < allContoursWithData.size(); i++) {                      // for all contours
        if (allContoursWithData[i].checkIfContourIsValid()) {                   // check if valid
            validContoursWithData.push_back(allContoursWithData[i]);            // if so, append to valid contour list
        }
    }
    // sort contours from left to right
    std::sort(validContoursWithData.begin(), validContoursWithData.end(), ContourWithData::sortByBoundingRectXPosition);

    std::string strFinalString;         // declare final string, this will have the final number sequence by the end of the program

    for (int i = 0; i < validContoursWithData.size(); i++) {            // for each contour

                                                                        // draw a green rect around the current char
        cv::rectangle(matTestingNumbers,                            // draw rectangle on original image
            validContoursWithData[i].boundingRect,        // rect to draw
            cv::Scalar(0, 255, 0),                        // green
            2);                                           // thickness

        cv::Mat matROI = matThresh(validContoursWithData[i].boundingRect);          // get ROI image of bounding rect

        cv::Mat matROIResized;
        cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     // resize image, this will be more consistent for recognition and storage

        cv::Mat matROIFloat;
        matROIResized.convertTo(matROIFloat, CV_32FC1);             // convert Mat to float, necessary for call to find_nearest

        cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);

        cv::Mat matCurrentChar(0, 0, CV_32F);

        kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);     // finally we can call find_nearest !!!

        float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

        strFinalString = strFinalString + char(int(fltCurrentChar));        // append current char to full string
    }

    std::cout << "\n\n" << "numbers read = " << strFinalString << "\n\n";       // show the full string

    cv::imshow("matTestingNumbers", matTestingNumbers);     // show input image with green boxes drawn around found digits

    cv::waitKey(0);                                         // wait for user key press

    return(0);
}


