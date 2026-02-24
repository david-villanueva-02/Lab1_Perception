#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/objdetect/aruco_detector.hpp>
#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include <string>
#include <map>

// Dictionary that maps the string input to the actual dictionary
static int dictFromString(const std::string& name){
    static const std::map<std::string,int> m = {
        {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
        {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
        {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
        {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},

        {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
        {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
        {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
        {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},

        {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
        {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
        {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
        {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},

        {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
        {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
        {"DICT_7X7_250", cv::aruco::DICT_7X7_250},
        {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},

        {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL}
    };

    auto it = m.find(name);
    if (it == m.end())
        throw std::runtime_error("Unknown dictionary: " + name);

    return it->second;
}

int main(int argc, char *argv[])
{
    if (argc < 8)
    {
        std::cerr << "Usage: ./calibrate <video_source_no> <dictionary_name> <detector_parameters> <board_rows> <board_cols> <marker_size> <separation>  <file_name>" << std::endl;
        return -1;
    }

    // Extract parameters
    std::string dictionaryName = argv[2];
    std::string detectorParamsFile = argv[3];
    int boardRows = std::atoi(argv[4]);
    int boardCols = std::atoi(argv[5]);
    float markerSize = std::stof(argv[6]);
    float separation = std::stof(argv[7]);
    std::string fileName = argv[8]; 

    // Process parameters.
    std::string dictionaryName = argv[2];
    static int dictionary_name = dictFromString(dictionaryName);

    // std::cout << std::atoi(argv[1]) << std::endl;
    cv::VideoCapture webCam(std::atoi(argv[1])); // VideoCapture object declaration. Usually 0 is the integrated, 2 is the first external USB one↪→

    if (webCam.isOpened() == false)
    { // Check if the VideoCapture object has been correctly associated to the webcam↪→
        std::cerr << "error: Webcam could not be connected." << std::endl;
        return -1;
    }

    cv::Mat imgOriginal;  // input image


    char charCheckForESCKey{0};
    char charCheckForCKey{0};
    int nframe = 0;



    // // Create board object and ArucoDetector
    // cv::aruco::GridBoard gridboard(cv::Size(boardRows, boardCols), markerSize, separation, dictionary_name);
    // cv::aruco::ArucoDetector detector(dictionary_name, detectorParamsFile);

    // // Collected frames for calibration
    // vector<vector<vector<Point2f>>> allMarkerCorners;
    // vector<vector<int>> allMarkerIds;
    // Size imageSize;

    // while(inputVideo.grab()) {
    //     Mat image, imageCopy;
    //     inputVideo.retrieve(image);

    //     vector<int> markerIds;
    //     vector<vector<Point2f>> markerCorners, rejectedMarkers;

    //     // Detect markers
    //     detector.detectMarkers(image, markerCorners, markerIds, rejectedMarkers);

    //     // Refind strategy to detect more markers
    //     if(refindStrategy) {
    //         detector.refineDetectedMarkers(image, gridboard, markerCorners, markerIds, rejectedMarkers);
    //     }
    //     if(key == 'c' && !markerIds.empty()) {
    //         cout << "Frame captured" << endl;
    //         allMarkerCorners.push_back(markerCorners);
    //         allMarkerIds.push_back(markerIds);
    //         imageSize = image.size();
    //     }
    // }

    return 0;
}
