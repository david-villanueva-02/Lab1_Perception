#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/objdetect/aruco_detector.hpp>
#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include <string>
#include <map>

#include <filesystem>

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
        std::cerr << "Usage: ./calibrate <video_source_no> <dictionary_name> <detector_parameters> <board_rows> <board_cols> <marker_size> <separation>  <file_name> <take_images_bool (optional)>" << std::endl;
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
    static int dictionary_name = dictFromString(dictionaryName);

    // std::cout << std::atoi(argv[1]) << std::endl;
    cv::VideoCapture webCam(std::atoi(argv[1])); // VideoCapture object declaration. Usually 0 is the integrated, 2 is the first external USB one↪→

    if (webCam.isOpened() == false)
    { // Check if the VideoCapture object has been correctly associated to the webcam↪→
        std::cerr << "error: Webcam could not be connected." << std::endl;
        return -1;
    }

    cv::Mat imgOriginal;  // input image
    std::string image_set_path;

    char charCheckForESCKey{0};
    char charCheckForCKey{0};
    int nframe = 0;

    // Loop to take images
    while (charCheckForESCKey != 27 && webCam.isOpened()){                                              
        bool frameSuccess = webCam.read(imgOriginal);

        if (!frameSuccess || imgOriginal.empty()){ 
            std::cerr << "error: Frame could not be read." << std::endl;
            break;
        }
        charCheckForCKey = cv::waitKey(1);

        // Leave the loop and go to calibration
        if(charCheckForCKey == 's' || std::atoi(argv[9]) == 0){
            // Leaves the while loop to begin calibration
            break;
        }
        
        // Show ArUcos in image
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;

        // Create the parametes for the detector and get the dictionary to use.
        cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
        cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(dictionary_name);
        cv::aruco::ArucoDetector detector(dictionary, detectorParams);

        // Detect the markers on the imageand draw the markers detected
        detector.detectMarkers(imgOriginal, markerCorners, markerIds, rejectedCandidates);
        cv::Mat outputImage = imgOriginal.clone();
        cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);

        cv::imshow("imgOriginal", outputImage);

        // Capture image when the c key is pressed and save it
        if (charCheckForCKey == 'c' && std::atoi(argv[9]) == 1 && !markerIds.empty()){
            
            printf("Capturing image %d\n", nframe);
            std::string fileName = "/home/david/Documents/IFRoS/Perception/labs/lab1/images/image_" + std::to_string(nframe) + ".jpg";
            cv::imwrite(fileName, imgOriginal);
            nframe++; // Counting the number of frames captured
        }

        // Gets the key pressed
        charCheckForESCKey = cv::waitKey(1); 
    }

    

    return 0;
}
