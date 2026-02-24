#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <opencv2/objdetect/aruco_detector.hpp>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <string>
#include <map>

#include <sstream>

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

// Generate a single marker
cv::Mat generateMarker(int markerId, int markerSize, cv::aruco::Dictionary dictionary){
    // Generate the marker
    cv::Mat markerImage;
    cv::aruco::generateImageMarker(dictionary, markerId, markerSize, markerImage, 1);
    return markerImage;
} 

int main(int argc, char *argv[])
{
    if (argc < 7)
    {
        std::cout << "Usage: ./generate_board <number_of_rows> <number_of_cols> <dictionary> <marker_size> <separation> <file_name> " << std::endl;
        return -1;
    }

    // Extract parameters
    int numberOfRows = std::atoi(argv[1]);
    int numberOfCols = std::atoi(argv[2]);
    int dictionary_name = dictFromString(argv[3]);
    int markerSize = std::atoi(argv[4]);
    int separation = std::atoi(argv[5]);
    std::string fileName = argv[6];

    // Extract the dictionary
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(dictionary_name);

    // Generate the background
    const int rows = (numberOfRows*markerSize + separation*(numberOfRows+1));
    const int cols = (numberOfCols*markerSize + separation*(numberOfCols+1));
    cv::Mat baseImage = cv::Mat::ones(rows, cols, CV_8UC1)*255;

    // Generate the markers and add them to the background
    for (int i = 0; i < numberOfRows; i++){
        for(int j = 0; j < numberOfCols; j++){
            // Generate the marker
            cv::Mat ArUco_i = generateMarker(i*numberOfCols+j, markerSize, dictionary);

            // Add the marker to the background
            ArUco_i.copyTo(baseImage(cv::Rect(j*markerSize + (j+1)*separation, i*markerSize + (i+1)*separation, markerSize, markerSize)));  
        }   
    }

    std::stringstream ss;
    ss << "/home/david/Documents/IFRoS/Perception/labs/lab1/images/" << fileName;
    cv::imwrite(ss.str(), baseImage);

    return 0;
}