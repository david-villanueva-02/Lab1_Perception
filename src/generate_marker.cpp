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

int main(int argc, char *argv[]){

    // Static variables
    static int dictionary_name;

    // Check the number of arguments
    if (argc != 5)
    {   
        std::cout << "Usage is ./generate_marker <dictionary> <id> <size> <file_name>" << std::endl;
        return -1;
    }

    // Extract parameters
    std::string dictionaryName = argv[1];
    int markerId = std::stoi(argv[2]);
    int markerSize = std::stoi(argv[3]);
    std::string fileName = argv[4];
    
    // Process parameters
    dictionary_name = dictFromString(dictionaryName);

    // Generate the marker
    std::cout << "Generating marker with the following parameters:" << std::endl;
    cv::Mat markerImage;
    cv::Mat markerImage2;

    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(dictionary_name);
    cv::aruco::generateImageMarker(dictionary, markerId, markerSize, markerImage, 1);

    // Add the border
    int left, right, top, bottom;
    top = (int) (0.05*markerSize); bottom = top;
    left = (int) (0.05*markerSize); right = left;

    cv::copyMakeBorder(markerImage, markerImage2, top, bottom, left, right, cv::BORDER_CONSTANT, 255);
    
    cv::imwrite("/home/david/Documents/IFRoS/Perception/labs/lab1/images/" + fileName, markerImage2);

    return 0;
}