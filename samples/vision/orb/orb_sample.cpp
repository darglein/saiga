#include "orb_sample.h"
#include <iostream>
#include <iomanip>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pangolin/pangolin.h>
#include <saiga/extra/opencv/opencv.h>

using namespace std;

void SequenceMode(string &imgPath, int nFeatures, float scaleFactor, int nLevels, int FASTThresholdInit,
                  int FASTThresholdMin, cv::Scalar color, int thickness, int radius, bool drawAngular,
                  Dataset dataset)
{
    cout << "\nStarting...\n";

    vector<string> vstrImageFilenamesLeft;
    vector<double> vTimestamps;
    vector<string> vstrImageFilenamesRight;

    if (dataset == tum)
    {
        string strFile = string(imgPath)+"/rgb.txt";
        LoadImagesTUM(strFile, vstrImageFilenamesLeft, vTimestamps);
    }

    else if (dataset == kitti)
    {
        LoadImagesKITTI(imgPath, vstrImageFilenamesLeft, vstrImageFilenamesRight, vTimestamps);
    }

    else if (dataset == euroc)
    {
        string pathLeft = imgPath, pathRight = imgPath, pathTimes = imgPath;
        pathLeft += "cam0/data/";
        pathRight += "cam1/data/";
        pathTimes += "MH03.txt";
        LoadImagesEUROC(pathLeft, pathRight, pathTimes,vstrImageFilenamesLeft, vstrImageFilenamesRight, vTimestamps);
    }
    else
    {
        cerr << "No valid dataset!";
        exit(EXIT_FAILURE);
    }

    int nImages = vstrImageFilenamesLeft.size();

    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    SaigaORB::ORBextractor myExtractor(nFeatures, scaleFactor, nLevels, FASTThresholdInit, FASTThresholdMin);

    cout << "\n-------------------------\n"
         << "Images in sequence: " << nImages << "\n";

    long myTotalDuration = 0;

    int softTh = 4;

    cv::Mat img;
    cv::Mat imgRight;

    pangolin::CreateWindowAndBind("Menu",210,600);

    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(210));

    pangolin::Var<bool> menuPause("menu. ~PAUSE~", true, true);
    pangolin::Var<int> menuSoftSSCThreshold("menu.Soft SSC Threshold", softTh, 0, 100);
    pangolin::Var<bool> menuDistrPerLvl("menu.Distribute Per Level", true, true);
    pangolin::Var<int> menuNFeatures("menu.Desired Features", nFeatures, 500, 2000);
    pangolin::Var<int> menuActualkpts("menu.Features Actual", false, 0);
    pangolin::Var<int> menuSetInitThreshold("menu.Init FAST Threshold", FASTThresholdInit, 5, 40);
    pangolin::Var<int> menuSetMinThreshold("menu.Min FAST Threshold", FASTThresholdMin, 1, 39);
    pangolin::Var<float> menuScaleFactor("menu.Scale Factor", scaleFactor, 1.001, 1.2);
    pangolin::Var<int> menuNLevels("menu.nLevels", nLevels, 2, 8);
    pangolin::Var<bool> menuSingleLvlOnly("menu.Dispay single level:", false, true);
    pangolin::Var<int> menuChosenLvl("menu.Limit to Level", 0, 0, myExtractor.GetLevels()-1);
    pangolin::Var<int> menuMeanProcessingTime("menu.Mean Processing Time", 0);
    pangolin::Var<int> menuLastFrametime("menu.Last Frame", 0);
    pangolin::Var<bool> menuSaveFeatures("menu.SAVE FEATURES", false, false);

    pangolin::FinishFrame();

    cv::namedWindow(string(imgPath));
    cv::moveWindow(string(imgPath), 240, 260);
    string imgTrackbar = string("image nr");

    int nn = 0;
    cv::createTrackbar(imgTrackbar, string(imgPath), &nn, nImages);
    /** Trackbar call if opencv was compiled without Qt support:
    //cv::createTrackbar(imgTrackbar, string(imgPath), nullptr, nImages);
     */

    cv::displayStatusBar(string(imgPath), "Current Distribution: Bucketing");

    int count = 0;
    bool distributePerLevel = menuDistrPerLvl;
    int soloLvl = -1;

    for(int ni=0; ni<nImages; ni++)
    {
        cv::setTrackbarPos("image nr", string(imgPath), ni);


        if (dataset == tum)
            img = cv::imread(string(imgPath) + "/" + vstrImageFilenamesLeft[ni], CV_LOAD_IMAGE_UNCHANGED);
        else
            img = cv::imread(vstrImageFilenamesLeft[ni], CV_LOAD_IMAGE_UNCHANGED);

        double tframe = vTimestamps[ni];

        if (img.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(imgPath) << "/" << vstrImageFilenamesLeft[ni] << endl;
            exit(EXIT_FAILURE);
        }

        cv::Mat imgGray;
        if (img.channels() > 1)
        {
            if (img.channels() == 4)
            {
                cv::cvtColor(img, imgGray, CV_BGRA2GRAY);
            }
            else if (img.channels() == 3)
            {
                cv::cvtColor(img, imgGray, CV_BGR2GRAY);
            }

        }
        else
        {
            imgGray = img;
        }


        img_t saigaImg = Saiga::MatToImageView<uchar>(imgGray);

        vector<Saiga::KeyPoint> mykpts;
        img_t mydescriptors;

        chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();

        myExtractor(saigaImg, mykpts, mydescriptors, distributePerLevel);

        chrono::high_resolution_clock::time_point t3 = chrono::high_resolution_clock::now();

        auto myduration = chrono::duration_cast<chrono::microseconds>(t3 - t2).count();

        ++count;

        myTotalDuration += myduration;

        pangolin::FinishFrame();

        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);


        DisplayKeypoints(img, mykpts, color, thickness, radius, drawAngular, string(imgPath));
        cv::waitKey(1);

        if (cv::getTrackbarPos(imgTrackbar, string(imgPath)) != ni)
        {
            ni = cv::getTrackbarPos(imgTrackbar, string(imgPath));
        }


        int n = menuNFeatures;
        if (n != nFeatures)
        {
            nFeatures = n;
            myExtractor.SetnFeatures(n);
        }

        float scaleF = menuScaleFactor;
        if (scaleF != myExtractor.GetScaleFactor())
        {
            myExtractor.SetScaleFactor(scaleF);
        }

        int nlvl = menuNLevels;
        if (nlvl != myExtractor.GetLevels())
        {
            myExtractor.SetnLevels(nlvl);
        }

        menuLastFrametime = myduration / 1000;
        menuMeanProcessingTime = myTotalDuration / 1000 / count;

        menuActualkpts = mykpts.size();

        if (menuSoftSSCThreshold != softTh)
        {
            myExtractor.SetSoftSSCThreshold(menuSoftSSCThreshold);
            softTh = menuSoftSSCThreshold;
        }

        if (menuSingleLvlOnly && (soloLvl != menuChosenLvl))
        {
            soloLvl = menuChosenLvl;
            menuDistrPerLvl = true;
            myExtractor.SetLevelToDisplay(soloLvl);
        }

        if (!menuSingleLvlOnly)
        {
            soloLvl = -1;
            myExtractor.SetLevelToDisplay(-1);
        }

        if (menuDistrPerLvl && !distributePerLevel)
            distributePerLevel = true;

        else if (!menuDistrPerLvl && distributePerLevel)
            distributePerLevel = false;

        if (menuPause)
        {
            --ni;
        }


        if (menuSetInitThreshold != FASTThresholdInit || menuSetMinThreshold != FASTThresholdMin)
        {
            FASTThresholdInit = menuSetInitThreshold;
            FASTThresholdMin = menuSetMinThreshold;
            myExtractor.SetFASTThresholds(FASTThresholdInit, FASTThresholdMin);
        }
    }

    cout << "\nTotal running time: " << myTotalDuration/1000 << " milliseconds\n";
}

void DisplayKeypoints(cv::Mat &image, std::vector<Saiga::KeyPoint> &keypoints, cv::Scalar &color,
                      int thickness, int radius, int drawAngular, string windowname)
{
    cv::namedWindow(windowname, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowname, image);

    for (const Saiga::KeyPoint &k : keypoints)
    {
        cv::Point2f point = cv::Point2f(k.pt.x, k.pt.y);
        cv::circle(image, point, radius, color, 1, CV_AA);
        if (drawAngular)
        {
            int len = radius;
            float angleRad =  k.angle * CV_PI / 180.f;
            float cos = std::cos(angleRad);
            float sin = std::sin(angleRad);
            int x = (int)round(point.x + len * cos);
            int y = (int)round(point.y + len * sin);
            cv::Point2f target = cv::Point2f(x, y);
            cv::line(image, point, target, color, thickness, CV_AA);
        }
    }
    cv::imshow(windowname, image);
}


void LoadImagesTUM(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}


void LoadImagesKITTI(const string &strPathToSequence, vector<string> &vstrImageLeft,
                     vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
}


void LoadImagesEUROC(const string &strPathLeft, const string &strPathRight, const string &strPathTimes,
                     vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImageLeft.reserve(5000);
    vstrImageRight.reserve(5000);
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            vstrImageLeft.push_back(strPathLeft + "/" + ss.str() + ".png");
            vstrImageRight.push_back(strPathRight + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            vTimeStamps.push_back(t/1e9);

        }
    }
}


int main(int argc, char **argv)
{
    std::chrono::high_resolution_clock::time_point program_start = std::chrono::high_resolution_clock::now();

    if (argc != 4)
    {
        cerr << "required arguments: <path to settings> <path to image / sequence> "
                "<kitti/tum/euroc>" << endl;
        exit(EXIT_FAILURE);
    }

    string settingsPath = string(argv[1]);
    cv::FileStorage settingsFile(settingsPath, cv::FileStorage::READ);
    if (!settingsFile.isOpened())
    {
        cerr << "Failed to load ORB settings at" << settingsPath << "!" << endl;
        exit(EXIT_FAILURE);
    }

    cout << "\nORB Settings loaded successfully!\n" << endl;


    int nFeatures = settingsFile["ORBextractor.nFeatures"];
    float scaleFactor = settingsFile["ORBextractor.scaleFactor"];
    int nLevels = settingsFile["ORBextractor.nLevels"];
    int FASTThresholdInit = settingsFile["ORBextractor.iniThFAST"];
    int FASTThresholdMin = settingsFile["ORBextractor.minThFAST"];

    cv::Scalar color = cv::Scalar(settingsFile["Color.r"], settingsFile["Color.g"], settingsFile["Color.b"]);
    int thickness = settingsFile["Line.thickness"];
    int radius = settingsFile["Circle.radius"];
    int drawAngular = settingsFile["drawAngular"];

    string imgPath = string(argv[2]);


    string strdataset = string(argv[3]);
    Dataset dataset = strdataset == "kitti" ? kitti : strdataset == "euroc"? euroc : tum;

    SequenceMode(imgPath, nFeatures, scaleFactor, nLevels, FASTThresholdInit, FASTThresholdMin,
                     color, thickness, radius, drawAngular, dataset);


    std::chrono::high_resolution_clock::time_point program_end = std::chrono::high_resolution_clock::now();
    auto program_duration = std::chrono::duration_cast<std::chrono::microseconds>(program_end - program_start).count();


    pangolin::QuitAll();
    std::cout << "\nProgram duration: " << program_duration << " microseconds.\n" <<
              "(equals ~" <<  (float)program_duration / 1000000.f << " seconds)\n";

    return 0;
}