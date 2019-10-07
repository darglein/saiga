#include "ORBextractor.h"

#include "ORBconstants.h"

#include <iterator>

#ifdef ORB_FIXED_DURATION
#    include <chrono>
#    include <unistd.h>
#endif

#include "saiga/core/image/templatedImage.h"
#include "saiga/extra/opencv/opencv.h"

#include "GaussianBlur.h"

#include <saiga/core/util/Range.h>

namespace Saiga
{
float ORBextractor::IntensityCentroidAngle(const uchar* pointer, int step)
{
    // m10 ~ x^1y^0, m01 ~ x^0y^1
    int x, y, m01 = 0, m10 = 0;

    int half_patch = PATCH_SIZE / 2;

    for (x = -half_patch; x <= half_patch; ++x)
    {
        m10 += x * pointer[x];
    }

    for (y = 1; y <= half_patch; ++y)
    {
        int cols = CIRCULAR_ROWS[y];
        int sumY = 0;
        for (x = -cols; x <= cols; ++x)
        {
            int uptown   = pointer[x + y * step];
            int downtown = pointer[x - y * step];
            sumY += uptown - downtown;
            m10 += x * (uptown + downtown);
        }
        m01 += y * sumY;
    }
    // return atan2f((float)m01, (float)m10) * RAD_TO_DEG_FACTOR;
    return cv::fastAtan2((float)m01, (float)m10);
}


ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels, int _iniThFAST, int _minThFAST)
    : nfeatures(_nfeatures),
      scaleFactor(_scaleFactor),
      nlevels(_nlevels),
      iniThFAST(_iniThFAST),
      minThFAST(_minThFAST),
      stepsChanged(true),
      levelToDisplay(-1),
      prevDims(-1, -1),
      pixelOffset{},
      fast(_iniThFAST, _minThFAST, _nlevels)
#ifdef ORB_FEATURE_FILEINTERFACE_ENABLED
      ,
      fileInterface(),
      saveFeatures(false),
      usePrecomputedFeatures(false)
#endif
{
    SetnLevels(_nlevels);

    SetFASTThresholds(_iniThFAST, _minThFAST);

    SetnFeatures(nfeatures);

    const int nPoints      = 512;
    const auto tempPattern = (const Point2i*)bit_pattern_31_;
    std::copy(tempPattern, tempPattern + nPoints, std::back_inserter(pattern));
}

void ORBextractor::SetnFeatures(int n)
{
    if (n < 1 || n > 10000) return;

    nfeatures = n;

    float fac                      = 1.f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures * (1.f - fac) / (1.f - (float)pow((double)fac, (double)nlevels));

    int sumFeatures = 0;
    for (int i = 0; i < nlevels - 1; ++i)
    {
        nfeaturesPerLevelVec[i] = round(nDesiredFeaturesPerScale);
        sumFeatures += nfeaturesPerLevelVec[i];
        nDesiredFeaturesPerScale *= fac;
    }
    nfeaturesPerLevelVec[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);
}

void ORBextractor::SetFASTThresholds(int ini, int min)
{
    if ((ini == iniThFAST && min == minThFAST)) return;

    iniThFAST = std::min(255, std::max(1, ini));
    minThFAST = std::min(iniThFAST, std::max(1, min));

    fast.SetFASTThresholds(ini, min);
}


void ORBextractor::operator()(cv::InputArray inputImage, cv::InputArray mask, std::vector<kpt_t>& resultKeypoints,
                              cv::OutputArray outputDescriptors, FeatureDistribution& distribution)
{
    // this->operator()(inputImage, mask, resultKeypoints, outputDescriptors, true);

    cv::Mat image = inputImage.getMat();
    SAIGA_ASSERT(image.type() == CV_8UC1, "Image must be single-channel!");

    Saiga::ImageView<uchar> saigaImage = Saiga::MatToImageView<uchar>(image);

    Saiga::TemplatedImage<uchar> saigaDescriptors;

    this->operator()(saigaImage, resultKeypoints, saigaDescriptors, distribution, true);

    cv::Mat cvRes = Saiga::ImageViewToMat<uchar>(saigaDescriptors);
}


/** @overload
 * @param inputImage single channel img-matrix
 * @param mask ignored
 * @param resultKeypoints keypoint vector in which results will be stored
 * @param outputDescriptors matrix in which descriptors will be stored
 * @param distributePerLevel true->distribute kpts per octave, false->distribute kpts per image
 */
void ORBextractor::operator()(img_t image, std::vector<kpt_t>& resultKeypoints,
                              Saiga::TemplatedImage<uchar>& outputDescriptors, FeatureDistribution& distribution,
                              bool distributePerLevel)
{
    cv::setNumThreads(0);

#ifdef ORB_FIXED_DURATION
    using clk                 = std::chrono::high_resolution_clock;
    clk::time_point funcEntry = clk::now();
#endif

    SAIGA_ASSERT(image.size() > 0, "image empty");

    if (prevDims.x != image.cols || prevDims.y != image.rows) stepsChanged = true;
#ifdef ORB_USE_OPENCV
    std::vector<cv::Mat> cvPyramid(nlevels);
    ComputeScalePyramid(image, cvPyramid);
#else
    imagePyramid[0] = image;
    std::vector<Saiga::TemplatedImage<uchar>> templatedImages(nlevels);

    for (int lvl = 1; lvl < nlevels; ++lvl)
    {
        int width  = round(image.cols * invScaleFactorVec[lvl]);
        int height = round(image.rows * invScaleFactorVec[lvl]);

        templatedImages[lvl] = Saiga::TemplatedImage<uchar>(height, width);
        image.copyScaleLinear((templatedImages[lvl].getImageView()));
        imagePyramid[lvl] = templatedImages[lvl].getImageView();
    }
#endif

    SetSteps();

    std::vector<std::vector<kpt_t>> allkpts(nlevels);

    DivideAndFAST(allkpts, distribution, 30, distributePerLevel);

    if (!distributePerLevel)
    {
        ComputeAngles(allkpts);
        for (int lvl = 0; lvl < nlevels; ++lvl)
        {
            resultKeypoints.insert(resultKeypoints.end(), allkpts[lvl].begin(), allkpts[lvl].end());
        }
        for (auto& kpt : resultKeypoints)
        {
            float size  = PATCH_SIZE * scaleFactorVec[kpt.octave];
            float scale = scaleFactorVec[kpt.octave];
            if (kpt.octave)
            {
                kpt.size = size;
                kpt.point *= scale;
            }
        }
        distribution.SetImageSize(make_ivec2(image.cols, image.rows));
        distribution(resultKeypoints);
    }

    if (distributePerLevel)
    {
        ComputeAngles(allkpts);
    }

    int nkpts = 0;
    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        nkpts += allkpts[lvl].size();
    }

    Saiga::TemplatedImage<uchar> t(std::max(nkpts, 1), 32);
    img_t BRIEFdescriptors = t.getImageView();

    ComputeDescriptors(allkpts, BRIEFdescriptors);
    outputDescriptors = t;

    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        resultKeypoints.insert(resultKeypoints.end(), allkpts[lvl].begin(), allkpts[lvl].end());
    }

    if (distributePerLevel)
    {
        for (auto& kpt : resultKeypoints)
        {
            float size  = PATCH_SIZE * scaleFactorVec[kpt.octave];
            float scale = scaleFactorVec[kpt.octave];
            if (kpt.octave)
            {
                kpt.size = size;
                kpt.point *= scale;
            }
        }
    }

#ifdef ORB_FEATURE_FILEINTERFACE_ENABLED
    if (saveFeatures)
    {
        fileInterface.SaveFeatures(resultKeypoints);
        fileInterface.SaveDescriptors(BRIEFdescriptors);
    }
#endif


#ifdef ORB_FIXED_DURATION
    // ensure feature detection always takes <orb_duration> ms
    unsigned long maxDuration = ORB_FIXED_DURATION;
    clk::time_point funcExit  = clk::now();
    auto funcDuration         = std::chrono::duration_cast<std::chrono::microseconds>(funcExit - funcEntry).count();
    SAIGA_ASSERT(funcDuration <= maxDuration, "ORB took too long");
    if (funcDuration < maxDuration)
    {
        auto sleeptime = maxDuration - funcDuration;
        usleep(sleeptime);
    }
#endif
}


void ORBextractor::ComputeAngles(std::vector<std::vector<kpt_t>>& allkpts)
{
#pragma omp parallel for num_threads(2) schedule(dynamic)
    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        for (int i = 0; i < (int)allkpts[lvl].size(); ++i)
        {
            allkpts[lvl][i].angle = IntensityCentroidAngle(
                &imagePyramid[lvl](allkpts[lvl][i].point.y(), allkpts[lvl][i].point.x()), imagePyramid[lvl].pitchBytes);
        }
    }
}


void ORBextractor::ComputeDescriptors(std::vector<std::vector<kpt_t>>& allkpts, img_t& descriptors)
{
    const auto degToRadFactor = (float)(CV_PI / 180.f);
    const Point2i* p          = &pattern[0];

    //    int current = 0;

    std::vector<int> scan;
    scan.push_back(0);

    for (int i = 1; i < nlevels; ++i)
    {
        scan[i] = scan[i - 1] + allkpts[i - 1].size();
    }



#pragma omp parallel for num_threads(2) schedule(dynamic)
    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        int current = scan[lvl];
        Saiga::TemplatedImage<uchar> t(imagePyramid[lvl].rows, imagePyramid[lvl].cols);
        img_t lvlClone = t.getImageView();
        imagePyramid[lvl].copyTo(lvlClone);
#ifdef ORB_USE_OPENCV
        cv::GaussianBlur(Saiga::ImageViewToMat<uchar>(lvlClone), Saiga::ImageViewToMat<uchar>(lvlClone), cv::Size(7, 7),
                         2, 2);
#else
        SaigaORB::GaussianBlur<uchar>(lvlClone, lvlClone, 7, 7, 2, 2);
#endif
        const int step = (int)lvlClone.pitchBytes;

        int i = 0, nkpts = (int)allkpts[lvl].size();
        for (int k = 0; k < nkpts; ++k, ++current)
        {
            const kpt_t& kpt          = allkpts[lvl][k];
            auto descPointer          = descriptors.rowPtr(current);  // ptr to beginning of current descriptor
            const uchar* pixelPointer = &lvlClone(kpt.point.y(), kpt.point.x());  // ptr to kpt in img

            float angleRad = kpt.angle * degToRadFactor;
            auto a = (float)cos(angleRad), b = (float)sin(angleRad);

            int byte = 0, v0, v1, idx0, idx1;
            for (i = 0; i <= 512; i += 2)
            {
                if (i > 0 && i % 16 == 0)  // working byte full
                {
                    descPointer[i / 16 - 1] = (uchar)byte;  // write current byte to descriptor-mat
                    byte                    = 0;            // reset working byte
                    if (i == 512)  // break out after writing very last byte, so oob indices aren't accessed
                        break;
                }

                idx0 = round(p[i].x * a - p[i].y * b) + round(p[i].x * b + p[i].y * a) * step;
                idx1 = round(p[i + 1].x * a - p[i + 1].y * b) + round(p[i + 1].x * b + p[i + 1].y * a) * step;

                v0 = pixelPointer[idx0];
                v1 = pixelPointer[idx1];

                byte |= (v0 < v1) << ((i % 16) / 2);  // write comparison bit to current byte
            }
        }
    }
}


/**
 * @param allkpts KeyPoint vector in which the result will be stored
 * @param mode decides which method to call for keypoint distribution over image, see Distribution.h
 * @param divideImage  true-->divide image into cellSize x cellSize cells, run FAST per cell
 * @param cellSize must be greater than 16 and lesser than min(rows, cols) of smallest image in pyramid
 */
void ORBextractor::DivideAndFAST(std::vector<std::vector<kpt_t>>& allkpts, FeatureDistribution& distribution,
                                 int cellSize, bool distributePerLevel)
{
    const int minimumX = EDGE_THRESHOLD - 3, minimumY = minimumX;
    {
        int c = std::min(imagePyramid[nlevels - 1].rows, imagePyramid[nlevels - 1].cols);
        SAIGA_ASSERT(cellSize < c && cellSize > 16);

        int minLvl = 0, maxLvl = nlevels;
        if (levelToDisplay != -1)
        {
            minLvl = levelToDisplay;
            maxLvl = minLvl + 1;
        }
//#pragma omp parallel for default(none) shared(minLvl, maxLvl, cellSize, distributePerLevel, allkpts)
#pragma omp parallel for num_threads(2) schedule(dynamic)
        for (int lvl = minLvl; lvl < maxLvl; ++lvl)
        {
            std::vector<kpt_t> levelkpts;
            levelkpts.clear();
            levelkpts.reserve(nfeatures * 10);

            const int maximumX = imagePyramid[lvl].cols - EDGE_THRESHOLD + 3;
            const int maximumY = imagePyramid[lvl].rows - EDGE_THRESHOLD + 3;
            const float width  = maximumX - minimumX;
            const float height = maximumY - minimumY;

            const int npatchesInX = width / cellSize;
            const int npatchesInY = height / cellSize;
            const int patchWidth  = ceil(width / npatchesInX);
            const int patchHeight = ceil(height / npatchesInY);

            for (int py = 0; py < npatchesInY; ++py)
            {
                float startY = minimumY + py * patchHeight;
                float endY   = startY + patchHeight + 6;

                if (startY >= maximumY - 3)
                {
                    continue;
                }

                if (endY > maximumY)
                {
                    endY = maximumY;
                }


                for (int px = 0; px < npatchesInX; ++px)
                {
                    float startX = minimumX + px * patchWidth;
                    float endX   = startX + patchWidth + 6;

                    if (startX >= maximumX - 6)
                    {
                        continue;
                    }


                    if (endX > maximumX)
                    {
                        endX = maximumX;
                    }

                    std::vector<kpt_t> patchkpts;
                    img_t patch = imagePyramid[lvl].subImageView(startY, startX, endY - startY, endX - startX);

                    fast.FAST(patch, patchkpts, iniThFAST, lvl);
                    if (patchkpts.empty())
                    {
                        fast.FAST(patch, patchkpts, minThFAST, lvl);
                    }

                    if (patchkpts.empty()) continue;

                    for (auto& kpt : patchkpts)
                    {
                        kpt.point.y() += py * patchHeight;
                        kpt.point.x() += px * patchWidth;
                        levelkpts.emplace_back(kpt);
                    }
                }
            }
            if (distributePerLevel)
            {
                distribution.SetN(nfeaturesPerLevelVec[lvl]);
                distribution.SetImageSize(make_ivec2(maximumX - minimumX, maximumY - minimumY));
                distribution(levelkpts);
            }


            for (auto& kpt : levelkpts)
            {
                kpt.point.y() += minimumY;
                kpt.point.x() += minimumX;
                kpt.octave = lvl;
            }
            featuresPerLevelActual[lvl] = levelkpts.size();
            allkpts[lvl]                = levelkpts;
        }
    }
}

void ORBextractor::ComputeScalePyramid(img_t& image, std::vector<cv::Mat>& tmpPyramid)
{
    cv::Mat img = Saiga::ImageViewToMat(image);
    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        int width  = (int)round((float)img.cols * invScaleFactorVec[lvl]);  // 1.f / getScale(lvl));
        int height = (int)round((float)img.rows * invScaleFactorVec[lvl]);  // 1.f / getScale(lvl));

        int doubleEdge     = EDGE_THRESHOLD * 2;
        int borderedWidth  = width + doubleEdge;
        int borderedHeight = height + doubleEdge;

        // Size sz(width, height);
        // Size borderedSize(borderedWidth, borderedHeight);

        cv::Mat borderedImg(borderedHeight, borderedWidth, img.type());
        cv::Range rowRange(EDGE_THRESHOLD, height + EDGE_THRESHOLD);
        cv::Range colRange(EDGE_THRESHOLD, width + EDGE_THRESHOLD);

        // imagePyramid[lvl] = borderedImg(cv::Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, width, height));
        tmpPyramid[lvl] = borderedImg(rowRange, colRange);


#if 1

        if (lvl)
        {
            cv::resize(tmpPyramid[lvl - 1], tmpPyramid[lvl], cv::Size(width, height), 0, 0, CV_INTER_LINEAR);

            cv::copyMakeBorder(tmpPyramid[lvl], borderedImg, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               EDGE_THRESHOLD, cv::BORDER_REFLECT_101 + cv::BORDER_ISOLATED);
        }
        else
        {
            cv::copyMakeBorder(img, borderedImg, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               cv::BORDER_REFLECT_101);
        }
#endif
    }
    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        imagePyramid[lvl] = Saiga::MatToImageView<uchar>(tmpPyramid[lvl]);
    }
}


void ORBextractor::SetSteps()
{
    if (stepsChanged)
    {
        std::vector<int> steps(nlevels);
        for (int lvl = 0; lvl < nlevels; ++lvl)
        {
            steps[lvl] = (int)imagePyramid[lvl].pitchBytes;
        }
        fast.SetLevels(nlevels);
        fast.SetStepVector(steps);

        stepsChanged = false;
    }
}

void ORBextractor::SetnLevels(int n)
{
    nlevels = std::max(std::min(12, n), 2);
    scaleFactorVec.resize(nlevels);
    invScaleFactorVec.resize(nlevels);
    imagePyramid.resize(nlevels);
    nfeaturesPerLevelVec.resize(nlevels);
    featuresPerLevelActual.resize(nlevels);
    levelSigma2Vec.resize(nlevels);
    invLevelSigma2Vec.resize(nlevels);
    pixelOffset.resize(nlevels * CIRCLE_SIZE);

    SetScaleFactor(scaleFactor);

    float fac                      = 1.f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures * (1.f - fac) / (1.f - (float)pow((double)fac, (double)nlevels));
    int sumFeatures                = 0;
    for (int i = 0; i < nlevels - 1; ++i)
    {
        nfeaturesPerLevelVec[i] = round(nDesiredFeaturesPerScale);
        sumFeatures += nfeaturesPerLevelVec[i];
        nDesiredFeaturesPerScale *= fac;
    }
    nfeaturesPerLevelVec[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);
}

void ORBextractor::SetScaleFactor(float s)
{
    stepsChanged         = true;
    scaleFactor          = std::max(std::min(1.5f, s), 1.001f);
    scaleFactorVec[0]    = 1.f;
    invScaleFactorVec[0] = 1.f;

    SetSteps();

    for (int i = 1; i < nlevels; ++i)
    {
        scaleFactorVec[i]    = scaleFactor * scaleFactorVec[i - 1];
        invScaleFactorVec[i] = 1 / scaleFactorVec[i];

        levelSigma2Vec[i]    = scaleFactorVec[i] * scaleFactorVec[i];
        invLevelSigma2Vec[i] = 1.f / levelSigma2Vec[i];
    }
}
}  // namespace Saiga
