/**
 * This file is part of ORB-SLAM2.
 * This file is based on the file orb.cpp from the OpenCV library (see BSD license below).
 *
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
 * For more information see <https://github.com/raulmur/ORB_SLAM2>
 *
 * ORB-SLAM2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
 */
/**
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ORBExtractor.h"


#ifdef SAIGA_USE_OPENCV

#    include "saiga/core/time/all.h"
#    include "saiga/core/util/Thread/omp.h"
#    include "saiga/vision/opencv/opencv.h"

#    include <opencv2/core/core.hpp>
#    include <opencv2/features2d/features2d.hpp>
#    include <opencv2/highgui/highgui.hpp>
#    include <opencv2/imgproc/imgproc.hpp>


namespace Saiga
{
const int PATCH_SIZE     = 31;
const int EDGE_THRESHOLD = 19;


ORBExtractor::ORBExtractor(int _nfeatures, float _scaleFactor, int _nlevels, int _iniThFAST, int _minThFAST,
                           int threads)
    : num_levels(_nlevels), th_fast(_iniThFAST), th_fast_min(_minThFAST), num_threads(threads)
{
    pyramid = Saiga::ScalePyramid(_nlevels, _scaleFactor, _nfeatures);
    levels.resize(num_levels);
}

void ORBExtractor::DetectKeypoints()
{
    const float W = 30;
#    pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int level = 0; level < num_levels; ++level)
    {
        auto& level_data = levels[level];
        level_data.keypoints_tmp.clear();

        auto image = Saiga::ImageViewToMat(level_data.image);

        const int minBorderX = EDGE_THRESHOLD - 3;
        const int minBorderY = minBorderX;
        const int maxBorderX = level_data.image.cols - EDGE_THRESHOLD + 3;
        const int maxBorderY = level_data.image.rows - EDGE_THRESHOLD + 3;


        const float width  = (maxBorderX - minBorderX);
        const float height = (maxBorderY - minBorderY);

        const int nCols = width / W;
        const int nRows = height / W;
        const int wCell = ceil(width / nCols);
        const int hCell = ceil(height / nRows);

        for (int i = 0; i < nRows; i++)
        {
            const float iniY = minBorderY + i * hCell;
            float maxY       = iniY + hCell + 6;

            if (iniY >= maxBorderY - 3) continue;
            if (maxY > maxBorderY) maxY = maxBorderY;

            for (int j = 0; j < nCols; j++)
            {
                const float iniX = minBorderX + j * wCell;
                float maxX       = iniX + wCell + 6;
                if (iniX >= maxBorderX - 6) continue;
                if (maxX > maxBorderX) maxX = maxBorderX;


                std::vector<cv::KeyPoint> cv_KeysCell;

                FAST(image.rowRange(iniY, maxY).colRange(iniX, maxX), cv_KeysCell, th_fast, true);
                int dis_before = cv_KeysCell.size();

                if (cv_KeysCell.empty())
                {
                    FAST(image.rowRange(iniY, maxY).colRange(iniX, maxX), cv_KeysCell, th_fast_min, true);
                    dis_before = cv_KeysCell.size();
                }

                for (auto& cvkp : cv_KeysCell)
                {
                    KeypointType kp;
                    kp.point    = Saiga::vec2(cvkp.pt.x, cvkp.pt.y);
                    kp.size     = cvkp.size;
                    kp.angle    = cvkp.angle;
                    kp.response = cvkp.response;
                    kp.octave   = cvkp.octave;


                    kp.point.x() += j * wCell;
                    kp.point.y() += i * hCell;
                    level_data.keypoints_tmp.push_back(kp);
                }
            }
        }

        level_data.keypoints_tmp =
            level_data.distributor.Distribute(level_data.keypoints_tmp, Saiga::vec2(minBorderX, minBorderY),
                                              Saiga::vec2(maxBorderX, maxBorderY), pyramid.Features(level));

        const int scaledPatchSize = PATCH_SIZE * pyramid.Scale(level);

        for (auto& kp : level_data.keypoints_tmp)
        {
            kp.point.x() += minBorderX;
            kp.point.y() += minBorderY;
            kp.octave = level;
            kp.size   = scaledPatchSize;
            kp.angle  = orb.ComputeAngle(level_data.image, kp.point);
        }
    }
}



void ORBExtractor::Detect(Saiga::ImageView<unsigned char> inputImage, std::vector<KeypointType>& _keypoints,
                          std::vector<Saiga::DescriptorORB>& outputDescriptors)
{
    cv::setNumThreads(1);
    if (inputImage.empty()) return;


    outputDescriptors.clear();
    ComputePyramid(inputImage);
    DetectKeypoints();


    int nkeypoints = 0;
    for (int level = 0; level < num_levels; ++level)
    {
        levels[level].offset = nkeypoints;
        int n                = (int)levels[level].keypoints_tmp.size();
        nkeypoints += n;
    }

    outputDescriptors.resize(nkeypoints);
    _keypoints.resize(nkeypoints);

#    pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int level = 0; level < num_levels; ++level)
    {
        auto& level_data    = levels[level];
        auto& keypoints     = level_data.keypoints_tmp;
        int nkeypointsLevel = (int)keypoints.size();

        if (nkeypointsLevel == 0) continue;

        auto image       = Saiga::ImageViewToMat(level_data.image);
        auto image_gauss = Saiga::ImageViewToMat(level_data.image_gauss.getImageView());

        cv::GaussianBlur(image, image_gauss, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);

        int offset = level_data.offset;
        for (size_t i = 0; i < keypoints.size(); i++)
        {
            outputDescriptors[offset + i] =
                orb.ComputeDescriptor(level_data.image_gauss.getImageView(), keypoints[i].point, keypoints[i].angle);
        }

        // Scale keypoint coordinates
        if (level != 0)
        {
            float scale = pyramid.Scale(level);
            for (auto& kp : keypoints) kp.point *= scale;
        }
        // And add the keypoints to the output
        for (int i = 0; i < nkeypointsLevel; ++i)
        {
            _keypoints[offset + i] = keypoints[i];
        }
    }
}

void ORBExtractor::AllocatePyramid(int rows, int cols)
{
    SAIGA_ASSERT(!levels.empty());
    if (levels.front().image.valid()) return;

    for (int level = 0; level < num_levels; ++level)
    {
        auto& level_data = levels[level];

        float scale    = pyramid.InverseScale(level);
        int level_rows = Saiga::iRound(rows * scale);
        int level_cols = Saiga::iRound(cols * scale);

        int level_rows_with_border = level_rows + EDGE_THRESHOLD * 2;
        int level_cols_with_border = level_cols + EDGE_THRESHOLD * 2;

        level_data.image_with_border.create(level_rows_with_border, level_cols_with_border);
        level_data.image = level_data.image_with_border.getImageView().subImageView(EDGE_THRESHOLD, EDGE_THRESHOLD,
                                                                                    level_rows, level_cols);
        level_data.image_gauss.create(level_rows, level_cols);

        level_data.keypoints_tmp.reserve(pyramid.total_num_features * 10);
    }
}

void ORBExtractor::ComputePyramid(Saiga::ImageView<unsigned char> image)
{
    AllocatePyramid(image.rows, image.cols);

    cv::Mat cv_image = Saiga::ImageViewToMat(image);
    assert(cv_image.type() == CV_8UC1);

    SAIGA_ASSERT(!levels.empty());
    cv::copyMakeBorder(cv_image, Saiga::ImageViewToMat(levels.front().image_with_border.getImageView()), EDGE_THRESHOLD,
                       EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, cv::BORDER_REFLECT_101);

    for (int level = 1; level < num_levels; ++level)
    {
        auto& level_data      = levels[level];
        auto& level_data_prev = levels[level - 1];

        auto image      = Saiga::ImageViewToMat(level_data.image);
        auto image_prev = Saiga::ImageViewToMat(level_data_prev.image);
        auto temp       = Saiga::ImageViewToMat(levels.front().image_with_border.getImageView());


        cv::resize(image_prev, image, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_LINEAR);

        cv::copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           cv::BORDER_REFLECT_101 + cv::BORDER_ISOLATED);
    }
}

}  // namespace Saiga

#endif
