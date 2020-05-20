/**
 * This file is part of ORB-SLAM2.
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
#pragma once

#include "saiga/core/image/imageView.h"
#include "saiga/vision/features/Features.h"
//#include "FeatureDistribution2.h"

#include <list>
#include <vector>

namespace Saiga
{
class SAIGA_VISION_API ORB
{
   public:
    ORB();
    float ComputeAngle(Saiga::ImageView<unsigned char> image, const vec2& pt);
    DescriptorORB ComputeDescriptor(Saiga::ImageView<unsigned char> image, const vec2& point, float angle_degrees);


   private:
    std::vector<int> u_max;
    std::vector<ivec2> descriptor_pattern;
};


}  // namespace Saiga
