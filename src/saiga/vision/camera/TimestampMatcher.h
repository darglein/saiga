/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/util/assert.h"
#include "saiga/vision/VisionTypes.h"

namespace Saiga
{
/**
 * Computer vision datasets from multiple sensors are usually stored with the timestamp.
 * So this file helps finding the best matching sensor data.
 *
 * @brief The TimestampMatcher class
 */
class TimestampMatcher
{
   public:
    static int findNearestNeighbour(double leftTime, const std::vector<double> rightTimes)
    {
        // Returns an iterator pointing to the first element in the range [first, last) that is not less than (i.e.
        // greater or equal to) value, or last if no such element is found.
        auto equalOrGreaterIt = std::lower_bound(rightTimes.begin(), rightTimes.end(), leftTime);

        // didn't find a lower bound
        // remove, because there might be clamping
        if (equalOrGreaterIt == rightTimes.end())
        {
            return -1;
        }

        // lower bound smaller than the smallest from right
        // -> accept only if equal
        if (equalOrGreaterIt == rightTimes.begin() && leftTime < rightTimes.front())
        {
            return -1;
        }

        auto equalOrSmallerIt = equalOrGreaterIt - 1;

        SAIGA_ASSERT(equalOrSmallerIt != equalOrGreaterIt);
        auto nearestNeighbour = std::abs(leftTime - *equalOrGreaterIt) < std::abs(leftTime - *equalOrSmallerIt)
                                    ? equalOrGreaterIt
                                    : equalOrSmallerIt;

        return nearestNeighbour - rightTimes.begin();
    }


    static std::tuple<int, int, double> findLowHighAlphaNeighbour(double leftTime, const std::vector<double> rightTimes)
    {
        // Returns an iterator pointing to the first element in the range [first, last) that is not less than (i.e.
        // greater or equal to) value, or last if no such element is found.
        auto equalOrGreaterIt = std::lower_bound(rightTimes.begin(), rightTimes.end(), leftTime);

        // we don't want to reference the border because there might be clamping
        if (equalOrGreaterIt == rightTimes.end() || equalOrGreaterIt == rightTimes.begin()) return {-1, -1, 0};

        auto equalOrSmallerIt = equalOrGreaterIt - 1;

        double alpha = (leftTime - *equalOrSmallerIt) / (*equalOrGreaterIt - *equalOrSmallerIt);

        return {equalOrSmallerIt - rightTimes.begin(), equalOrGreaterIt - rightTimes.begin(), alpha};
    }
};

}  // namespace Saiga
