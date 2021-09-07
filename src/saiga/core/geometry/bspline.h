/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/math/math.h"
#include "saiga/core/util/DataStructures/ArrayView.h"
#include "saiga/core/util/assert.h"

#include <iostream>
#include <vector>

namespace Saiga
{
template <typename P, typename T>
class Bspline
{
   public:
    Bspline() {}
    Bspline(std::vector<P> controlPoints, int degree = 3) : controlPoints(controlPoints), degree(degree)
    {
        MakeUniformKnots();
    }

    /**
     * @brief Bspline::getPointOnCurve
     * @param a: The position on the curve in the range [0,1]
     */
    P getPointOnCurve(T a);

    void addPoint(const P& p)
    {
        knots.push_back(controlPoints.size());
        controlPoints.push_back(p);
    }

    void SetEndinterpolationknots()
    {
        int numKnots = knots.size();
        for (int i = 0; i < degree; ++i)
        {
            knots[i]                = knots[degree];
            knots[numKnots - i - 1] = knots[numKnots - degree - 1];
        }
    }


    void MakeUniformKnots(bool interpolateEnds = true)
    {
        int numKnots = controlPoints.size() + degree + 1;

        // uniform knot vector
        knots.clear();
        for (int i = 0; i < numKnots; ++i)
        {
            knots.push_back(i);
        }


        if (interpolateEnds)
        {
            SetEndinterpolationknots();
        }
    }

    void MakeDistanceKnots(ArrayView<T> distances)
    {
        SAIGA_ASSERT(distances.size() == controlPoints.size() - 1);
        SetEndinterpolationknots();
    }


    template <typename A, typename B>
    friend std::ostream& operator<<(std::ostream& os, const Bspline<A, B>& bs);


    std::vector<P> controlPoints;
    int degree;

   private:
    std::vector<T> knots;

    // temp storage needed for deBoor
    std::vector<P> dd;


    P deBoor(T u);
};

template <typename P, typename T>
P Bspline<P, T>::getPointOnCurve(T a)
{
    SAIGA_ASSERT(knots.size() > degree);
    a = clamp(a, 0.f, 1.f);
    a = a * (knots[controlPoints.size()] - knots[degree]) + knots[degree];
    return deBoor(a);
}

template <typename P, typename T>
P Bspline<P, T>::deBoor(T u)
{
    int m = controlPoints.size();
    SAIGA_ASSERT(m > degree);

    dd.resize((degree + 1) * (degree + 1));

    // find interval
    int j = -1;
    for (j = degree; j < m - 1; j++)
        if (knots[j] <= u && u < knots[j + 1]) break;

    SAIGA_ASSERT(j != -1);


    for (int i = 0; i <= degree; ++i) dd[i] = controlPoints[j - degree + i];

    for (int k = 1; k <= degree; ++k)
        for (int i = k; i <= degree; ++i)
        {
            T a         = (u - knots[j - degree + i]) / (knots[j + i + 1 - k] - knots[j - degree + i]);
            int ind     = i - k + 1;
            dd[ind - 1] = mix(dd[ind - 1], dd[ind], a);
        }

    return dd[0];
}


template <typename P, typename T>
std::ostream& operator<<(std::ostream& os, const Bspline<P, T>& bs)
{
    os << "Bspline n=" << bs.degree << " m=" << bs.controlPoints.size() << " #knots=" << bs.knots.size() << std::endl;
    std::cout << "knots:  ";
    for (auto n : bs.knots) os << n << ", ";
    os << std::endl;
    std::cout << "control points:  ";
    for (auto n : bs.controlPoints) os << n << ", ";
    return os;
}

}  // namespace Saiga
