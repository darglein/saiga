/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include <vector>
#include "saiga/util/glm.h"

namespace Saiga {

template <typename P>
class Bspline{
public:
    Bspline(std::vector<P> controlPoints, int degree);
    ~Bspline();

    /**
     * @brief Bspline::getPointOnCurve
     * @param a: The position on the curve in the range [0,1]
     */
    P getPointOnCurve(float a);
private:

    std::vector<float> knots;
    std::vector<P> controlPoints;
    int degree;

    //temp storage needed for deBoor
//    P* dd;
    std::vector<P> dd;

    /**
     * @brief deBoor
      Evaluate the b-spline curve.
     * @param n The degree
     * @param m The number of control points
     * @param d The control point vector
     * @param t The knot vector
     * @param u The position from [0,1]
     * @return
     */
    P deBoor(int n, int m, P *d, float* t, float u);
};

template <typename P>
Bspline<P>::Bspline(std::vector<P> controlPoints, int degree) : controlPoints(controlPoints), degree(degree)
{
    int numKnots = controlPoints.size() + degree;

    //uniform knot vector
    for (int i = 0; i < numKnots; ++i){
        knots.push_back(i);
    }

//    dd = new P[(degree+1)*(degree+1)];
    dd.resize((degree+1)*(degree+1));
}

template <typename P>
Bspline<P>::~Bspline(){
//    delete[] dd;
}

template <typename P>
P Bspline<P>::getPointOnCurve(float a)
{
    a = glm::clamp(a,0.f,1.f);
    return deBoor(degree,controlPoints.size(), &controlPoints[0], &knots[0], a*(knots[controlPoints.size()] - knots[degree]) + knots[degree]);
}

template <typename P>
P Bspline<P>::deBoor(int n, int m, P *d, float* t, float u)
{
    // find interval
    int j;
    for (j = n; j < m-1; j++)
        if (t[j] <= u && u < t[j+1])
            break;


    for (int i = 0; i <= n; ++i)
        dd[i] = d[j-n+i];

    for (int k = 1; k <= n; ++k)
        for (int i = k; i <= n; ++i){
            float a = (u-t[j-n+i])/(t[j+i+1-k]-t[j-n+i]);
            int ind = i-k+1;
            dd[ind-1] = (1-a) * dd[ind-1] +  a * dd[ind];
        }

    return dd[0];
}

//vec2 deCasteljau(float u) const
//{
//    std::vector<vec2> tmp = points;
//    int size = tmp.size();

//    while (size > 1)
//    {
//        for (int i = 1; i < size; ++i)
//        {
//            tmp[i - 1] = vec2((1 - u) * tmp[i - 1].x + u * tmp[i].x, (1 - u) * tmp[i - 1].y + u * tmp[i].y);
//        }
//        size--;
//    }


//    return tmp[0];
//}

}
