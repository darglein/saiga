#pragma once
#include <vector>
#include "glm/glm.hpp"

template <typename P>
class Bspline{
public:
    Bspline::Bspline(std::vector<P> controlPoints, int degree) : controlPoints(controlPoints), degree(degree)
    {
        int numKnots = controlPoints.size() + degree;

        //uniform knot vector
        for (int i = 0; i < numKnots; ++i){
            knots.push_back(i);
        }

        dd = new P[(degree+1) * (degree+1)];
    }

    ~Bspline(){
        delete dd;
    }

    /**
     * @brief Bspline::getPointOnCurve
     * @param a: The position on the curve in the range [0,1]
     */
    P Bspline::getPointOnCurve(float a)
    {
        a = glm::clamp(a,0.f,1.f);
        return deBoor(degree,controlPoints.size(), &controlPoints[0], &knots[0], a*(knots[controlPoints.size()] - knots[degree]) + knots[degree]);
    }
private:

    std::vector<float> knots;
    std::vector<P> controlPoints;
    int degree;

    //temp storage need for deBoor
    P* dd;


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
    P Bspline::deBoor(int n, int m, P *d, float* t, float u)
    {
        // find interval
        int j;
        for (j = n; j < m-1; j++)
            if (t[j] <= u && u < t[j+1])
                break;

    #define access(x,y) ((x) + (n+1) * (y))

        for (int i = 0; i <= n; i++)
            dd[access(0,i)] = d[j-n+i];

        for (int k = 1; k <= n; k++)
            for (int i = k; i <= n; i++){
                float a = (u-t[j-n+i])/(t[j+i+1-k]-t[j-n+i]);
                dd[access(k,i)] = (1-a) * dd[access(k-1,i-1)] +  a * dd[access(k-1,i)];
            }

        return dd[access(n,n)];
    #undef access
    }
};
