#pragma once
#include "saiga/util/glm.h"
#include "saiga/util/assert.h"
#include <vector>

//the control point type. For example: float, vec2, vec3, vec4
template <typename P>
class Bezier{
public:
    std::vector<P> controlPoints;
    int N; //== degree of bezier curve == #controlpoints-1

    Bezier(std::vector<P> controlPoints);

    //evaluate with Casteljau.
    P evaluate(float t) const;

    //increases the degree by one. This adds one controlpoint without changing the curve.
    void degreeElavation();

    //subdivide this bezier curve c times. Each step doubles the number of output curves.
    std::vector<Bezier<P>> subdivide(int c = 1);

    //returns a line that approximates the bezier curve. subdivisions=0 returns the controlpolygon.
    std::vector<P> createLine(int subdivisions = 0);
};

template <typename P>
Bezier<P>::Bezier(std::vector<P> _controlPoints) : controlPoints(_controlPoints), N(_controlPoints.size()-1)
{
    SAIGA_ASSERT(N >= 0);
}

template <typename P>
P Bezier<P>::evaluate(float t) const
{
    std::vector<P> tmp = controlPoints;
    for (int j = 0 ; j < N ; ++j)
    {
        for (int i = 0; i < N-j; ++i)
        {
            tmp[i] = (1 - t) * tmp[i] + t * tmp[i+1];
        }
    }
    return tmp[0];
}


template <typename P>
void Bezier<P>::degreeElavation(){
    std::vector<P> newControlPoints(N + 2);
    newControlPoints[0] = controlPoints[0];
    newControlPoints[N + 1] = controlPoints[N];

    for(int i = 1 ; i < N+1 ; ++i){
        float alpha = (float)i / (N+1);
        newControlPoints[i] = alpha * controlPoints[i-1] + (1 - alpha) * controlPoints[i];
    }
    controlPoints = newControlPoints;
    N++;
}


template <typename P>
std::vector<Bezier<P> > Bezier<P>::subdivide(int c)
{
    if(c == 0){
        return {*this};
    }
    float t = 0.5f;
    std::vector<P> left(N + 1);
    std::vector<P> right(N + 1);

    left[0] = controlPoints[0];
    right[N] = controlPoints[N];

    std::vector<P> tmp = controlPoints;
    for (int j = 0 ; j < N ; ++j)
    {
        for (int i = 0; i < N-j; ++i)
        {
            tmp[i] = (1 - t) * tmp[i] + t * tmp[i+1];
        }
        left[j+1] = tmp[0];
        right[N-j-1] = tmp[N-j-1];
    }


    auto l = Bezier<P>(left).subdivide(c-1);
    auto r = Bezier<P>(right).subdivide(c-1);

    l.insert(l.end(),r.begin(),r.end());

    return l;
}

template <typename P>
std::vector<P> Bezier<P>::createLine(int subdivisions)
{
    auto sub = subdivide(subdivisions);
    std::vector<P> line;
    for(auto bez : sub){
        //don't add last controlpoint because it's also the first of the next control polygon
        for(int i = 0 ; i < (int)bez.controlPoints.size()-1; ++i){
            line.push_back(bez.controlPoints[i]);
        }
    }
    //add last cp of the last curve
    line.push_back(sub.back().controlPoints.back());
    return line;
}
