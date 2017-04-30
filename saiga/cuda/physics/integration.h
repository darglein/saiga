#pragma once

#include "saiga/util/glm.h"
#include "saiga/cuda/cudaHelper.h"



namespace CUDA{


struct Derivative{
    vec3 dx;
    vec3 dv;
};

HD Derivative evaluate( const vec3& p, const vec3& v,
                                float dT,
                                const vec3& a,
                                const Derivative &d )
{

    vec3 x = p + d.dx*dT;
    vec3 p2 = v + d.dv*dT;


    Derivative output;
    output.dx = p2;
    output.dv = a;
    return output;
}


HD void integrateRungeKutta4XP(vec3& p, vec3& v, const vec3& a, float dT)
{



    Derivative k0,k1,k2,k3;

    k0 = evaluate(p,v,0,a,k0);
    k1 = evaluate( p,v,  dT*0.5f,a, k0 );
    k2 = evaluate( p,v,  dT*0.5f,a, k1 );
    k3 = evaluate( p,v,  dT,a, k2 );

    vec3 dxdt = 1.0f / 6.0f *
            ( k0.dx + 2.0f*(k1.dx + k2.dx) + k3.dx );

    vec3 dvdt = 1.0f / 6.0f *
            ( k0.dv + 2.0f*(k1.dv + k2.dv) + k3.dv );

    p = p + dxdt * dT;
    v = v + dvdt * dT;
}

HD void integrateEuler(vec3& p, vec3& v, const vec3& a, float dT)
{
    p = p + v * dT;
    v = v + a * dT;
}


}
