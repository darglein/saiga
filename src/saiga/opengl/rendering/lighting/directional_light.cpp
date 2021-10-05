/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "directional_light.h"

#include "saiga/core/geometry/clipping.h"
#include "saiga/core/geometry/obb.h"
#include "saiga/core/imgui/imgui.h"

#include "internal/noGraphicsAPI.h"
namespace Saiga
{
void DirectionalLight::BuildCascades(int _numCascades)
{
    SAIGA_ASSERT(_numCascades > 0);
    this->numCascades = _numCascades;
    orthoBoxes.resize(_numCascades);


    depthCutsRelative.resize(_numCascades + 1);
    depthCuts.resize(_numCascades + 1);

    for (int i = 0; i < _numCascades; ++i)
    {
        depthCutsRelative[i] = float(i) / _numCascades;
    }
    depthCutsRelative.back() = 1.0f;
}


void DirectionalLight::setDirection(const vec3& dir)
{
    direction = normalize(dir);

    vec3 d     = -direction;
    vec3 right = normalize(cross(vec3(1, 1, 0), d));
    vec3 up    = normalize(cross(d, right));


    mat3 m;
    m.col(0) = right;
    m.col(1) = up;
    m.col(2) = d;

    vec3 cp = make_vec3(0);

    this->shadowCamera.setPosition(cp);


    this->shadowCamera.rot = quat_cast(m);

    this->shadowCamera.calculateModel();
    this->shadowCamera.updateFromModel();
}


void DirectionalLight::fitShadowToCamera(Camera* cam)
{
    if (!castShadows) return;
#if 0
    vec3 dir = -direction;
    vec3 right = normalize(cross(vec3(1,1,0),dir));
    vec3 up = normalize(cross(dir,right));

    OBB obb;
    obb.setOrientationScale( normalize(right), normalize(up), normalize(dir) );

    obb.fitToPoints(0,cam->vertices.data(),8);
    obb.fitToPoints(1,cam->vertices.data(),8);
    obb.fitToPoints(2,cam->vertices.data(),8);


    vec3 increase(0,0,5.0);

    float xDiff = 2.0f * obb.orientationScale.col(0).norm() + increase[0];
    float yDiff = 2.0f * obb.orientationScale.col(1).norm()  + increase[1];
    float zDiff = 2.0f * obb.orientationScale.col(2).norm()  + increase[2];

    shadowNearPlane = 0;
    this->cam.setProj(
                -xDiff / 2.0f ,xDiff / 2.0f,
                -yDiff / 2.0f ,yDiff / 2.0f,
                -zDiff / 2.0f ,zDiff / 2.0f
                );

    this->cam.setPosition( obb.center );

    obb.normalize();
    this->cam.rot = quat_cast( obb.orientationScale );

    this->cam.calculateModel();
    this->cam.updateFromModel();


    //    vec4 test = this->cam.proj * this->cam.view * vec4(obb.center,1);
    //    std::cout << "test " << test << std::endl;
#else
    // other idea use bounding sphere of frustum
    // make sure shadow box aligned to light fits bounding sphere
    // note: camera movement or rotation doesn't change the size of the shadow box anymore
    // translate the box only by texel size increments to remove flickering



    Sphere boundingSphere = cam->boundingSphere;

    for (int i = 0; i < (int)depthCutsRelative.size(); ++i)
    {
        float a      = depthCutsRelative[i];
        depthCuts[i] = (1.0f - a) * cam->zNear + (a)*cam->zFar;
    }

    for (int c = 0; c < numCascades; ++c)
    {
        AABB& orthoBox = orthoBoxes[c];

        {
            PerspectiveCamera* pc = static_cast<PerspectiveCamera*>(cam);
            // compute bounding sphere for cascade

            //        vec3 d = -vec3(cam->model[2]);
            vec3 right = make_vec3(cam->model.col(0));
            vec3 up    = make_vec3(cam->model.col(1));
            vec3 dir   = -make_vec3(cam->model.col(2));


            float zNear = depthCuts[c] - cascadeInterpolateRange;
            float zFar  = depthCuts[c + 1] + cascadeInterpolateRange;
            //        float zNear = cam->zNear;
            //        float zFar = cam->zFar;

            //            float zNear = 1;
            //            float zFar = 10;

            //            if(c == 1){
            //                zNear = 10;
            //                zFar = 50;
            //            }

            //            std::cout << "znear/far: " << zNear << " " << zFar << std::endl;

            float tang = (float)tan(pc->fovy * 0.5);

            float fh = zFar * tang;
            float fw = fh * pc->aspect;

            vec3 nearplanepos = cam->getPosition() + dir * zNear;
            vec3 farplanepos  = cam->getPosition() + dir * zFar;
            vec3 v            = farplanepos + fh * up - fw * right;


            vec3 sphereMid = (nearplanepos + farplanepos) * 0.5f;
            float r        = distance(v, sphereMid);

            boundingSphere.r   = r;
            boundingSphere.pos = sphereMid;
        }

        vec3 lightPos = this->shadowCamera.getPosition();

        float r = boundingSphere.r;
        r       = ceil(r);

        vec3 smsize = make_vec3(make_vec2(shadow_map_size), 128468);

        vec3 texelSize;
        texelSize = 2.0f * r / smsize.array();

        // project the position of the actual camera to light space
        vec3 p = boundingSphere.pos;
        mat3 v = make_mat3(this->shadowCamera.view);
        vec3 t = v * p - v * lightPos;
        t[2]   = -t[2];



        orthoBox.min = t - make_vec3(r);
        orthoBox.max = t + make_vec3(r);

        {
            // move camera in texel size increments
            orthoBox.min = (orthoBox.min.array() / texelSize.array());
            orthoBox.min = (orthoBox.min).array().floor();
            orthoBox.min = orthoBox.min.array() * texelSize.array();

            //            orthoBox.max /= texelSize;
            orthoBox.max = (orthoBox.max.array() / texelSize.array());
            orthoBox.max = (orthoBox.max).array().floor();
            orthoBox.max = orthoBox.max.array() * texelSize.array();
        }
    }


#endif
}

void DirectionalLight::fitNearPlaneToScene(AABB sceneBB)
{
    //    vec3 orthoMin(cam.left,cam.bottom,cam[2]Near);
    //    vec3 orthoMax(cam.right,cam.top,cam[2]Far);


    for (auto& orthoBox : orthoBoxes)
    {
        // transform scene AABB to light space
        auto tris = sceneBB.toTriangles();
        std::vector<PolygonType> trisp;
        for (auto t : tris)
        {
            trisp.push_back(Polygon::toPolygon(t));
        }
        for (auto& p : trisp)
        {
            for (auto& v : p)
            {
                v = make_vec3(this->shadowCamera.view * make_vec4(v, 1));
            }
        }


        // clip triangles of scene AABB to the 4 side planes of the frustum
        for (auto& p : trisp)
        {
            p = Clipping::clipPolygonAxisAlignedPlane(p, 0, orthoBox.min[0], true);
            p = Clipping::clipPolygonAxisAlignedPlane(p, 0, orthoBox.max[0], false);

            p = Clipping::clipPolygonAxisAlignedPlane(p, 1, orthoBox.min[1], true);
            p = Clipping::clipPolygonAxisAlignedPlane(p, 1, orthoBox.max[1], false);
        }

        float maxZ = -1e-10;
        float minZ = 1e10;

        for (auto& p : trisp)
        {
            for (auto& v : p)
            {
                minZ = std::min(minZ, v[2]);
                maxZ = std::max(maxZ, v[2]);
            }
        }

        std::swap(minZ, maxZ);
        minZ = -minZ;
        maxZ = -maxZ;

        //    std::cout << "min max Z " << minZ << " " << maxZ << std::endl;
        //    std::cout << "ortho min max Z " << orthoMin[2] << " " << orthoMax[2] << std::endl;


        orthoBox.min[2] = minZ;
        orthoBox.max[2] = maxZ;
    }

    //    this->cam.setProj(orthoBox);
    //    this->cam.setProj(
    //                orthoMin[0] ,orthoMax[0],
    //                orthoMin[1] ,orthoMax[1],
    //                orthoMin[2] ,orthoMax[2]
    //                );
}


void DirectionalLight::setDepthCutsRelative(const std::vector<float>& value)
{
    SAIGA_ASSERT((int)value.size() == numCascades + 1);
    depthCutsRelative = value;
}


std::vector<float> DirectionalLight::getDepthCutsRelative() const
{
    return depthCutsRelative;
}



void DirectionalLight::renderImGui()
{
    LightBase::renderImGui();
    ImGui::InputFloat("ambientIntensity", &ambientIntensity, 0.1, 1);
    ImGui::InputFloat("Cascade Interpolate Range", &cascadeInterpolateRange);
    if (ImGui::Direction("Direction", direction))
    {
        setDirection(direction);
    }
}

}  // namespace Saiga
