/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/Frustum.h"
#include "saiga/core/geometry/object3d.h"
#include "saiga/core/geometry/plane.h"
#include "saiga/core/geometry/sphere.h"
#include "saiga/core/math/math.h"

namespace Saiga
{
struct ViewPort
{
    ivec2 position;
    ivec2 size;
    ViewPort() = default;
    ViewPort(ivec2 position, ivec2 size) : position(position), size(size) {}

    vec4 getVec4() const { return vec4(position(0), position(1), size(0), size(1)); }
};



class SAIGA_CORE_API Camera : public Object3D, public Frustum
{
   public:
    std::string name;

    //    mat4 model;
    mat4 view     = mat4::Identity();
    mat4 proj     = mat4::Identity();
    mat4 viewProj = mat4::Identity();


    float zNear, zFar;
    //    float nw,nh,fw,fh; //dimensions of near and far plane

    //    Frustum frustum;

    bool vulkanTransform = false;

    Camera();
    virtual ~Camera() {}



    void setView(const mat4& v);
    void setView(const vec3& eye, const vec3& center, const vec3& up);


    void setProj(const mat4& p);
    //    virtual void setProj( double fovy, double aspect, double zNear, double zFar){}
    //    virtual void setProj( float left, float right,float bottom,float top,float near,  float far){}

    void updateFromModel();
    void recalculateMatrices();
    virtual void recalculatePlanes();

    // linearize the depth (for rendering)
    float linearDepth(float d) const;
    float nonlinearDepth(float l) const;

    float toViewDepth(float d) const;
    float toNormalizedDepth(float d) const;



    /**
     * Calculates the frustum planes by backprojecting the unit cube to world space.
     */

    void recalculatePlanesFromMatrices();


    vec3 projectToViewSpace(vec3 worldPosition) const;

    vec3 projectToNDC(vec3 worldPosition) const;

    vec2 projectToScreenSpace(vec3 worldPosition, int w, int h) const;

    vec3 inverseprojectToWorldSpace(vec2 ip, float depth, int w, int h) const;


    virtual void recomputeProj() {}

    void imgui();

   private:
    friend std::ostream& operator<<(std::ostream& os, const Camera& ca);
};

//========================= PerspectiveCamera =========================

class SAIGA_CORE_API PerspectiveCamera : public Camera
{
   public:
    float fovy, aspect;
    float tang;
    PerspectiveCamera() {}
    void setProj(float fovy, float aspect, float zNear, float zFar, bool vulkanTransform = false);
    SAIGA_CORE_API friend std::ostream& operator<<(std::ostream& os, const PerspectiveCamera& ca);

    void imgui();
    virtual void recomputeProj() override;
    virtual void recalculatePlanes() override;
};

//========================= OrthographicCamera =========================

class SAIGA_CORE_API OrthographicCamera : public Camera
{
   public:
    float left, right, bottom, top;
    OrthographicCamera() {}
    void setProj(float left, float right, float bottom, float top, float near, float far);
    void setProj(AABB bb);

    SAIGA_CORE_API friend std::ostream& operator<<(std::ostream& os, const OrthographicCamera& ca);

    void imgui();
    virtual void recomputeProj() override;
    virtual void recalculatePlanes() override;
};


/**
 * Equivalent to the uniform block defined in camera.glsl
 * This makes uploading the camera parameters more efficient, because they can be shared in multiple shaders and
 * be upload at once.
 */
struct CameraDataGLSL
{
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    vec4 camera_position;

    CameraDataGLSL() {}
    CameraDataGLSL(Camera* cam)
    {
        view            = cam->view;
        proj            = cam->proj;
        viewProj        = proj * view;
        camera_position = cam->position;
    }
};

}  // namespace Saiga
