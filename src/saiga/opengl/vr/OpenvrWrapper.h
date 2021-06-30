/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/camera/all.h"
#include "saiga/core/math/math.h"
#include "saiga/opengl/texture/all.h"

#ifndef SAIGA_VR
#    error Saiga was build without VR
#endif

#include "openvr/openvr.h"



namespace Saiga
{
struct VrDeviceData
{
    mat4 model;
};


// The interface between saiga and openvr
class SAIGA_OPENGL_API OpenVRWrapper
{
   public:
    OpenVRWrapper();
    ~OpenVRWrapper();



    void update();

    mat4 GetHMDProjectionMatrix(vr::Hmd_Eye nEye, float newPlane, float farPlane);
    mat4 GetHMDModelMatrix();

    // view matrix of the eye relative to the head
    mat4 HeadToEyeModel(vr::Hmd_Eye nEye);

    // send an opengl image to the HMD
    void submitImage(vr::Hmd_Eye nEye, TextureBase* texture);

    vec3 LookingDirection();



    // create camera for the left and right eye given the "head" camera.
    std::pair<PerspectiveCamera, PerspectiveCamera> getEyeCameras(const PerspectiveCamera& camera);

    int renderWidth()
    {
        SAIGA_ASSERT(vr_system);
        return m_nRenderWidth;
    }
    int renderHeight()
    {
        SAIGA_ASSERT(vr_system);
        return m_nRenderHeight;
    }

   private:
    vr::IVRSystem* vr_system = nullptr;

    // All tracked devices, for example the HMD and the controllers
    std::array<VrDeviceData, vr::k_unMaxTrackedDeviceCount> device_data;

    uint32_t m_nRenderWidth;
    uint32_t m_nRenderHeight;
};

}  // namespace Saiga
