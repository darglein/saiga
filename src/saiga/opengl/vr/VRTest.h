/**
 * Copyright (c) 2017 Darius Rückert
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
namespace VR
{
// the base interface between saiga and openvr

class SAIGA_OPENGL_API OpenVRWrapper
{
   public:
    bool init();



    mat4 GetHMDProjectionMatrix(vr::Hmd_Eye nEye, float newPlane, float farPlane);

    // view matrix of the eye relative to the head
    mat4 GetHMDViewMatrix(vr::Hmd_Eye nEye);

    // send an opengl image to the HMD
    void submitImage(vr::Hmd_Eye nEye, TextureBase* texture);

    void handleInput();

    void UpdateHMDMatrixPose();

    // create camera for the left and right eye given the "head" camera.
    std::pair<PerspectiveCamera, PerspectiveCamera> getEyeCameras(const PerspectiveCamera& camera);

    int renderWidth()
    {
        SAIGA_ASSERT(m_pHMD);
        return m_nRenderWidth;
    }
    int renderHeight()
    {
        SAIGA_ASSERT(m_pHMD);
        return m_nRenderHeight;
    }

   private:
    vr::IVRSystem* m_pHMD = nullptr;
    vr::TrackedDevicePose_t m_rTrackedDevicePose[vr::k_unMaxTrackedDeviceCount];
    int m_iValidPoseCount;
    //    int m_iValidPoseCount_Last;
    std::string m_strPoseClasses;  // what classes we saw poses for this frame
    mat4 m_rmat4DevicePose[vr::k_unMaxTrackedDeviceCount];
    char m_rDevClassChar[vr::k_unMaxTrackedDeviceCount];  // for each device, a character representing its class
    mat4 m_mat4HMDPose;
    uint32_t m_nRenderWidth;
    uint32_t m_nRenderHeight;
};



}  // namespace VR
}  // namespace Saiga
