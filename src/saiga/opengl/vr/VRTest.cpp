/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "VRTest.h"

#include "openvr/openvr.h"

#include <iostream>
namespace Saiga
{
namespace VR
{
//-----------------------------------------------------------------------------
// Purpose: Helper to get a string from a tracked device property and turn it
//			into a std::string
//-----------------------------------------------------------------------------
std::string GetTrackedDeviceString(vr::TrackedDeviceIndex_t unDevice, vr::TrackedDeviceProperty prop,
                                   vr::TrackedPropertyError* peError = NULL)
{
    uint32_t unRequiredBufferLen = vr::VRSystem()->GetStringTrackedDeviceProperty(unDevice, prop, NULL, 0, peError);
    if (unRequiredBufferLen == 0) return "";

    char* pchBuffer = new char[unRequiredBufferLen];
    unRequiredBufferLen =
        vr::VRSystem()->GetStringTrackedDeviceProperty(unDevice, prop, pchBuffer, unRequiredBufferLen, peError);
    std::string sResult = pchBuffer;
    delete[] pchBuffer;
    return sResult;
}


bool OpenVRWrapper::init()
{
    std::string m_strDriver;
    std::string m_strDisplay;

    // Loading the SteamVR Runtime
    vr::EVRInitError eError = vr::VRInitError_None;
    m_pHMD                  = vr::VR_Init(&eError, vr::VRApplication_Scene);

    if (eError != vr::VRInitError_None)
    {
        m_pHMD = NULL;
        std::cout << "Unable to init VR runtime: " << vr::VR_GetVRInitErrorAsEnglishDescription(eError) << std::endl;
        //        SDL_ShowSimpleMessageBox( SDL_MESSAGEBOX_ERROR, "VR_Init Failed", buf, NULL );
        return false;
    }


    m_strDriver  = "No Driver";
    m_strDisplay = "No Display";

    m_strDriver  = GetTrackedDeviceString(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_TrackingSystemName_String);
    m_strDisplay = GetTrackedDeviceString(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_SerialNumber_String);

    std::cout << "OpenVR Driver : " << m_strDriver << std::endl;
    std::cout << "OpenVR Display: " << m_strDisplay << std::endl;



    if (!vr::VRCompositor())
    {
        printf("Compositor initialization failed. See log file for details\n");
        return false;
    }

    m_pHMD->GetRecommendedRenderTargetSize(&m_nRenderWidth, &m_nRenderHeight);

    return true;
}


mat4 OpenVRWrapper::GetHMDProjectionMatrix(vr::Hmd_Eye nEye, float newPlane, float farPlane)
{
    SAIGA_ASSERT(m_pHMD);

    vr::HmdMatrix44_t mat = m_pHMD->GetProjectionMatrix(nEye, newPlane, farPlane);

    mat4 matrixObj;
    matrixObj << mat.m[0][0], mat.m[1][0], mat.m[2][0], mat.m[3][0], mat.m[0][1], mat.m[1][1], mat.m[2][1], mat.m[3][1],
        mat.m[0][2], mat.m[1][2], mat.m[2][2], mat.m[3][2], mat.m[0][3], mat.m[1][3], mat.m[2][3], mat.m[3][3];

    return matrixObj.transpose();
}

mat4 OpenVRWrapper::GetHMDViewMatrix(vr::Hmd_Eye nEye)
{
    SAIGA_ASSERT(m_pHMD);

    vr::HmdMatrix34_t matEyeRight = m_pHMD->GetEyeToHeadTransform(nEye);

    mat4 matrixObj;
    matrixObj << matEyeRight.m[0][0], matEyeRight.m[1][0], matEyeRight.m[2][0], 0.0, matEyeRight.m[0][1],
        matEyeRight.m[1][1], matEyeRight.m[2][1], 0.0, matEyeRight.m[0][2], matEyeRight.m[1][2], matEyeRight.m[2][2],
        0.0, matEyeRight.m[0][3], matEyeRight.m[1][3], matEyeRight.m[2][3], 1.0f;

    return matrixObj.transpose().inverse();
}

void OpenVRWrapper::submitImage(vr::Hmd_Eye nEye, TextureBase* texture)
{
    SAIGA_ASSERT(m_pHMD);

    vr::Texture_t vrtex          = {(void*)(uintptr_t)texture->getId(), vr::TextureType_OpenGL, vr::ColorSpace_Gamma};
    vr::EVRCompositorError error = vr::VRCompositor()->Submit(nEye, &vrtex);

    if (error != vr::VRCompositorError_None)
    {
        std::cout << "submit error: " << error << std::endl;
    }
}

void ProcessVREvent(const vr::VREvent_t& event)
{
    switch (event.eventType)
    {
        case vr::VREvent_TrackedDeviceDeactivated:
        {
            //            dprintf("Device %u detached.\n", event.trackedDeviceIndex);
        std::cout << "Device detached " <<  event.trackedDeviceIndex << std::endl;
//            std::terminate();
        }
        break;
        case vr::VREvent_TrackedDeviceUpdated:
        {
//            std::terminate();
                        std::cout << "Device updated. " <<  event.trackedDeviceIndex << std::endl;
        }
        break;
    }
}

void OpenVRWrapper::handleInput()
{
    // Process SteamVR events
    vr::VREvent_t event;
    while (m_pHMD->PollNextEvent(&event, sizeof(event)))
    {
        ProcessVREvent(event);
    }
}

mat4 ConvertSteamVRMatrixToMatrix4(const vr::HmdMatrix34_t& matPose)
{
    mat4 matrixObj;
    matrixObj << matPose.m[0][0], matPose.m[1][0], matPose.m[2][0], 0.0, matPose.m[0][1], matPose.m[1][1],
        matPose.m[2][1], 0.0, matPose.m[0][2], matPose.m[1][2], matPose.m[2][2], 0.0, matPose.m[0][3], matPose.m[1][3],
        matPose.m[2][3], 1.0f;
    return matrixObj.transpose();
}


void OpenVRWrapper::UpdateHMDMatrixPose()
{
    if (!m_pHMD) return;

    vr::VRCompositor()->WaitGetPoses(m_rTrackedDevicePose, vr::k_unMaxTrackedDeviceCount, NULL, 0);

    m_iValidPoseCount = 0;
    m_strPoseClasses  = "";
    for (int nDevice = 0; nDevice < vr::k_unMaxTrackedDeviceCount; ++nDevice)
    {
        if (m_rTrackedDevicePose[nDevice].bPoseIsValid)
        {
            m_iValidPoseCount++;
            m_rmat4DevicePose[nDevice] =
                ConvertSteamVRMatrixToMatrix4(m_rTrackedDevicePose[nDevice].mDeviceToAbsoluteTracking);
            if (m_rDevClassChar[nDevice] == 0)
            {
                switch (m_pHMD->GetTrackedDeviceClass(nDevice))
                {
                    case vr::TrackedDeviceClass_Controller:
                        m_rDevClassChar[nDevice] = 'C';
                        break;
                    case vr::TrackedDeviceClass_HMD:
                        m_rDevClassChar[nDevice] = 'H';
                        break;
                    case vr::TrackedDeviceClass_Invalid:
                        m_rDevClassChar[nDevice] = 'I';
                        break;
                    case vr::TrackedDeviceClass_GenericTracker:
                        m_rDevClassChar[nDevice] = 'G';
                        break;
                    case vr::TrackedDeviceClass_TrackingReference:
                        m_rDevClassChar[nDevice] = 'T';
                        break;
                    default:
                        m_rDevClassChar[nDevice] = '?';
                        break;
                }
            }
            m_strPoseClasses += m_rDevClassChar[nDevice];
        }
    }

    if (m_rTrackedDevicePose[vr::k_unTrackedDeviceIndex_Hmd].bPoseIsValid)
    {
        m_mat4HMDPose = m_rmat4DevicePose[vr::k_unTrackedDeviceIndex_Hmd];
        m_mat4HMDPose = m_mat4HMDPose.inverse();
    }
}

std::pair<PerspectiveCamera, PerspectiveCamera> OpenVRWrapper::getEyeCameras(const PerspectiveCamera& camera)
{
    PerspectiveCamera left  = camera;
    PerspectiveCamera right = camera;

    left.proj  = GetHMDProjectionMatrix(vr::Hmd_Eye::Eye_Left, camera.zNear, camera.zFar);
    right.proj = GetHMDProjectionMatrix(vr::Hmd_Eye::Eye_Right, camera.zNear, camera.zFar);

    mat4 vl = GetHMDViewMatrix(vr::Hmd_Eye::Eye_Left);
    mat4 vr = GetHMDViewMatrix(vr::Hmd_Eye::Eye_Right);

    left.view  = vl * m_mat4HMDPose * camera.view;
    right.view = vr * m_mat4HMDPose * camera.view;

//    -58.6051 -19.3548 -56.7005 -7.57974
//    13.4702  -79.1438 -5.04516 -20.4803
//    57.4138  24.3135  -54.5785 15.7798
//    0        -0       0        -81.9813

    std::cout << vl << std::endl;
std::cout << m_mat4HMDPose << std::endl;
std::cout << std::endl;
    left.setModelMatrix(left.view.inverse());
    left.recalculateMatrices();

    right.setModelMatrix(right.view.inverse());
    right.recalculateMatrices();

    return {left, right};
}
}  // namespace VR
}  // namespace Saiga
