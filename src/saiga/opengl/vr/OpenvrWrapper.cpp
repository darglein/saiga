/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "OpenvrWrapper.h"

//#include "openvr/openvr_driver.h"
#include "openvr/openvr.h"

#include <iostream>
namespace Saiga
{
static mat4 ConvertSteamVRMatrixToMatrix4(const vr::HmdMatrix34_t& mat)
{
    mat4 matrixObj;
    matrixObj =
        make_mat4_row_major(mat.m[0][0], mat.m[1][0], mat.m[2][0], 0.0, mat.m[0][1], mat.m[1][1], mat.m[2][1], 0.0,
                            mat.m[0][2], mat.m[1][2], mat.m[2][2], 0.0, mat.m[0][3], mat.m[1][3], mat.m[2][3], 1.0f);

    return matrixObj.transpose();
}

static mat4 ConvertSteamVRMatrixToMatrix4(const vr::HmdMatrix44_t& mat)
{
    mat4 matrixObj;
    matrixObj = make_mat4_row_major(mat.m[0][0], mat.m[1][0], mat.m[2][0], mat.m[3][0], mat.m[0][1], mat.m[1][1],
                                    mat.m[2][1], mat.m[3][1], mat.m[0][2], mat.m[1][2], mat.m[2][2], mat.m[3][2],
                                    mat.m[0][3], mat.m[1][3], mat.m[2][3], mat.m[3][3]);
    return matrixObj.transpose();
}


//-----------------------------------------------------------------------------
// Purpose: Helper to get a string from a tracked device property and turn it
//			into a std::string
//-----------------------------------------------------------------------------
static std::string GetTrackedDeviceString(vr::TrackedDeviceIndex_t unDevice, vr::TrackedDeviceProperty prop,
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


OpenVRWrapper::OpenVRWrapper()
{
    std::string m_strDriver;
    std::string m_strDisplay;

    // Loading the SteamVR Runtime
    vr::EVRInitError eError = vr::VRInitError_None;
    vr_system               = vr::VR_Init(&eError, vr::VRApplication_Scene);

    if (eError != vr::VRInitError_None)
    {
        vr_system = NULL;
        std::cout << "Unable to init VR runtime: " << vr::VR_GetVRInitErrorAsEnglishDescription(eError) << std::endl;
        //        SDL_ShowSimpleMessageBox( SDL_MESSAGEBOX_ERROR, "VR_Init Failed", buf, NULL );
        return;
    }

    //    vr::Prop_DisplayAvailableFrameRates_Float_Array

    m_strDriver  = "No Driver";
    m_strDisplay = "No Display";

    m_strDriver  = GetTrackedDeviceString(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_TrackingSystemName_String);
    m_strDisplay = GetTrackedDeviceString(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_SerialNumber_String);

    std::cout << "OpenVR Driver : " << m_strDriver << std::endl;
    std::cout << "OpenVR Display: " << m_strDisplay << std::endl;



    if (!vr::VRCompositor())
    {
        vr_system = nullptr;
        printf("Compositor initialization failed. See log file for details\n");
        return;
    }

    SAIGA_ASSERT(vr_system);

    vr_system->GetRecommendedRenderTargetSize(&m_nRenderWidth, &m_nRenderHeight);
}
OpenVRWrapper::~OpenVRWrapper()
{
    std::cout << "Shutdown VR" << std::endl;
    vr::VR_Shutdown();
}


mat4 OpenVRWrapper::GetHMDProjectionMatrix(vr::Hmd_Eye nEye, float newPlane, float farPlane)
{
    SAIGA_ASSERT(vr_system);
    vr::HmdMatrix44_t mat = vr_system->GetProjectionMatrix(nEye, newPlane, farPlane);
    return ConvertSteamVRMatrixToMatrix4(mat);
}

mat4 OpenVRWrapper::HeadToEyeModel(vr::Hmd_Eye nEye)
{
    SAIGA_ASSERT(vr_system);

    vr::HmdMatrix34_t matEyeRight = vr_system->GetEyeToHeadTransform(nEye);

    mat4 matrixObj;
    matrixObj = make_mat4_row_major(matEyeRight.m[0][0], matEyeRight.m[1][0], matEyeRight.m[2][0], 0.0,
                                    matEyeRight.m[0][1], matEyeRight.m[1][1], matEyeRight.m[2][1], 0.0,
                                    matEyeRight.m[0][2], matEyeRight.m[1][2], matEyeRight.m[2][2], 0.0,
                                    matEyeRight.m[0][3], matEyeRight.m[1][3], matEyeRight.m[2][3], 1.0f);

    return matrixObj.transpose();
}

void OpenVRWrapper::submitImage(vr::Hmd_Eye nEye, TextureBase* texture)
{
    SAIGA_ASSERT(vr_system);

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
            std::cout << "Device detached " << event.trackedDeviceIndex << std::endl;
            //            std::terminate();
        }
        break;
        case vr::VREvent_TrackedDeviceUpdated:
        {
            //            std::terminate();
            std::cout << "Device updated. " << event.trackedDeviceIndex << std::endl;
        }
        break;
    }
}


std::pair<PerspectiveCamera, PerspectiveCamera> OpenVRWrapper::getEyeCameras(const PerspectiveCamera& camera)
{
    PerspectiveCamera left  = camera;
    PerspectiveCamera right = camera;

    left.proj  = GetHMDProjectionMatrix(vr::Hmd_Eye::Eye_Left, camera.zNear, camera.zFar);
    right.proj = GetHMDProjectionMatrix(vr::Hmd_Eye::Eye_Right, camera.zNear, camera.zFar);

    mat4 vl = HeadToEyeModel(vr::Hmd_Eye::Eye_Left);
    mat4 vr = HeadToEyeModel(vr::Hmd_Eye::Eye_Right);

    auto hmd_model = GetHMDModelMatrix();

    left.model  = camera.model * hmd_model * vl;
    right.model = camera.model * hmd_model * vr;

    left.updateFromModel();
    right.updateFromModel();

    // std::cout << "lr pos " << left.position.transpose() << " | " << right.position.transpose() << std::endl;
    return {left, right};
}
VrDeviceData OpenVRWrapper::GetController(int controller_index)
{
    for (int i = 0; i < m_numTrackedDevices; ++i)
    {
        if (device_data[i].device_class == vr::TrackedDeviceClass_Controller)
        {
            if (controller_index == 0)
            {
                return device_data[i];
            }
            --controller_index;
        }
    }
    return {mat4::Identity()};
}
void OpenVRWrapper::update()
{
    SAIGA_ASSERT(vr_system);

    // Process SteamVR events
    vr::VREvent_t event;
    while (vr_system->PollNextEvent(&event, sizeof(event)))
    {
        ProcessVREvent(event);
    }


    vr::TrackedDevicePose_t m_rTrackedDevicePose[vr::k_unMaxTrackedDeviceCount];
    vr::VRCompositor()->WaitGetPoses(m_rTrackedDevicePose, vr::k_unMaxTrackedDeviceCount, NULL, 0);

    m_numTrackedDevices = 0;
    for (int i = 0; i < vr::k_unMaxTrackedDeviceCount; ++i)
    {
        if (m_rTrackedDevicePose[i].bPoseIsValid)
        {
            auto& device = device_data[m_numTrackedDevices++];

            device.model = ConvertSteamVRMatrixToMatrix4(m_rTrackedDevicePose[i].mDeviceToAbsoluteTracking);
            // std::cout << "got pose " << i << " " << device_data[i].model.col(3).transpose() << std::endl;
            device.device_class = vr_system->GetTrackedDeviceClass(i);

            if (device.device_class == vr::TrackedDeviceClass_Controller)
            {
                vr::VRControllerState_t controllerState;
                vr_system->GetControllerState(i, &controllerState, sizeof(controllerState));

                if (m_lastPacketNum != controllerState.unPacketNum)
                {
                    bool old_left_down  = device.controller_button_left.down;
                    bool old_right_down = device.controller_button_right.down;
                    bool old_up_down    = device.controller_button_up.down;
                    bool old_down_down  = device.controller_button_down.down;
                    bool old_trigger_down = device.controller_trigger.down;

                    device.controller_button_left.pressed  = false;
                    device.controller_button_right.pressed = false;
                    device.controller_button_up.pressed    = false;
                    device.controller_button_down.pressed  = false;
                    device.controller_trigger.pressed  = false;

                    device.controller_button_left.down  = false;
                    device.controller_button_right.down = false;
                    device.controller_button_up.down    = false;
                    device.controller_button_down.down  = false;
                    device.controller_trigger.down  = false;

                    if ((controllerState.ulButtonPressed & (1 << vr::k_EButton_ProximitySensor)) != 0)
                    {
                        device.controller_button_left.down  = controllerState.rAxis[0].x < -0.2f;
                        device.controller_button_right.down = controllerState.rAxis[0].x > 0.2f;
                        device.controller_button_up.down    = controllerState.rAxis[0].y > 0.2f;
                        device.controller_button_down.down  = controllerState.rAxis[0].y < -0.2f;

                        device.controller_button_left.pressed  = device.controller_button_left.down && !old_left_down;
                        device.controller_button_right.pressed = device.controller_button_right.down && !old_right_down;
                        device.controller_button_up.pressed    = device.controller_button_up.down && !old_up_down;
                        device.controller_button_down.pressed  = device.controller_button_down.down && !old_down_down;
                    }

                    if ((controllerState.ulButtonPressed & (1ull << vr::k_EButton_SteamVR_Trigger)) != 0)
                    {
                        device.controller_trigger.down = controllerState.rAxis[1].x > 0.2f;
                        device.controller_trigger.pressed = device.controller_trigger.down && !old_trigger_down;
                    }

                    //if (device.controller_button_left.pressed)
                    //{
                    //    std::cout << "Left\n";
                    //}
                    //
                    //if (device.controller_button_right.pressed)
                    //{
                    //    std::cout << "Right\n";
                    //}
                    //
                    //if (device.controller_button_up.pressed)
                    //{
                    //    std::cout << "Up\n";
                    //}
                    //
                    //if (device.controller_button_down.pressed)
                    //{
                    //    std::cout << "Down\n";
                    //}

                    m_lastPacketNum = controllerState.unPacketNum;
                }
            }
        }
    }
}
mat4 OpenVRWrapper::GetHMDModelMatrix()
{
    return device_data[0].model;
}
vec3 OpenVRWrapper::LookingDirection()
{
    return -GetHMDModelMatrix().col(2).eval().head<3>();
}
}  // namespace Saiga
