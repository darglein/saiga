/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "controllable_camera.h"

#include "saiga/core/imgui/imgui.h"
namespace Saiga
{
void CameraController::imgui()
{
    ImGui::InputFloat("movementSpeed", &movementSpeed);
    ImGui::InputFloat("movementSpeedFast", &movementSpeedFast);
    ImGui::InputFloat("rotationSpeed", &rotationSpeed);
    ImGui::Checkbox("mouseTurnLocal", &mouseTurnLocal);
}

}  // namespace Saiga
