/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "BABase.h"

#include "saiga/core/imgui/imgui.h"

namespace Saiga
{
void BAOptions::imgui()
{
    ImGui::InputFloat("huberMono", &huberMono);
    ImGui::InputFloat("huberStereo", &huberStereo);
}



}  // namespace Saiga
