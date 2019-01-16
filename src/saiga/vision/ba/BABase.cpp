/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "BABase.h"

#include "saiga/imgui/imgui.h"

namespace Saiga
{
void BAOptions::imgui()
{
    ImGui::InputInt("maxIterations", &maxIterations);
    ImGui::InputInt("maxIterativeIterations", &maxIterativeIterations);
    ImGui::InputDouble("iterativeTolerance", &iterativeTolerance);

    int currentItem             = (int)solverType;
    static const char* items[3] = {"Brute Force", "Sort and Sweep", "Linked Cell"};
    ImGui::Combo("CollideAlgorithm", &currentItem, items, 3);
    solverType = (SolverType)currentItem;
}



}  // namespace Saiga
