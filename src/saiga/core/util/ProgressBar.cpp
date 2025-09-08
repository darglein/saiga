/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ProgressBar.h"

#include "saiga/core/imgui/imgui.h"
namespace Saiga
{

bool ProgressBarManager::Imgui()
{
    std::shared_ptr<ProgressBarBase> bar;
    {
        std::unique_lock l(lock);
        bar = current_bar;


        if (bar && bar->current == bar->end)
        {
            current_bar = {};
        }
    }

    if (!bar) return false;

    int64_t c = bar->current;
    int64_t e = bar->end;


    float f = double(c) / e;

    ImGui::Text("%s, %lld/%lld", bar->prefix.c_str(), c, e);
    ImGui::ProgressBar(f);



    return true;
}
std::shared_ptr<ProgressBarBase> ProgressBarManager::ScopedProgressBar(std::string name, int64_t end)
{
    auto new_bar = std::make_shared<ProgressBarBase>(end, name);
    {
        std::unique_lock l(lock);
        current_bar = new_bar;
    }
    return new_bar;
}
}  // namespace Saiga
