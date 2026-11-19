/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// ClassList: the registered game object classes.

#include "saiga/engine/gameplay/ClassList.h"

namespace Saiga
{
namespace Engine
{
const ClassList::ClassInfo* ClassList::find(const std::string& name) const
{
    for (auto& c : classes)
        if (c.factory && name == c.name) return &c;
    return nullptr;
}

size_t ClassList::size() const
{
    size_t n = 0;
    for (auto& c : classes) n += c.factory ? 1 : 0;
    return n;
}

void ClassList::validateRegistration() const
{
    // Engine IDs 0..63 and game IDs 64.. must each be a prefix (every enum value up to the last registered one)
    auto checkRange = [&](size_t first, size_t end, const char* kind)
    {
        size_t last = first;
        bool any    = false;
        for (size_t i = first; i < end && i < classes.size(); ++i)
            if (classes[i].factory)
            {
                last = i;
                any  = true;
            }
        if (!any) return;
        for (size_t i = first; i <= last; ++i)
            SAIGA_ASSERT(classes[i].factory, std::string("ClassList: ") + kind + " class ID " + std::to_string(i) +
                                                 " is not registered (IDs must be contiguous)");
    };
    checkRange(0, kFirstGameClass, "engine");
    checkRange(kFirstGameClass, classes.size(), "game");
}

}  // namespace Engine
}  // namespace Saiga
