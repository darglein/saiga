//ImGui License:
//The MIT License (MIT)

//Copyright (c) 2014-2015 Omar Cornut and ImGui contributors

//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:

//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

//-----------------------------------------------------------------------------
// USER IMPLEMENTATION
// This file contains compile-time options for ImGui.
// Other options (memory allocation overrides, callbacks, etc.) can be set at runtime via the ImGuiIO structure - ImGui::GetIO().
//-----------------------------------------------------------------------------

#pragma once

#include "saiga/config.h"
#include "saiga/util/assert.h"
#include "saiga/util/glm.h"

//---- Define assertion handler. Defaults to calling SAIGA_ASSERT().
#define IM_ASSERT(_EXPR)  SAIGA_ASSERT(_EXPR)

//---- Define attributes of all API symbols declarations, e.g. for DLL under Windows.
#define IMGUI_API SAIGA_GLOBAL
//#define IMGUI_API __declspec( dllimport )

//---- Include imgui_user.h at the end of imgui.h
#define IMGUI_INCLUDE_IMGUI_USER_H

//---- Don't implement default handlers for Windows (so as not to link with OpenClipboard() and others Win32 functions)
//#define IMGUI_DISABLE_WIN32_DEFAULT_CLIPBOARD_FUNCS
//#define IMGUI_DISABLE_WIN32_DEFAULT_IME_FUNCS

//---- Don't implement help and test window functionality (ShowUserGuide()/ShowStyleEditor()/ShowTestWindow() methods will be empty)
//#define IMGUI_DISABLE_TEST_WINDOWS

//---- Don't define obsolete functions names
//#define IMGUI_DISABLE_OBSOLETE_FUNCTIONS

//---- Implement STB libraries in a namespace to avoid conflicts
//#define IMGUI_STB_NAMESPACE     ImGuiStb

//---- Define constructor and implicit cast operators to convert back<>forth from your math types and ImVec2/ImVec4.

#define IM_VEC2_CLASS_EXTRA                                                 \
        ImVec2(const glm::vec2& f) { x = f.x; y = f.y; }                       \
        operator glm::vec2() const { return glm::vec2(x,y); }

#define IM_VEC3_CLASS_EXTRA                                                 \
        ImVec3(const glm::vec3& f) { x = f.x; y = f.y; z = f.z;}     \
        operator glm::vec3() const { return glm::vec3(x,y,z); }

#define IM_VEC4_CLASS_EXTRA                                                 \
        ImVec4(const glm::vec4& f) { x = f.x; y = f.y; z = f.z; w = f.w; }     \
        operator glm::vec4() const { return glm::vec4(x,y,z,w); }


//---- Tip: You can add extra functions within the ImGui:: namespace, here or in your own headers files.
//---- e.g. create variants of the ImGui::Value() helper for your low-level math types, or your own widgets/helpers.
/*
namespace ImGui
{
    void    Value(const char* prefix, const MyMatrix44& v, const char* float_format = NULL);
}
*/

