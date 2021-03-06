// ImGui License:
// The MIT License (MIT)

// Copyright (c) 2014-2015 Omar Cornut and ImGui contributors

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


#pragma once
// ImGui GLFW binding with OpenGL3 + shaders
// In this binding, ImTextureID is used to store an OpenGL 'GLuint' texture identifier. Read the FAQ about ImTextureID
// in imgui.cpp.

// You can copy and use unmodified imgui_impl_* files in your project. See main.cpp for an example of using this.
// If you use this binding you'll need to call 4 functions: ImGui_ImplXXXX_Init(), ImGui_ImplXXXX_NewFrame(),
// ImGui::Render() and ImGui_ImplXXXX_Shutdown(). If you are new to ImGui, see examples/README.txt and documentation at
// the top of imgui.cpp. https://github.com/ocornut/imgui


#include "saiga/opengl/imgui/imgui_opengl.h"

#ifdef SAIGA_USE_GLFW
#    include "saiga/core/glfw/glfw_eventhandler.h"

struct GLFWwindow;

namespace Saiga
{
class SAIGA_OPENGL_API ImGui_GLFW_Renderer : public ImGui_GL_Renderer,
                                             public glfw_KeyListener,
                                             public glfw_MouseListener
{
   public:
    ImGui_GLFW_Renderer(GLFWwindow* window, const ImGuiParameters& params);
    ~ImGui_GLFW_Renderer();

    virtual void beginFrame() override;

    void keyPressed(int key, int scancode, int mods) override;
    void keyReleased(int key, int scancode, int mods) override;
    void character(unsigned int codepoint) override;

    void scroll(double xoffset, double yoffset) override;

   protected:
    double g_Time          = 0.0f;
    bool g_MousePressed[3] = {false, false, false};
    float g_MouseWheel     = 0.0f;

    // not nice to make this window static
    static GLFWwindow* g_Window;
    static const char* ImGui_ImplGlfwGL3_GetClipboardText(void* user_data);
    static void ImGui_ImplGlfwGL3_SetClipboardText(void* user_data, const char* text);
};

}  // namespace Saiga
#endif
