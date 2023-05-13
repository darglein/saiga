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

// ImGui GLFW binding with OpenGL3 + shaders
// In this binding, ImTextureID is used to store an OpenGL 'GLuint' texture identifier. Read the FAQ about ImTextureID
// in imgui.cpp.

// You can copy and use unmodified imgui_impl_* files in your project. See main.cpp for an example of using this.
// If you use this binding you'll need to call 4 functions: ImGui_ImplXXXX_Init(), ImGui_ImplXXXX_NewFrame(),
// ImGui::Render() and ImGui_ImplXXXX_Shutdown(). If you are new to ImGui, see examples/README.txt and documentation at
// the top of imgui.cpp. https://github.com/ocornut/imgui

#include "imgui_impl_glfw_gl3.h"

#include "internal/imgui_impl_glfw.h"
#include "internal/imgui_impl_opengl3.h"
#ifdef SAIGA_USE_GLFW
#    include "saiga/core/glfw/saiga_glfw.h"
#    include "saiga/core/imgui/imgui.h"
#    include "saiga/core/imgui/imgui_main_menu.h"
#    include "saiga/opengl/opengl.h"

#    ifdef _WIN32
#        undef APIENTRY
#        define GLFW_EXPOSE_NATIVE_WIN32
#        define GLFW_EXPOSE_NATIVE_WGL
#        include <GLFW/glfw3native.h>
#    endif

namespace Saiga
{
GLFWwindow* ImGui_GLFW_Renderer::g_Window = NULL;



ImGui_GLFW_Renderer::ImGui_GLFW_Renderer(GLFWwindow* window, const ImGuiParameters& params) : ImGuiRenderer(params)
{
    ImGui_ImplOpenGL3_Init();
    g_Window = window;
    ImGui_ImplGlfw_InitForOpenGL(window, true);
}

ImGui_GLFW_Renderer::~ImGui_GLFW_Renderer()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
}

void ImGui_GLFW_Renderer::beginFrame()
{
    ImGuiIO& io = ImGui::GetIO();
    if (!io.Fonts->IsBuilt())
    {
        ImGui_ImplOpenGL3_CreateFontsTexture();
    }

    ImGui_ImplOpenGL3_NewFrame();
    int display_w, display_h;
    glfwGetFramebufferSize(g_Window, &display_w, &display_h);


    ImGui_ImplGlfw_NewFrame();

    // Start the frame
    ImGui::NewFrame();

    editor_gui.render(display_w, display_h);
    // main_menu.render();
}
void ImGui_GLFW_Renderer::render()
{
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}


void ImGui_GLFW_Renderer::keyPressed(int key, int scancode, int mods)
{
    main_menu.Keypressed(key);
}


void ImGui_GLFW_Renderer::keyReleased(int key, int scancode, int mods) {}

void ImGui_GLFW_Renderer::character(unsigned int codepoint) {}


// bool ImGui_GLFW_Renderer::mouse_button_event(GLFWwindow* window, int button, int action, int mods)
//{
//    if (action == GLFW_PRESS && button >= 0 && button < 3) g_MousePressed[button] = true;
//    return false;
//    //    return wantsCaptureMouse;
//}

void ImGui_GLFW_Renderer::scroll(double xoffset, double yoffset) {}

}  // namespace Saiga
#endif
