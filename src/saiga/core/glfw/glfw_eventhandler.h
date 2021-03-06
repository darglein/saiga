/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/config.h"
#include "saiga/core/util/keyboard.h"
#include "saiga/core/util/mouse.h"

#include <algorithm>
#include <vector>

struct GLFWwindow;

namespace Saiga
{
class SAIGA_CORE_API glfw_JoystickListener
{
   public:
    glfw_JoystickListener();
    virtual ~glfw_JoystickListener();
    virtual bool joystick_event(int button, bool pressed) = 0;
};

class SAIGA_CORE_API glfw_KeyListener
{
   public:
    glfw_KeyListener();
    virtual ~glfw_KeyListener();
    virtual bool key_event(GLFWwindow* window, int key, int scancode, int action, int mods) = 0;
    virtual bool character_event(GLFWwindow* window, unsigned int codepoint)                = 0;
};

class SAIGA_CORE_API glfw_MouseListener
{
   public:
    glfw_MouseListener();
    virtual ~glfw_MouseListener();
    virtual bool cursor_position_event(GLFWwindow* window, double xpos, double ypos)      = 0;
    virtual bool mouse_button_event(GLFWwindow* window, int button, int action, int mods) = 0;
    virtual bool scroll_event(GLFWwindow* window, double xoffset, double yoffset)         = 0;
};

class SAIGA_CORE_API glfw_ResizeListener
{
   public:
    glfw_ResizeListener();
    virtual ~glfw_ResizeListener();
    virtual bool window_size_callback(GLFWwindow* window, int width, int height) = 0;
};

class SAIGA_CORE_API glfw_EventHandler
{
   public:
    static std::vector<glfw_JoystickListener*> joystickListener;
    static std::vector<glfw_KeyListener*> keyListener;
    static std::vector<glfw_MouseListener*> mouseListener;
    static std::vector<glfw_ResizeListener*> resizeListener;


    // called from glfw window joystick //NOTE it is called inside the update step, not the glfwPollEvents function
    static void joystick_key_callback(int button, bool pressed);

    static void window_size_callback(GLFWwindow* window, int width, int height);

    // these functions will be called by glfw from the method glfwPollEvents();
    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void character_callback(GLFWwindow* window, unsigned int codepoint);
};


}  // namespace Saiga
