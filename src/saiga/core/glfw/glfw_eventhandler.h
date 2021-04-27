/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/config.h"
#include "saiga/core/util/keyboard.h"
#include "saiga/core/util/mouse.h"

#include <vector>

struct GLFWwindow;

namespace Saiga
{
class SAIGA_CORE_API glfw_JoystickListener
{
   public:
    glfw_JoystickListener();
    virtual ~glfw_JoystickListener();
    virtual void joystick_event(int button, bool pressed) {}
};

class SAIGA_CORE_API glfw_KeyListener
{
   public:
    glfw_KeyListener();
    virtual ~glfw_KeyListener();
    virtual void keyPressed(int key, int scancode, int mods) {}
    virtual void keyReleased(int key, int scancode, int mods) {}
    virtual void character(unsigned int codepoint) {}
};

class SAIGA_CORE_API glfw_MouseListener
{
   public:
    glfw_MouseListener();
    virtual ~glfw_MouseListener();

    virtual void mouseMoved(int x, int y) {}
    virtual void mousePressed(int key, int x, int y) {}
    virtual void mouseReleased(int key, int x, int y) {}
    virtual void scroll(double xoffset, double yoffset) {}


    //    virtual bool cursor_position_event(GLFWwindow* window, double xpos, double ypos) { return false; }
    //    virtual bool mouse_button_event(GLFWwindow* window, int button, int action, int mods) { return false; }
    //    virtual bool scroll_event(GLFWwindow* window, double xoffset, double yoffset) { return false; }
};

class SAIGA_CORE_API glfw_ResizeListener
{
   public:
    glfw_ResizeListener();
    virtual ~glfw_ResizeListener();
    virtual void window_size_callback(int width, int height) {}
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
