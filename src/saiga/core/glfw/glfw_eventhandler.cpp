/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "glfw_eventhandler.h"

#include "saiga/core/glfw/saiga_glfw.h"

#include "internal/noGraphicsAPI.h"

#include <algorithm>

#include "glfw_joystick.h"

namespace Saiga
{
std::vector<glfw_JoystickListener*> glfw_EventHandler::joystickListener;
std::vector<glfw_KeyListener*> glfw_EventHandler::keyListener;
std::vector<glfw_MouseListener*> glfw_EventHandler::mouseListener;
std::vector<glfw_ResizeListener*> glfw_EventHandler::resizeListener;



glfw_JoystickListener::glfw_JoystickListener()
{
    glfw_EventHandler::joystickListener.push_back(this);
}

glfw_JoystickListener::~glfw_JoystickListener()
{
    auto& v = glfw_EventHandler::joystickListener;
    v.erase(std::remove(v.begin(), v.end(), this), v.end());
}


glfw_KeyListener::glfw_KeyListener()
{
    glfw_EventHandler::keyListener.push_back(this);
}

glfw_KeyListener::~glfw_KeyListener()
{
    auto& v = glfw_EventHandler::keyListener;
    v.erase(std::remove(v.begin(), v.end(), this), v.end());
}

glfw_MouseListener::glfw_MouseListener()
{
    glfw_EventHandler::mouseListener.push_back(this);
}

glfw_MouseListener::~glfw_MouseListener()
{
    auto& v = glfw_EventHandler::mouseListener;
    v.erase(std::remove(v.begin(), v.end(), this), v.end());
}

glfw_ResizeListener::glfw_ResizeListener()
{
    glfw_EventHandler::resizeListener.push_back(this);
}

glfw_ResizeListener::~glfw_ResizeListener()
{
    auto& v = glfw_EventHandler::resizeListener;
    v.erase(std::remove(v.begin(), v.end(), this), v.end());
}


void glfw_EventHandler::joystick_key_callback(int button, bool pressed)
{
    for (auto& rl : joystickListener)
    {
        rl->joystick_event(button, pressed);
    }
}

void glfw_EventHandler::window_size_callback(GLFWwindow* window, int width, int height)
{
    for (auto& rl : resizeListener)
    {
        rl->window_size_callback(width, height);
    }
}

void glfw_EventHandler::cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    mouse.setPosition(ivec2(xpos, ypos));

    for (auto& ml : mouseListener)
    {
        ml->mouseMoved(xpos, ypos);
    }
}

void glfw_EventHandler::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    int mx = mouse.getX();
    int my = mouse.getY();
    if (action == GLFW_PRESS)
    {
        mouse.setKeyState(button, true);
        for (auto& ml : mouseListener)
        {
            ml->mousePressed(button, mx, my);
        }
    }
    else if (action == GLFW_RELEASE)
    {
        mouse.setKeyState(button, false);
        for (auto& ml : mouseListener)
        {
            ml->mouseReleased(button, mx, my);
        }
    }
}

void glfw_EventHandler::scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    for (auto& ml : mouseListener)
    {
        ml->scroll(xoffset, yoffset);
    }
}

void glfw_EventHandler::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        keyboard.setKeyState(key, true);
        for (auto& kl : keyListener)
        {
            kl->keyPressed(key, scancode, mods);
        }
    }
    else if (action == GLFW_RELEASE)
    {
        keyboard.setKeyState(key, false);
        for (auto& kl : keyListener)
        {
            kl->keyReleased(key, scancode, mods);
        }
    }
}

void glfw_EventHandler::character_callback(GLFWwindow* window, unsigned int codepoint)
{
    for (auto& kl : keyListener)
    {
        kl->character(codepoint);
    }
}

}  // namespace Saiga
