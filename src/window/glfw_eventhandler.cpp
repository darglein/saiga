#include "window/glfw_eventhandler.h"


std::vector<glfw_EventHandler::Listener<glfw_KeyListener>> glfw_EventHandler::keyListener;
std::vector<glfw_EventHandler::Listener<glfw_MouseListener>> glfw_EventHandler::mouseListener;


void glfw_EventHandler::addKeyListener(glfw_KeyListener* kl,int priority){
//    keyListener.push_back(Listener<glfw_KeyListener>(kl,priority));

    auto iter=keyListener.begin();
    for(;iter!=keyListener.end();++iter){
        if((*iter).priority<priority)
            break;
    }
    keyListener.insert(iter,Listener<glfw_KeyListener>(kl,priority));
}
void glfw_EventHandler::addMouseListener(glfw_MouseListener* ml,int priority){
//    mouseListener.push_back(Listener<glfw_MouseListener>(ml,priority));

    auto iter=mouseListener.begin();
    for(;iter!=mouseListener.end();++iter){
        if((*iter).priority<priority)
            break;
    }
    mouseListener.insert(iter,Listener<glfw_MouseListener>(ml,priority));
}

void glfw_EventHandler::cursor_position_callback(GLFWwindow* window, double xpos, double ypos){
    //forward event to all listeners
    for(auto &ml : mouseListener){
        if(ml.listener->cursor_position_event(window,xpos,ypos))
            return;
    }
}

void glfw_EventHandler::mouse_button_callback(GLFWwindow* window, int button, int action, int mods){
    //forward event to all listeners
    for(auto &ml : mouseListener){
        if(ml.listener->mouse_button_event(window,button,action,mods))
            return;
    }
}

void glfw_EventHandler::scroll_callback(GLFWwindow* window, double xoffset, double yoffset){
    //forward event to all listeners
    for(auto &ml : mouseListener){
        if(ml.listener->scroll_event(window,xoffset,yoffset))
                return;
    }
}

void glfw_EventHandler::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){
    //forward event to all listeners
    for(auto &kl : keyListener){
        if(kl.listener->key_event(window,key,scancode,action,mods))
           return;
    }
}

void glfw_EventHandler::character_callback(GLFWwindow* window, unsigned int codepoint){
    //forward event to all listeners
    for(auto &kl : keyListener){
        if(kl.listener->character_event(window,codepoint))
           return;
    }
}
