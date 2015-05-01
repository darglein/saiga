#include "glfw/glfw_eventhandler.h"

#include <iostream>
#include <algorithm>

std::vector<glfw_EventHandler::Listener<glfw_KeyListener>> glfw_EventHandler::keyListener;
std::vector<glfw_EventHandler::Listener<glfw_MouseListener>> glfw_EventHandler::mouseListener;


void glfw_EventHandler::addKeyListener(glfw_KeyListener* kl,int priority){
    //    keyListener.push_back(Listener<glfw_KeyListener>(kl,priority));

    Listener<glfw_KeyListener> l(kl,priority);
    auto it = std::find(keyListener.begin(),keyListener.end(),l);

    if(it!=keyListener.end())
        return;

    auto iter=keyListener.begin();
    for(;iter!=keyListener.end();++iter){
        if((*iter).priority<priority)
            break;
    }
    keyListener.insert(iter,l);

//    std::cout<<"addKeyListener "<<keyListener.size()<<" "<<priority<<std::endl;
}
void glfw_EventHandler::addMouseListener(glfw_MouseListener* ml,int priority){
    //    mouseListener.push_back(Listener<glfw_MouseListener>(ml,priority));
    Listener<glfw_MouseListener> l(ml,priority);
    auto it = std::find(mouseListener.begin(),mouseListener.end(),l);

    if(it!=mouseListener.end())
        return;

    auto iter=mouseListener.begin();
    for(;iter!=mouseListener.end();++iter){
        if((*iter).priority<priority)
            break;
    }
    mouseListener.insert(iter,l);
}

void glfw_EventHandler::removeKeyListener(glfw_KeyListener *kl)
{
    auto it=keyListener.begin();
    for(;it!=keyListener.end();++it){
        if(it->listener==kl){
            keyListener.erase(it);
            return;
        }
    }

}

void glfw_EventHandler::removeMouseListener(glfw_MouseListener *ml)
{
    auto it=mouseListener.begin();
    for(;it!=mouseListener.end();++it){
        if(it->listener==ml){
            mouseListener.erase(it);
            return;
        }
    }

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
//    std::cout<<"keyy callback"<<std::endl;
    for(auto &kl : keyListener){
//        std::cout<<"key event "<<action<<" "<<kl.priority<<std::endl;
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
