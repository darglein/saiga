#include "util/keyboardbinds.h"


KeyboardBinds::KeyboardBinds(){
    glfw_EventHandler::addKeyListener(this,20);

    IC.add("bind", [this](ICPARAMS){
        nextCommand = args;
        //todo check if command valid
        *(args.os)<<"Press any key to bind the commad '"<<nextCommand.args<<"'"<<std::endl;
        this->waitingForKey = true;
    });
}

bool KeyboardBinds::key_event(GLFWwindow* window, int key, int scancode, int action, int mods){

    if(waitingForKey){
        if(action!=GLFW_PRESS)
            return true;

        auto it = keyMap.find(key);
        if(it!=keyMap.end()){
            *(nextCommand.os)<<"KeyboardBinds: key already bound to '"<<(it->second)<<"'"<<std::endl;
        }else{
            keyMap.insert(mapElement(key,nextCommand.args));
            *(nextCommand.os)<<"Key Bind added: ("<<key<<" "<<nextCommand.args<<")"<<std::endl;
        }
        waitingForKey = false;
        return true;
    }else{
        auto it = keyMap.find(key);
        if(it!=keyMap.end()){
            if(action!=GLFW_PRESS)
                return true;
            IC.execute(it->second);
            return true;
        }
    }


    return false;
}

bool KeyboardBinds::character_event(GLFWwindow* window, unsigned int codepoint){
    return false;
}
