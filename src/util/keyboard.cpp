#include "saiga/util/keyboard.h"
#include "saiga/util/assert.h"
#include <iostream>

Keyboard keyboard(1024);


Keyboard::Keyboard(int numKeys)
{
    keystate.resize(numKeys,0);
}

int Keyboard::getKeyState(int key) {
    assert(key >= 0 && key < (int)keystate.size());
    return keystate[key];
}

int Keyboard::getMappedKeyState(int mappedKey, const std::vector<int> &keymap) {
    assert(mappedKey >= 0 && mappedKey < (int)keymap.size());
    int key = keymap[mappedKey];
    assert(key >= 0 && key < (int)keystate.size());
    return keystate[key];
}


int Keyboard::setKeyState(int key, int state)
{
    if(key >= 0 && key < (int)keystate.size()){
        assert(state == 0 || state == 1);
        int old = keystate[key];
        keystate[key] = state;
        if(old ^ state)
            return key;
    }
    return -1;

}

void Keyboard::printKeyState()
{
    std::cout << "[";
    for(auto k : keystate){
        std::cout << k << ",";
    }
    std::cout << "]" << std::endl;
}
