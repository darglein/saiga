#include "saiga/util/keyboard.h"
#include "saiga/util/assert.h"

Keyboard keyboard(1024);


Keyboard::Keyboard(int numKeys)
{
    keystate.resize(numKeys);
    for(int i = 0 ; i < numKeys ; ++i){
        keystate[i] = 0;
    }
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


void Keyboard::setKeyState(int key, int state)
{
    assert(key >= 0 && key < (int)keystate.size());
    assert(state == 0 || state == 1);
    keystate[key] = state;
}
