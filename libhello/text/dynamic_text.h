#pragma once
#include "libhello/text/text.h"

#include <iostream>



class DynamicText : public Text{
public:
    int size; //dynamic text has fixed size
    DynamicText(int size);
    virtual ~DynamicText(){}
    void updateGLBuffer(int start);

    void compressText(string &str, int &start);
    char updateText(string &str, int start);


};
