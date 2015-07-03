#pragma once
#include "libhello/text/text.h"

#include <iostream>



class SAIGA_GLOBAL DynamicText : public Text{
public:
    int size; //dynamic text has fixed size
    DynamicText(int size);
    virtual ~DynamicText(){}
    void updateGLBuffer(int start);

    void compressText(std::string &str, int &start);
    char updateText(std::string &str, int start);


};
