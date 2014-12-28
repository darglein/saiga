#pragma once



#include <vector>
#include <string>
#include <iostream>
#include <memory>

#include <FreeImagePlus.h>

using std::string;
using std::cout;
using std::endl;


class Freeimage{
public:
    Freeimage();
    void load(string filename);

};
