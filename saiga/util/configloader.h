#pragma once

#include <saiga/config.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>



class SAIGA_GLOBAL ConfigLoader
{
private:
    enum class State{
        ERROR,
        EMPTY,
        LOADED
    };

    std::string getLine(const std::string &key, const std::string &defaultvalue, const std::string &description);
    void parseValue(std::string &value);
    State state;
    std::string name;
    std::fstream stream;
public:

    ConfigLoader();
    ~ConfigLoader();
    bool loadFile(const std::string &name);


    int getInt(const std::string &key , int defaultvalue, const std::string &description="");
    float getFloat(const std::string &key , float defaultvalue, const std::string &description="");
    std::string getString(const std::string &key , const std::string &defaultvalue, const std::string &description="");
};


