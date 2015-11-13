#pragma once

#include <saiga/config.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>



class SAIGA_GLOBAL ConfigLoader
{
private:
    enum class State{
        ERROR,
        EMPTY,
        LOADED
    };

    class ConfigEntry{
    public:
        std::string key;
        std::string value;
        std::string description;
        ConfigEntry(const std::string& key, const std::string& value, const std::string& description);
        std::string toString();
    };

    std::vector<ConfigEntry> entries;


    State state;
    std::string name;
    std::fstream stream;


    std::string getLine(const std::string &key, const std::string &defaultvalue, const std::string &description);
    std::string getLine2(const std::string &key, const std::string &defaultvalue, const std::string &description);

    void parseValue(std::string &value);
public:

    ConfigLoader();
    ~ConfigLoader();
    bool loadFile(const std::string &name);

    bool loadFile2(const std::string &name);
    bool writeFile();


    int getInt(const std::string &key , int defaultvalue, const std::string &description="");
    float getFloat(const std::string &key , float defaultvalue, const std::string &description="");
    std::string getString(const std::string &key , const std::string &defaultvalue, const std::string &description="");

    void setInt(const std::string &key , int value);

};


