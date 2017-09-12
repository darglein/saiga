/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/configloader.h"

namespace Saiga {

using std::cout;
using std::endl;

ConfigLoader::ConfigEntry::ConfigEntry(const std::string &key, const std::string &value, const std::string &description)
    :key(key),value(value),description(description)
{

}

std::string ConfigLoader::ConfigEntry::toString()
{
    std::string desc = "";
    if (description != ""){
        int maxAlign = 27;
        int fillSize = maxAlign-(key.size() + value.size());
        if(fillSize<=0) fillSize = 1;

        std::string fill = std::string(fillSize, ' ');
        desc = fill + "#" + description;
    }
    return key+"="+value+desc;
}


ConfigLoader::ConfigLoader() : state(State::EMPTY)
{
}

ConfigLoader::~ConfigLoader(){
    if(state == State::LOADED){
        stream.close();
    }
}

bool ConfigLoader::loadFile2(const std::string &_name)
{
    entries.clear();
    this->name = _name;


    try {
        stream.open (name.c_str(), std::fstream::in | std::fstream::out | std::fstream::app);
    }
    catch (const std::fstream::failure &e) {
        std::cout<< e.what() <<std::endl;
        std::cout << "Exception opening/reading file\n";
        return false;
    }



    std::string line;
    while (std::getline(stream, line))
    {
        auto pos = line.find('=');

        if(pos!=std::string::npos){

            std::string key = line.substr(0,pos);
            std::string value = line.substr(pos+1,line.length()-pos-1);
            std::string comment;

            auto commentStart = value.find('#');

            if(commentStart != std::string::npos){
                comment = value.substr(commentStart+1);
                value = value.substr(0,commentStart);
            }


            //string trimming
            auto strBegin = value.find_first_not_of(" ");
            auto strEnd = value.find_last_not_of(" ");
            auto strRange = strEnd - strBegin + 1;
            value = value.substr(strBegin, strRange);


            entries.emplace_back(key,value,comment);
#ifdef SAIGA_DEBUG
            std::cout << "Key " << key
                      << "  Value='" << value << "'"
                      << " Comment: " << comment << std::endl;
#endif
        }
    }

    stream.close();
    state = State::LOADED;
    return true;
}

bool ConfigLoader::writeFile()
{
    if(!update){
#ifdef SAIGA_DEBUG
        cout << "Config file " << name << " not updated." << endl;
#endif
        return true;
    }

    try {
        stream.open (name,  std::fstream::out);
    }
    catch (const std::fstream::failure &e) {
        std::cout<< e.what() <<std::endl;
        std::cout << "Exception opening/reading file\n";
        return false;
    }

    for(ConfigEntry& ce : entries){
        stream << ce.toString() << std::endl;
    }
    cout << "Saved config file " << name << endl;

    stream.close();
    return true;
}

void ConfigLoader::parseValue(std::string &value)
{
    //removes description, leading and ending spaces
    auto commentStart = value.find('#');

    if(commentStart != std::string::npos){
        value = value.substr(0,commentStart);
    }


    //string trimming
    auto strBegin = value.find_first_not_of(" ");
    auto strEnd = value.find_last_not_of(" ");
    auto strRange = strEnd - strBegin + 1;
    value = value.substr(strBegin, strRange);
}

std::string ConfigLoader::getLine2(const std::string &key, const std::string &defaultvalue, const std::string &description)
{
    for(ConfigEntry& ce : entries){
        if(ce.key==key){
            return ce.value;
        }
    }

    update = true;
    entries.emplace_back(key,defaultvalue,description);
    return defaultvalue;
}



int ConfigLoader::getInt(const std::string &key , int defaultvalue, const std::string &description){
    std::string s = getLine2(key,std::to_string(defaultvalue),description);
    return atoi(s.c_str());
}


float ConfigLoader::getFloat(const std::string &key , float defaultvalue, const std::string &description){
    std::string s = getLine2(key,std::to_string(defaultvalue),description);
    return atof(s.c_str());
}

std::string ConfigLoader::getString(const std::string &key , const std::string &defaultvalue, const std::string &description){
    return getLine2(key,defaultvalue,description);
}

void ConfigLoader::setInt(const std::string &key, int value, const std::string &description, bool updateDescription)
{
    for(ConfigEntry& ce : entries){
        if(ce.key==key){
            ce.value = std::to_string(value);
            if (updateDescription){
                ce.description = description;
            }
            return;
        }
    }
}

}
