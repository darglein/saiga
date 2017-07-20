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


bool ConfigLoader::loadFile(const std::string &name)
{
    this->name = name;

//    stream.exceptions ( std::fstream::failbit | std::fstream::badbit );


    try {
        stream.open (name.c_str(), std::fstream::in | std::fstream::out | std::fstream::app);
    }
    catch (const std::fstream::failure &e) {
        std::cout<< e.what() <<std::endl;
        std::cout << "Exception opening/reading file\n";
        return false;
    }

    state = State::LOADED;
    return true;
}

bool ConfigLoader::loadFile2(const std::string &name)
{
    entries.clear();
    this->name = name;


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
            std::cout<<"Key "<<key<<"  Value='"<<value<<"'"<<" Comment: "<<comment<<std::endl;

        }
    }

    stream.close();
    state = State::LOADED;
    return true;
}

bool ConfigLoader::writeFile()
{

    try {
        stream.open (name,  std::fstream::out);
    }
    catch (const std::fstream::failure &e) {
        std::cout<< e.what() <<std::endl;
        std::cout << "Exception opening/reading file\n";
        return false;
    }

    for(ConfigEntry& ce : entries){
        stream<<ce.toString()<<std::endl;
    }

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

std::string ConfigLoader::getLine(const std::string &key, const std::string &defaultvalue, const std::string &description){
    if(state!=State::LOADED)
        return "";

    stream.seekg (0, stream.beg); //set pointer to beginning of file

    //read file line by line
    std::string line;
    while (std::getline(stream, line))
    {
        if(line.find(key) == 0){
            auto pos = line.find('=');
            std::string value = line.substr(pos+1,line.length()-pos-1);
            parseValue(value);
            std::cout<<"Key "<<key<<" found. Value='"<<value<<"'"<<std::endl;
            return value;
        }
    }


    std::cout << "Key '"<<key<<"' not found. Adding default value '"<<defaultvalue<<"'."<<std::endl;
    stream.clear();
    stream << key << "=" << defaultvalue ;


    if (description != ""){
        int maxAlign = 25;
        int fillSize = maxAlign-(key.size() + defaultvalue.size());
        if(fillSize<=0) fillSize = 1;

        std::string fill = std::string(fillSize, ' ');
        stream  <<  fill << "# " << description;
    }

    stream<<std::endl;

    stream.flush();
    return defaultvalue;

}

std::string ConfigLoader::getLine2(const std::string &key, const std::string &defaultvalue, const std::string &description)
{
    for(ConfigEntry& ce : entries){
        if(ce.key==key){
            return ce.value;
        }
    }

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
