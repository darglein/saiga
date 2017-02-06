#pragma once

#include <saiga/config.h>
#include <string>
#include <iostream>

#include <vector>
#include <tuple>
#include "assert.h"

struct SAIGA_GLOBAL NoParams{
//    bool operator==(const NoParams& other){
//        return true;
//    }
};

SAIGA_GLOBAL inline bool operator==(const NoParams& lhs, const NoParams& rhs) {
	(void)lhs; (void)rhs;
    return true;
}


template <typename object_t, typename param_t = NoParams>
class Loader{
protected:
    std::vector<std::string> locations; //locations where to search


    //string: name passed in load(), param: params passed in load()
    typedef std::tuple<std::string, param_t, object_t> data_t;


    std::vector<data_t> objects;
public:
//    Loader(){};
    virtual ~Loader();
    void clear();
    void addPath(const std::string &path){locations.push_back(path);}



    virtual object_t load(const std::string &name, const param_t &params=param_t());
    virtual object_t getLoaded(const std::string &name, const param_t &params=param_t());
    void put(const std::string &name, object_t* obj, const param_t &params=param_t());

protected:
    virtual object_t exists(const std::string &name, const param_t &params=param_t());
    virtual object_t loadFromFile(const std::string &name, const param_t &params) = 0;
};

template<typename object_t, typename param_t >
Loader<object_t,param_t>:: ~Loader(){
    clear();
}


template<typename object_t, typename param_t >
void Loader<object_t,param_t>::clear(){
    for(data_t &object : objects){
        delete std::get<2>(object);
    }
    objects.clear();
}

template<typename object_t, typename param_t >
object_t Loader<object_t,param_t>::exists(const std::string &name, const param_t &params){
    //check if already exists
    for(data_t &data : objects){
        if(std::get<0>(data)==name && std::get<1>(data)==params){
            return std::get<2>(data);
        }
    }
    return nullptr;
}


template<typename object_t, typename param_t >
object_t Loader<object_t,param_t>::load(const std::string &name, const param_t &params){

    object_t object = exists(name,params);
    if(object){
        std::cerr << name << " already loaded!! " << std::endl;
        return object;
    }


    for(std::string &path : locations){
        std::string complete_path = path + "/" + name;
        object = loadFromFile(complete_path,params);
        if (object){
            std::cout<<"Loaded from file: "<<complete_path<<std::endl;
            objects.emplace_back(name,params,object);
            return object;
        }
    }
    std::cout<<"Failed to load "<<name<<"!!!"<<std::endl;


    return NULL;
}



template<typename object_t, typename param_t >
object_t Loader<object_t,param_t>::getLoaded(const std::string &name, const param_t &params){

    object_t object = exists(name,params);
    if(object){
        return object;
    }

    std::cout<< name << " not loaded !!!" <<std::endl;
    SAIGA_ASSERT(false);

    return NULL;
}

template<typename object_t, typename param_t >
void Loader<object_t,param_t>::put(const std::string &name, object_t* obj, const param_t &params){

    SAIGA_ASSERT(!exists(name,params) && "object was already loaded!");
    SAIGA_ASSERT(obj);

    objects.emplace_back(name,params,obj);
}
