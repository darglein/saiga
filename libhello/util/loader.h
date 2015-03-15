#pragma once

#include <string>
#include <iostream>

#include <vector>
template <typename T>
class Loader{
protected:
    std::vector<std::string> locations;
    std::vector<T*> objects;
public:
    Loader(){};
    ~Loader();
    void addPath(const std::string &path){locations.push_back(path);}
    virtual T* load(const std::string &name);

    virtual T* loadFromFile(const std::string &name) = 0;
};

template<class T>
Loader<T>:: ~Loader(){
    for(T* &object : objects){
        delete object;
    }
}

template<class T>
T* Loader<T>::load(const std::string &name){
    //check if already exists
    for(T* &object : objects){
        if(object->name == name)
            return object;
    }

    T* object;

    for(std::string &path : locations){
        std::string complete_path = path + "/" + name;
        object = loadFromFile(complete_path);
        if (object){
            object->name = name;
            std::cout<<"Loaded from file: "<<complete_path<<std::endl;
            objects.push_back(object);
            return object;
        }
    }
    std::cout<<"Failed to load "<<name<<"!!!"<<std::endl;


    return NULL;
}


