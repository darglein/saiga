#pragma once

#include <saiga/config.h>
#include <map>
#include <string>
#include <functional>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdarg>
#include "saiga/util/glm.h"

#define IC InputController::getInstance()
#define ICPARAMS InputController::Operation::Arguments &args

#define ICPARSEVEC3 vec3(args.next(std::stof),args.next(std::stof),args.next(std::stof) )
#define ICPARSEVEC4 vec3(args.next(std::stof),args.next(std::stof),args.next(std::stof),args.next(std::stof) )


class InputController
{
public:


    struct Operation{
        struct Arguments{
            std::ostream* os;
            Arguments():os(nullptr),valid(true){}
            Arguments(std::ostream *os,std::string args):os(os),valid(true),args(args){}
            bool isValid(){return valid;}
            template<typename T>
            T next();

            bool valid;
            std::string args;
        private:
            template<typename T>
            T next(T (*arg)(const std::string&,std::size_t*,int));
            template<typename T>
            T next(T (*arg)(const std::string&,std::size_t*));

        };
        typedef std::function<void(ICPARAMS)> op_t;
        int argc;
        op_t op;
        std::string description;
        Operation(const int &argc,const op_t &op,const std::string &description):argc(argc),op(op),description(description){}
    };
    typedef Operation::op_t op_t;
    typedef std::pair<std::string,Operation> mapElement;

    static InputController& getInstance(){
        static InputController    instance;
        return instance;
    }

	SAIGA_GLOBAL void add(const std::string& key, op_t op, int argc = 0, const std::string &description = "");
	SAIGA_GLOBAL void add(const std::string& key, const Operation &value);


	SAIGA_GLOBAL bool execute(const std::string& line);
	SAIGA_GLOBAL bool execute(const std::string& key, Operation::Arguments &args);
	SAIGA_GLOBAL void setOstream(std::ostream* stream){ this->stream = stream; }

	SAIGA_GLOBAL void functionList(std::vector<std::string> &out);

private:
    std::ostream* stream;
	SAIGA_GLOBAL InputController();
	SAIGA_GLOBAL InputController(InputController const&);
	SAIGA_GLOBAL void operator=(InputController const&);


    std::map<std::string,Operation> functionMap;
};

//the unspecialized next() is undefined!
template<>
SAIGA_GLOBAL int InputController::Operation::Arguments::next<int>();
template<>
SAIGA_GLOBAL unsigned int InputController::Operation::Arguments::next<unsigned int>();
template<>
SAIGA_GLOBAL float InputController::Operation::Arguments::next<float>();
template<>
SAIGA_GLOBAL vec3 InputController::Operation::Arguments::next<vec3>();
template<>
SAIGA_GLOBAL vec4 InputController::Operation::Arguments::next<vec4>();

template<typename T>
T InputController::Operation::Arguments::next(T (*arg)(const std::string&,std::size_t*,int)){
    if(!valid){
        return 0;
    }
    size_t idx;
    T i = 0;
    try{
        i = arg(args, &idx,10);
        args = args.substr(idx);
    }catch(const std::exception& e){
        *os<<"Invalid Arguments!"<<std::endl;
        valid = false;
    }
    return i;
}

template<typename T>
T InputController::Operation::Arguments::next(T (*arg)(const std::string&,std::size_t*)){
    if(!valid){
        return 0;
    }
    size_t idx;
    T i = 0;
    try{
        i = arg(args, &idx);
        args = args.substr(idx);
	}
	catch (const std::exception& e){
        *os<<"Invalid Arguments!"<<std::endl;
        valid = false;
    }
    return i;
}
