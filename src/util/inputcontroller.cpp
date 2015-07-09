#include "saiga/util/inputcontroller.h"

template<>
int InputController::Operation::Arguments::next(){
    return next(std::stoi);
}

template<>
unsigned int InputController::Operation::Arguments::next(){
    return next(std::stoul);
}

template<>
float InputController::Operation::Arguments::next(){
    return next(std::stof);
}

template<>
vec3 InputController::Operation::Arguments::next(){
    vec3 ret;
    ret.x = next(std::stof);
    ret.y = next(std::stof);
    ret.z = next(std::stof);
    return ret;
}

template<>
vec4 InputController::Operation::Arguments::next(){
    vec4 ret(next<vec3>(),0);
    ret.w = next(std::stof);
    return ret;
}



InputController::InputController() : stream(&std::cout){
}

void InputController::add(const std::string& key,op_t op, int argc, const std::string& description){
    add(key,Operation(argc,op,description));
}

void InputController::add(const std::string& key, const Operation &value){
    auto it = functionMap.find(key);
    if(it!=functionMap.end()){
        *stream<<"InputController: key already exists '"<<key<<"'"<<std::endl;
        return;
    }
    functionMap.insert(mapElement(key,value));
}

bool InputController::execute(const std::string& line){
    size_t pos = line.find(' ');
    Operation::Arguments arg;
    arg.os = stream;
    if(pos==std::string::npos){
        return execute(line,arg);
    }
    std::string key = line.substr(0,pos);



    arg.args = line.substr(pos+1);
    return execute(key,arg);
}

bool InputController::execute(const std::string& key, Operation::Arguments &args){
    auto it = functionMap.find(key);
    if(it==functionMap.end()){
        *stream<<"InputController: Unknown command '"<<key<<"'"<<std::endl;
        return false;
    }
    *stream<<"InputController: Executing command '"<<key<<"'  '"<<args.args<<"'..."<<std::endl;

    Operation &op = it->second;


    op.op(args);
    return true;

}

void InputController::functionList(std::vector<std::string> &out){
    for(auto iter=functionMap.begin(); iter!=functionMap.end(); ++iter){
        out.push_back(iter->first);
    }
    std::sort(out.begin(), out.end());

}
