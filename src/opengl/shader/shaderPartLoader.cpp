#include "saiga/opengl/shader/shaderPartLoader.h"

#include "saiga/util/error.h"
#include <fstream>
#include <algorithm>

#define STATUS_WAITING 0 //waiting for "start"
#define STATUS_START 1 //found start + looking for type
#define STATUS_READING 2 //found start + found type
#define STATUS_ERROR 3

ShaderPartLoader::ShaderPartLoader() : ShaderPartLoader("","",ShaderCodeInjections()){
}

ShaderPartLoader::~ShaderPartLoader(){


}

ShaderPartLoader::ShaderPartLoader(const std::string &file, const std::string &prefix, const ShaderCodeInjections &injections)
    : file(file),prefix(prefix),injections(injections){
    std::cout<<"TODO: ShaderPartLoader"<<std::endl;
}






