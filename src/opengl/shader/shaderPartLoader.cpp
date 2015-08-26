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
    //    std::cout<<"TODO: ShaderPartLoader"<<std::endl;
}


bool ShaderPartLoader::load()
{
    std::vector<std::string> data = loadAndPreproccess(file);
    if(data.size()<=0)
        return false;


    //        cout<<"ShaderPartLoader::load "<<data.size()<<endl;

    std::vector<std::string> code;
    int status = STATUS_WAITING;
    GLenum type = GL_INVALID_ENUM;
    int lineCount =0;

    for(std::string line : data){
        lineCount++;
        if(line.compare("##start")==0){
            status = (status==STATUS_WAITING)?STATUS_START:STATUS_ERROR;

        }else if(line.compare("##end")==0){
            status = (status==STATUS_READING)?STATUS_WAITING:STATUS_ERROR;

            if(status != STATUS_ERROR){
                //reading shader part sucessfull
                addShader(code,type);
                code.clear();
            }

        }else if(line.compare("##vertex")==0){
            status = (status==STATUS_START)?STATUS_READING:STATUS_ERROR;
            type = GL_VERTEX_SHADER;

        }else if(line.compare("##fragment")==0){
            status = (status==STATUS_START)?STATUS_READING:STATUS_ERROR;
            type = GL_FRAGMENT_SHADER;

        }else if(line.compare("##geometry")==0){
            status = (status==STATUS_START)?STATUS_READING:STATUS_ERROR;
            type = GL_GEOMETRY_SHADER;
        }else if(status == STATUS_READING){
            //normal code line
            code.push_back(line+'\n');
        }



        if(status == STATUS_ERROR){
            //                std::cerr<<"Shader-Loader: Error "<<errorMsg<<" in line "<<lineCount<<"\n";
            return false;
        }
    }


    return true;
}


std::vector<std::string> ShaderPartLoader::loadAndPreproccess(const std::string &file)
{
    std::vector<std::string> ret;

    std::ifstream fileStream(prefix+"/"+file, std::ios::in);
    if(!fileStream.is_open()) {
        return ret;
    }

    const std::string include("#include ");

    while(!fileStream.eof()) {
        std::string line;
        std::getline(fileStream, line);

        if(include.size()<line.size() && line.compare(0, include.length(), include)==0){
            line = line.substr(include.size()-1);

            auto it = std::remove(line.begin(),line.end(),'"');
            line.erase(it,line.end());

            it = std::remove(line.begin(),line.end(),' ');
            line.erase(it,line.end());

            //recursivly load includes
            std::vector<std::string> tmp = loadAndPreproccess(line);
            ret.insert(ret.end(),tmp.begin(),tmp.end());
        }else{
            ret.push_back(line);
        }


    }
    return ret;
}

void ShaderPartLoader::addShader(std::vector<std::string> &content, GLenum type)
{
    //    cout<<"ShaderPartLoader::addShader "<<content.size()<<" "<<type<<endl;

    auto shader = std::make_shared<ShaderPart>();
    //    ShaderPart shader;
    shader->code = content;
    shader->type = type;
    shader->addInjections(injections);

    shader->createGLShader();
    if(shader->compile()){
        shaders.push_back(shader);
    }

    Error::quitWhenError("ShaderPartLoader::addShader");
}




