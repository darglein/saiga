#include "saiga/opengl/shader/shaderPartLoader.h"
#include "saiga/opengl/shader/shader.h"
#include "saiga/util/error.h"
#include <fstream>
#include <algorithm>

#define STATUS_WAITING 0
#define STATUS_READING 1

ShaderPartLoader::ShaderPartLoader() : ShaderPartLoader("","",ShaderCodeInjections()){
}

ShaderPartLoader::~ShaderPartLoader(){
}



ShaderPartLoader::ShaderPartLoader(const std::string &file, const std::string &prefix, const ShaderCodeInjections &injections)
    : file(file),prefix(prefix),injections(injections){
}


bool ShaderPartLoader::load()
{

    std::vector<std::string> data;

    if(!loadAndPreproccess(file,data))
        return false;


    std::vector<std::string> code;
    int status = STATUS_WAITING;
    GLenum type = GL_INVALID_ENUM;
    int lineCount =0;

    for(std::string line : data){
        lineCount++;

        bool readLine = true;
        for(int i = 0 ; i < ShaderPart::shaderTypeCount ; ++i){
            std::string key("##"+ShaderPart::shaderTypeStrings[i]);
            //this only compares the first characteres of line, so that for example addittional '\r's are ignored.
            if(line.compare(0,key.size(),key)==0){
                if(status==STATUS_READING){
                    addShader(code,type);
                    code.clear();
                }
                status = STATUS_READING;
                type = ShaderPart::shaderTypes[i];
                readLine = false;
                break;

            }
        }

        if(status == STATUS_READING && readLine){
            code.push_back(line+'\n');
        }

    }

    if(status==STATUS_READING){
        addShader(code,type);
        code.clear();
    }


    return true;
}


bool ShaderPartLoader::loadAndPreproccess(const std::string &file, std::vector<std::string> &ret)
{

    std::ifstream fileStream(prefix+"/"+file, std::ios::in);
    if(!fileStream.is_open()) {
        return false;
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
            std::vector<std::string> tmp;
            if(!loadAndPreproccess(line,tmp)){
                std::cerr<<"ShaderPartLoader: Could not open included file: "<<prefix+"/"+line<<endl;
                std::cerr<<"Make sure it exists and the path is correct"<<endl;
                assert(0);
            }
            ret.insert(ret.end(),tmp.begin(),tmp.end());
        }else{
            ret.push_back(line);
        }


    }
    return true;
}

void ShaderPartLoader::addShader(std::vector<std::string> &content, GLenum type)
{
    auto shader = std::make_shared<ShaderPart>();
    shader->code = content;
    shader->type = type;
    shader->addInjections(injections);

    shader->createGLShader();
    if(shader->compile()){
        shaders.push_back(shader);
    }

    assert_no_glerror();
}

void ShaderPartLoader::reloadShader(Shader *shader)
{
//    cout<<"ShaderPartLoader::reloadShader"<<endl;
    shader->destroyProgram();

    shader->shaders = shaders;
    shader->createProgram();

    std::cout<<"Loaded: "<<prefix + "/" + file<<" ( ";
    for(auto& sp : shaders){
        std::cout<<sp->type<<" ";
    }
    std::cout<<")"<<std::endl;

    assert_no_glerror();
}




