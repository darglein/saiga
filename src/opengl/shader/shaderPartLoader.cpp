#include "saiga/opengl/shader/shaderPartLoader.h"

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
    //    std::cout<<"TODO: ShaderPartLoader"<<std::endl;
}


bool ShaderPartLoader::load()
{



    std::vector<std::string> data = loadAndPreproccess(file);
    if(data.size()<=0)
        return false;


    std::vector<std::string> code;
    int status = STATUS_WAITING;
    GLenum type = GL_INVALID_ENUM;
    int lineCount =0;

    for(std::string line : data){
        lineCount++;

        bool readLine = true;
        for(int i = 0 ; i < ShaderPart::shaderTypeCount ; ++i){
            if(line.compare("##"+ShaderPart::shaderTypeStrings[i])==0){
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




