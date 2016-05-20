#include "saiga/framework.h"

#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/objloader.h"
#include "saiga/opengl/texture/textureLoader.h"

#include "saiga/util/configloader.h"
#include "saiga/util/assert.h"

bool initialized = false;

std::string SHADER_PATH;
std::string TEXTURE_PATH;
std::string MATERIAL_PATH;
std::string OBJ_PATH;

void readConfigFile(){
    ConfigLoader cl;
    cl.loadFile2("saiga-config.txt");

    SHADER_PATH = cl.getString("SHADER_PATH","/usr/local/share/saiga/shader");
    TEXTURE_PATH = cl.getString("TEXTURE_PATH","textures");
    MATERIAL_PATH = cl.getString("MATERIAL_PATH","objs");
    OBJ_PATH = cl.getString("OBJ_PATH","objs");

    cl.writeFile();

}

void writeExtensions(){

    std::ofstream myfile;
    myfile.open ("opengl-extensions.txt");



    int n = getExtensionCount();
    for (GLint i=0; i<n; i++)
    {
        const char* extension = (const char*) glGetStringi(GL_EXTENSIONS, i);

        myfile << extension<<endl;
    }

    myfile.close();
}

void initSaiga()
{
    assert(!initialized);

    //    writeExtensions();
    readConfigFile();

    ShaderLoader::instance()->addPath(SHADER_PATH);
    ShaderLoader::instance()->addPath(SHADER_PATH+"/geometry");
    ShaderLoader::instance()->addPath(SHADER_PATH+"/lighting");
    ShaderLoader::instance()->addPath(SHADER_PATH+"/post_processing");

    shaderPathes.addSearchPath(SHADER_PATH);
    shaderPathes.addSearchPath(SHADER_PATH+"/geometry");
    shaderPathes.addSearchPath(SHADER_PATH+"/lighting");
    shaderPathes.addSearchPath(SHADER_PATH+"/post_processing");

    TextureLoader::instance()->addPath(TEXTURE_PATH);
    TextureLoader::instance()->addPath(OBJ_PATH);
    TextureLoader::instance()->addPath(".");

    MaterialLoader::instance()->addPath(".");
    MaterialLoader::instance()->addPath(OBJ_PATH);

    ObjLoader::instance()->addPath(".");
    ObjLoader::instance()->addPath(OBJ_PATH);




    cout<<"========================== Framework initialization done! =========================="<<endl;
    initialized = true;

}

void cleanupSaiga()
{
    assert(initialized);

    ShaderLoader::instance()->clear();
    TextureLoader::instance()->clear();
    MaterialLoader::instance()->clear();
    ObjLoader::instance()->clear();
    cout<<"========================== Framework cleanup done! =========================="<<endl;
    initialized = false;
}

