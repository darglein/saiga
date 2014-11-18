#include "opengl/shader.h"

#define STATUS_WAITING 0 //waiting for "start"
#define STATUS_START 1 //found start + looking for type
#define STATUS_READING 2 //found start + found type
#define STATUS_ERROR 3

Shader::Shader() : Shader(""){
}

Shader::~Shader(){
    if(program!=0){
        glDeleteProgram(program);
        program = 0;
    }
    if(vertShader){
        glDeleteShader(vertShader);
        vertShader = 0;
    }
    if(geoShader){
        glDeleteShader(geoShader);
        geoShader = 0;
    }
    if(fragShader){
        glDeleteShader(fragShader);
        fragShader = 0;
    }
}

Shader::Shader(const string &multi_file) : shaderPath(multi_file),program(0),vertShader(0),geoShader(0),fragShader(0){

}


bool Shader::reload(){
    if(shaderPath.length()<=0){
        cerr<<"Reload only works if the Shader object is created with a multishader in the constructor"<<endl;
        return false;
    }
    if(program!=0){
        glDeleteProgram(program);
        program = 0;
    }

    return addMultiShaderFromFile(shaderPath);


}

bool Shader::addMultiShaderFromFile(const string &multi_file) {
//    cout<<"Shader-Loader: Reading file "<<filePath<<"\n";
    std::string content;
    std::ifstream fileStream(multi_file, std::ios::in);

    if(!fileStream.is_open()) {
//        std::cerr << "Could not read file " << filePath << ". File does not exist." << std::endl;
        return false;
    }

    int status = STATUS_WAITING;
    int type = -1;
    std::string line = "";
    int lineCount =0;
    string errorMsg = "";
    while(!fileStream.eof()) {
        std::getline(fileStream, line);
        lineCount++;
        if(line.compare("##start")==0){
            status = (status==STATUS_WAITING)?STATUS_START:STATUS_ERROR;

        }else if(line.compare("##end")==0){
            status = (status==STATUS_READING)?STATUS_WAITING:STATUS_ERROR;

            if(status != STATUS_ERROR){
                //reading shader part sucessfull
                addShader(content.c_str(),type);
                content = "";
                for(int i=0;i<lineCount-1;i++)
                    content.append("\n");
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
        }



        if(status == STATUS_ERROR){
            std::cerr<<"Shader-Loader: Error "<<errorMsg<<" in line "<<lineCount<<"\n";
            fileStream.close();
            return false;
        }else if(status == STATUS_READING){
            content.append(line);
        }
        content.append("\n");
    }

    fileStream.close();
    createProgram();
    return true;
}



GLuint Shader::createProgram(){

    program = glCreateProgram();

    if(vertShader)
        glAttachShader(program, vertShader);
    if(geoShader){
        glAttachShader(program, geoShader);
        glProgramParameteriEXT(program,GL_GEOMETRY_INPUT_TYPE_EXT,GL_TRIANGLES);
        glProgramParameteriEXT(program,GL_GEOMETRY_OUTPUT_TYPE_EXT,GL_TRIANGLE_STRIP);

    }
    if(fragShader)
        glAttachShader(program, fragShader);

    glLinkProgram(program);

    printProgramLog(program);


    if(vertShader){
        glDeleteShader(vertShader);
        vertShader = 0;
    }
    if(geoShader){
        glDeleteShader(geoShader);
        geoShader = 0;
    }
    if(fragShader){
        glDeleteShader(fragShader);
        fragShader = 0;
    }

    checkUniforms();
    return program;
}

GLuint Shader::addShader(const char* content, int type){
    GLuint id = glCreateShader(type);


    GLint result = GL_FALSE;
    // Compile vertex shader
    glShaderSource(id, 1, &content, NULL);
    glCompileShader(id);
    // Check vertex shader
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);

    printShaderLog(id);


    if(!result){
        glDeleteShader(id);
        id = 0;
    }

    switch(type){
    case GL_VERTEX_SHADER:
        vertShader = id;
        break;
    case GL_GEOMETRY_SHADER:
        geoShader = id;
        break;
    case GL_FRAGMENT_SHADER:
        fragShader = id;
        break;

    }

    return id;
}

GLuint Shader::addShaderFromFile(const char* file, int type){
    cout<<"Shader-Loader: Reading file "<<file<<"\n";
    std::string content;
    std::ifstream fileStream(file, std::ios::in);

    if(!fileStream.is_open()) {
        std::cerr << "Could not read file " << file << ". File does not exist." << std::endl;
        return 0;
    }

    std::string line = "";

    while(!fileStream.eof()) {
        std::getline(fileStream, line);
        content.append(line);
        content.append("\n");
    }

    fileStream.close();

    return addShader(content.c_str(),type);
}

string Shader::typeToName(int type){
    switch(type){
    case GL_VERTEX_SHADER:
        return "Vertex Shader";
    case GL_GEOMETRY_SHADER:
        return "Geometry Shader";
    case GL_FRAGMENT_SHADER:
        return "Fragment Shader";
    default:
        return "Unkown Shader type!";
    }
}

void Shader::bind(){
    if(program==0){
        std::cerr<<"bind: no shader loaded!\n";
        exit(1);
        return;
    }
    glUseProgram(program);
}

void Shader::unbind(){
    glUseProgram(0);
}

GLint Shader::getUniformLocation(const char* name){
    int i = glGetUniformLocation(program,name);
    if(i==-1){
//        cout<<"Cannot find uniform: "<<name<<endl;
    }
    return i;
}


void Shader::printProgramLog( GLuint program ){
    //Make sure name is shader
    if( glIsProgram( program ) )
    {
        //Program log length
        int infoLogLength = 0;
        int maxLength = infoLogLength;

        //Get info string length
        glGetProgramiv( program, GL_INFO_LOG_LENGTH, &maxLength );

        //Allocate string
        char* infoLog = new char[ maxLength ];

        //Get info log
        glGetProgramInfoLog( program, maxLength, &infoLogLength, infoLog );
        if( infoLogLength > 0 )
        {
            //Print Log
            printf( "%s\n", infoLog );
        }

        //Deallocate string
        delete[] infoLog;
    }
    else
    {
        printf( "Name %d is not a program\n", program );
    }
}

void Shader::printShaderLog( GLuint shader ){
    //Make sure name is shader
    if( glIsShader( shader ) )
    {
        //Shader log length
        int infoLogLength = 0;
        int maxLength = infoLogLength;

        //Get info string length
        glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &maxLength );

        //Allocate string
        char* infoLog = new char[ maxLength ];

        //Get info log
        glGetShaderInfoLog( shader, maxLength, &infoLogLength, infoLog );
        if( infoLogLength > 0 )
        {
            //Print Log
            printf( "%s\n", infoLog );
        }

        //Deallocate string
        delete[] infoLog;
    }
    else
    {
        printf( "Name %d is not a shader\n", shader );
    }
}

void Shader::upload(int location, const mat4 &m){
    glUniformMatrix4fv(location,1,GL_FALSE, (GLfloat*)&m[0]);
}

void Shader::upload(int location, const vec4 &v){
    glUniform4fv(location,1,(GLfloat*)&v[0]);
}

void Shader::upload(int location, const vec3 &v){
    glUniform3fv(location,1,(GLfloat*)&v[0]);
}

void Shader::upload(int location, const vec2 &v){
    glUniform2fv(location,1,(GLfloat*)&v[0]);
}

void Shader::upload(int location, const int &i){
    glUniform1i(location,(GLint)i);
}

void Shader::upload(int location, const float &f){
    glUniform1f(location,(GLfloat)f);
}


void Shader::upload(int location, int count, mat4* m){
    glUniformMatrix4fv(location,count,GL_FALSE,(GLfloat*)m);
}

void Shader::upload(int location, int count, vec4* v){
    glUniform4fv(location,count,(GLfloat*)v);
}

void Shader::upload(int location, int count, vec3* v){
    glUniform3fv(location,count,(GLfloat*)v);
}

void Shader::upload(int location, int count, vec2* v){
    glUniform2fv(location,count,(GLfloat*)v);
}


Shader* ShaderLoader::loadFromFile(const std::string &name){
    Shader* shader = new Shader(name);
    if(shader->reload()){
        return shader;
    }
    delete shader;
    return NULL;
}

void ShaderLoader::reload(){
    for(Shader* &object : objects){
        object->reload();
    }
}
