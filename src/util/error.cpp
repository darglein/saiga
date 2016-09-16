#include "saiga/util/error.h"



void Error::DebugLogWin32(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, const GLvoid *userParam){
    (void)userParam; (void)length;

    if (id == 131185){
        //Buffer detailed info
//        return;
    }

    auto typestr = getStringForType(type);
    if(typestr == "Other"){
        //intel mesa drivers prints random status messages as 'other'
//        return;
    }

    if(id == 131){
        //intel mesa message: Using a blit copy to avoid stalling on glBufferSubData
//        return;
    }

    std::cout<<"Type : "<< typestr <<
               " ; Source : "<<getStringForSource(source)<<
               "; ID : "<<id<<
               "; Severity : "<<getStringForSeverity(severity)<<std::endl;


    std::cout<< "Message : "<<message<<std::endl;
}

void Error::DebugLog(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, GLvoid *userParam){
    (void)length; //unused variables
    (void)userParam;

    DebugLogWin32(source,type,id,severity,length,message,userParam);

}

bool Error::checkGLError(){
    //don't call glGetError when OpenGL is not initialized
    if (!OpenGLisInitialized()){
        return false;
    }

    GLenum errCode;
    if ((errCode = glGetError()) != GL_NO_ERROR) {
        std::cout<<"OpenGL error: "<<errCode<<std::endl;
        return true;
    }
    return false;
}

std::string Error::getStringForSource(GLenum source) {

    switch(source) {
    case GL_DEBUG_SOURCE_API_ARB:
        return("API");
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM_ARB:
        return("Window System");
    case GL_DEBUG_SOURCE_SHADER_COMPILER_ARB:
        return("Shader Compiler");
    case GL_DEBUG_SOURCE_THIRD_PARTY_ARB:
        return("Third Party");
    case GL_DEBUG_SOURCE_APPLICATION_ARB:
        return("Application");
    case GL_DEBUG_SOURCE_OTHER_ARB:
        return("Other");
    default:
        return("");
    }
}

std::string Error::getStringForType(GLenum type) {

    switch(type) {
    case GL_DEBUG_TYPE_ERROR_ARB:
        return("Error");
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB:
        return("Deprecated Behaviour");
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB:
        return("Undefined Behaviour");
    case GL_DEBUG_TYPE_PORTABILITY_ARB:
        return("Portability Issue");
    case GL_DEBUG_TYPE_PERFORMANCE_ARB:
        return("Performance Issue");
    case GL_DEBUG_TYPE_OTHER_ARB:
        return("Other");
    default:
        return("");
    }
}

std::string Error::getStringForSeverity(GLenum severity) {

    switch(severity) {
    case GL_DEBUG_SEVERITY_HIGH_ARB:
        return("High");
    case GL_DEBUG_SEVERITY_MEDIUM_ARB:
        return("Medium");
    case GL_DEBUG_SEVERITY_LOW_ARB:
        return("Low");
    default:
        return("");
    }
}
