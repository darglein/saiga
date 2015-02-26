#ifndef ERROR_H
#define ERROR_H

#include <iostream>
#include <cstdlib>

class Error{
public:

    static bool checkGLError(){
        GLenum errCode;
        const GLubyte *errString;

        if ((errCode = glGetError()) != GL_NO_ERROR) {
            errString = gluErrorString(errCode);
           std::cerr<< "OpenGL Error: "<<errString<<std::endl;
           return true;
        }
        return false;
    }

    static void quitWhenError(const char* func){
        if(Error::checkGLError()){
            std::cout<<"Error in "<<func<<"."<<std::endl;
            exit(1);
        }
    }


    // aux function to translate source to string
    static std::string getStringForSource(GLenum source) {

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

    // aux function to translate severity to string
    static std::string getStringForSeverity(GLenum severity) {

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

    // aux function to translate type to string
    static std::string getStringForType(GLenum type) {

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

    static void DebugLogWin32( GLenum source , GLenum type , GLuint id , GLenum severity ,
                   GLsizei length , const GLchar * message ,const GLvoid * userParam){
        std::cout<<"Type : "<<getStringForType(type)<<
               " ; Source : "<<getStringForSource(source)<<
               "; ID : "<<id<<
               "; Severity : "<<getStringForSeverity(severity)<<std::endl;


         std::cout<< "Message : "<<message<<std::endl;
    }

    static void DebugLog( GLenum source , GLenum type , GLuint id , GLenum severity ,
                   GLsizei length , const GLchar * message , GLvoid * userParam){
        (void)length; //unused variables
        (void)userParam;

        DebugLogWin32(source,type,id,severity,length,message,userParam);

    }
};

#endif // ERROR_H
