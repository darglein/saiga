#pragma once

#include "saiga/util/glm.h"
#include "saiga/opengl/opengl.h"
#include "saiga/opengl/texture/glImageFormatMap.h"


template<typename T, ImageElementFormat format>
struct SAIGA_GLOBAL TexelElementToFloatConversion{
};

template<>
struct SAIGA_GLOBAL TexelElementToFloatConversion<GLubyte,ImageElementFormat::UnsignedNormalized>{
    static float toFloat(GLubyte e){
        return e / 255.0f;
    }
};

template<>
struct SAIGA_GLOBAL TexelElementToFloatConversion<GLushort,ImageElementFormat::UnsignedNormalized>{
    static float toFloat(GLushort e){
        return e / 65535.0f;
    }
};

template<>
struct SAIGA_GLOBAL TexelElementToFloatConversion<GLuint,ImageElementFormat::UnsignedNormalized>{
    static float toFloat(GLuint e){
        //some precision is lost here...
        return e / 4294967295.0f;
    }
};

template<>
struct SAIGA_GLOBAL TexelElementToFloatConversion<GLfloat,ImageElementFormat::FloatingPoint>{
    static float toFloat(GLfloat e){
        return e;
    }
};


template<typename T, ImageElementFormat format>
struct SAIGA_GLOBAL TexelElementFromFloatConversion{
};

template<>
struct SAIGA_GLOBAL TexelElementFromFloatConversion<GLubyte,ImageElementFormat::UnsignedNormalized>{
    static GLubyte fromFloat(float f){
        return f * 255.0f;
    }
};

template<>
struct SAIGA_GLOBAL TexelElementFromFloatConversion<GLushort,ImageElementFormat::UnsignedNormalized>{
    static GLushort fromFloat(float f){
        return f * 65535.0f;
    }
};

template<>
struct SAIGA_GLOBAL TexelElementFromFloatConversion<GLuint,ImageElementFormat::UnsignedNormalized>{
    static GLuint fromFloat(float f){
        //some precision is lost here...
        return f * 4294967295.0f;
    }
};

template<>
struct SAIGA_GLOBAL TexelElementFromFloatConversion<GLfloat,ImageElementFormat::FloatingPoint>{
    static GLfloat fromFloat(float f){
        return f;
    }
};


template<int CHANNELS, int bitDepth, ImageElementFormat format>
struct SAIGA_GLOBAL Texel{
};



template<int bitDepth, ImageElementFormat format>
struct Texel<1,bitDepth,format>{
    typedef typename GLImageFormatMapElementType<bitDepth,format>::elementType elementType;
    typedef TexelElementToFloatConversion<elementType,format> toFloatConversion;
    typedef TexelElementFromFloatConversion<elementType,format> fromFloatConversion;
    const static GLenum type = GL_R;
    union{
        elementType r;
        elementType x;
    };

    vec4 toVec4(){
        return vec4( toFloatConversion::toFloat(r) , 0 , 0 , 0);
    }

    void fromVec4(vec4 t){
        r = fromFloatConversion::fromFloat(t.r);
    }
};

template<int bitDepth, ImageElementFormat format>
struct Texel<2,bitDepth,format>{
    typedef typename GLImageFormatMapElementType<bitDepth,format>::elementType elementType;
    typedef TexelElementToFloatConversion<elementType,format> toFloatConversion;
    typedef TexelElementFromFloatConversion<elementType,format> fromFloatConversion;
    const static GLenum type = GL_RG;
    union{
        elementType r;
        elementType x;
    };
    union{
        elementType g;
        elementType y;
    };

    vec4 toVec4(){
        return vec4( toFloatConversion::toFloat(r) , toFloatConversion::toFloat(g) , 0 , 0);
    }

    void fromVec4(vec4 t){
        r = fromFloatConversion::fromFloat(t.r);
        g = fromFloatConversion::fromFloat(t.g);
    }
};

template<int bitDepth, ImageElementFormat format>
struct Texel<3,bitDepth,format>{
    typedef typename GLImageFormatMapElementType<bitDepth,format>::elementType elementType;
    typedef TexelElementToFloatConversion<elementType,format> toFloatConversion;
    typedef TexelElementFromFloatConversion<elementType,format> fromFloatConversion;
    const static GLenum type = GL_RGB;
    union{
        elementType r;
        elementType x;
    };
    union{
        elementType g;
        elementType y;
    };
    union{
        elementType b;
        elementType z;
    };

    vec4 toVec4(){
        return vec4( toFloatConversion::toFloat(r) , toFloatConversion::toFloat(g) , toFloatConversion::toFloat(b) , 0);
    }

    void fromVec4(vec4 t){
        r = fromFloatConversion::fromFloat(t.r);
        g = fromFloatConversion::fromFloat(t.g);
        b = fromFloatConversion::fromFloat(t.b);
    }
};

template<int bitDepth, ImageElementFormat format>
struct Texel<4,bitDepth,format>{
    typedef typename GLImageFormatMapElementType<bitDepth,format>::elementType elementType;
    typedef TexelElementToFloatConversion<elementType,format> toFloatConversion;
    typedef TexelElementFromFloatConversion<elementType,format> fromFloatConversion;
    const static GLenum type = GL_RGBA;
    union{
        elementType r;
        elementType x;
    };
    union{
        elementType g;
        elementType y;
    };
    union{
        elementType b;
        elementType z;
    };
    union{
        elementType a;
        elementType w;
    };

    vec4 toVec4(){
        return vec4( toFloatConversion::toFloat(r) , toFloatConversion::toFloat(g) , toFloatConversion::toFloat(b) , toFloatConversion::toFloat(a));
    }

    void fromVec4(vec4 t){
        r = fromFloatConversion::fromFloat(t.r);
        g = fromFloatConversion::fromFloat(t.g);
        b = fromFloatConversion::fromFloat(t.b);
        a = fromFloatConversion::fromFloat(t.a);
    }
};
