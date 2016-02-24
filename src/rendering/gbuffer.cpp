#include "saiga/rendering/gbuffer.h"
#include "saiga/opengl/vertexBuffer.h"

GBuffer::GBuffer()
{

}

GBuffer::GBuffer(int w, int h, GBufferParameters params)
{
    init(w,h,params);
}

void GBuffer::init(int w, int h, GBufferParameters params)
{
    this->params = params;
    this->create();
    Texture* color = new Texture();
    if(params.srgb){
        color->createEmptyTexture(w,h,GL_RGB,GL_SRGB8,GL_UNSIGNED_BYTE);
    }else{
        switch(params.colorQuality){
        case Quality::LOW:
            color->createEmptyTexture(w,h,GL_RGB,GL_RGB8,GL_UNSIGNED_BYTE);
            break;
        case Quality::MEDIUM:
            color->createEmptyTexture(w,h,GL_RGB,GL_RGB16,GL_UNSIGNED_SHORT);
            break;
        case Quality::HIGH:
            color->createEmptyTexture(w,h,GL_RGB,GL_RGB16,GL_UNSIGNED_SHORT);
            break;
        }
    }
    attachTexture(color);


    Texture* normal = new Texture();
    switch(params.normalQuality){
    case Quality::LOW:
        normal->createEmptyTexture(w,h,GL_RG,GL_RG8,GL_UNSIGNED_BYTE);
        break;
    case Quality::MEDIUM:
        normal->createEmptyTexture(w,h,GL_RG,GL_RG16 ,GL_UNSIGNED_SHORT);
        break;
    case Quality::HIGH:
        normal->createEmptyTexture(w,h,GL_RG,GL_RG16 ,GL_UNSIGNED_SHORT);
        break;
    }
    attachTexture(normal);

    //specular and emissive texture
    Texture* data = new Texture();
    switch(params.dataQuality){
    case Quality::LOW:
        data->createEmptyTexture(w,h,GL_RGBA,GL_RGBA8,GL_UNSIGNED_BYTE);
        break;
    case Quality::MEDIUM:
        data->createEmptyTexture(w,h,GL_RGBA,GL_RGBA16,GL_UNSIGNED_SHORT);
        break;
    case Quality::HIGH:
        data->createEmptyTexture(w,h,GL_RGBA,GL_RGBA16,GL_UNSIGNED_SHORT);
        break;
    }
    attachTexture(data);


    //    Texture* position = new Texture();
    //    position->createEmptyTexture(w,h,GL_RGB,GL_RGB32F ,GL_FLOAT);
    //    attachTexture(position);

    Texture* depth = new Texture();
    switch(params.depthQuality){
    case Quality::LOW:
        depth->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT16,GL_UNSIGNED_SHORT);
        break;
    case Quality::MEDIUM:
        depth->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT16,GL_UNSIGNED_SHORT);
        break;
    case Quality::HIGH:
        depth->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32,GL_UNSIGNED_INT);
        break;
    }
    attachTextureDepth(depth);

    //don't need stencil in gbuffer (but blit would fail otherwise)
    //depth and stencil texture combined
    //    Texture* depth_stencil = new Texture();
    //    depth_stencil->createEmptyTexture(w,h,GL_DEPTH_STENCIL, GL_DEPTH24_STENCIL8,GL_UNSIGNED_INT_24_8);
    //    attachTextureDepthStencil(depth_stencil);


    int count = colorBuffers.size();

    std::vector<GLenum> DrawBuffers(count);
    for(int i = 0 ;i < count ; ++i){
        DrawBuffers[i] = GL_COLOR_ATTACHMENT0 + i;
    }
    glDrawBuffers(count, &DrawBuffers[0]);


    check();
    unbind();
}

void GBuffer::sampleNearest()
{
    depthBuffer->setFiltering(GL_NEAREST);
    for(Texture* t : colorBuffers){
        t->setFiltering(GL_NEAREST);
    }
}

void GBuffer::sampleLinear()
{
    depthBuffer->setFiltering(GL_LINEAR);
    for(Texture* t : colorBuffers){
        t->setFiltering(GL_LINEAR);
    }
}
