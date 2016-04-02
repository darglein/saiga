#include "saiga/opengl/framebuffer.h"
#include "saiga/util/error.h"

Framebuffer::Framebuffer(){

}

Framebuffer::~Framebuffer()
{



    destroy();
    if(depthBuffer == stencilBuffer){
        delete depthBuffer;
    }else{

        delete depthBuffer;
        delete stencilBuffer;
    }
    for(raw_Texture* t : colorBuffers){
        delete t;
    }
}

void Framebuffer::create(){
    if(id){
        std::cerr<<"Warning Framebuffer already created!"<<std::endl;
    }
    glGenFramebuffers(1, &id);
    bind();
    //    glFramebufferParameteri(GL_DRAW_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_WIDTH, 1000);
    //    glFramebufferParameteri(GL_DRAW_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_HEIGHT, 1000);

}

void Framebuffer::destroy(){
    if(!id)
        return;
    glDeleteFramebuffers(1,&id);
    id = 0;
}



void Framebuffer::bind(){
    glBindFramebuffer(GL_FRAMEBUFFER, id);
}

void Framebuffer::unbind(){
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Framebuffer::check(){
    glBindFramebuffer(GL_FRAMEBUFFER, id);
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE){
        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        switch(status)
        {
        case GL_FRAMEBUFFER_COMPLETE:                       std::cerr << ("GL_FRAMEBUFFER_COMPLETE\n") << std::endl;                        break;
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:         std::cerr <<("GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER\n") << std::endl;          break;
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:          std::cerr <<("GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT\n") << std::endl;           break;
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:  std::cerr <<("GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT\n") << std::endl;   break;
        case GL_FRAMEBUFFER_UNSUPPORTED:                    std::cerr <<("GL_FRAMEBUFFER_UNSUPPORTED\n")<< std::endl ;                     break;
        default:                                            std::cerr <<"Unknown issue " << status << std::endl;                     break;
        }

        std::cerr << "Framebuffer error!" << std::endl;
        assert(0);
    }
}

void Framebuffer::attachTexture(raw_Texture* texture, GLenum textTarget){
    bind();
    int index = colorBuffers.size();
    colorBuffers.push_back(texture);
    GLenum cid = GL_COLOR_ATTACHMENT0+index;
    //    glFramebufferTexture(GL_FRAMEBUFFER, cid,texture->id, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, cid,texture->getTarget(),texture->getId(), 0);

}

void Framebuffer::attachTextureDepth(raw_Texture* texture, GLenum textTarget){
    bind();
    depthBuffer = texture;
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,  GL_DEPTH_ATTACHMENT, texture->getTarget(),texture->getId(), 0);
}

void Framebuffer::attachTextureStencil(raw_Texture* texture, GLenum textTarget){
    bind();
    stencilBuffer = texture;
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,  GL_STENCIL_ATTACHMENT, texture->getTarget(),texture->getId(), 0);

}


void Framebuffer::attachTextureDepthStencil(raw_Texture* texture, GLenum textTarget){
    bind();
    depthBuffer = texture;
    stencilBuffer = texture;
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,  GL_DEPTH_STENCIL_ATTACHMENT, texture->getTarget(),texture->getId(), 0);
}

void Framebuffer::blitDepth(int otherId){
    glBindFramebuffer(GL_READ_FRAMEBUFFER, id);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, otherId);
    glBlitFramebuffer(0, 0, depthBuffer->getWidth(), depthBuffer->getHeight(), 0, 0, depthBuffer->getWidth(), depthBuffer->getHeight(),GL_DEPTH_BUFFER_BIT, GL_NEAREST);
}

void Framebuffer::blitColor(int otherId){
    glBindFramebuffer(GL_READ_FRAMEBUFFER, id);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, otherId);
    glBlitFramebuffer(0, 0, colorBuffers[0]->getWidth(), colorBuffers[0]->getHeight(), 0, 0, colorBuffers[0]->getWidth(), colorBuffers[0]->getHeight(),GL_COLOR_BUFFER_BIT, GL_NEAREST);
}

void Framebuffer::resize(int width, int height)
{
    if(depthBuffer)
        depthBuffer->resize(width,height);
    if(stencilBuffer)
        stencilBuffer->resize(width,height);
    for(raw_Texture* t : colorBuffers)
        t->resize(width,height);
}
