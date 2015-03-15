#include "opengl/framebuffer.h"
#include "libhello/util/error.h"
Framebuffer::Framebuffer(){

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
        case 0x8CDB:                                        std::cerr <<("GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER\n") << std::endl;          break;
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:          std::cerr <<("GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT\n") << std::endl;           break;
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:  std::cerr <<("GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT\n") << std::endl;   break;
        case GL_FRAMEBUFFER_UNSUPPORTED:                    std::cerr <<("GL_FRAMEBUFFER_UNSUPPORTED\n")<< std::endl ;                     break;
        default:                                            std::cerr <<"Unknown issue " << status << std::endl;                     break;
        }

        std::cerr << "Framebuffer error!" << std::endl;
        exit(1);
    }
}

void Framebuffer::attachTexture(Texture* texture){
    bind();
    int index = colorBuffers.size();
    colorBuffers.push_back(texture);
    int cid = GL_COLOR_ATTACHMENT0+index;
    //    glFramebufferTexture(GL_FRAMEBUFFER, cid,texture->id, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, cid,GL_TEXTURE_2D,texture->getId(), 0);

}

void Framebuffer::attachTextureDepth(Texture* texture){
    bind();
    depthBuffer = texture;
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,  GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,texture->getId(), 0);

}

void Framebuffer::attachTextureDepthStencil(Texture* texture){
    bind();
    depthBuffer = texture;
    stencilBuffer = texture;
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,  GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D,texture->getId(), 0);
}

void Framebuffer::makeToDeferredFramebuffer(int w, int h){

    Texture* color = new Texture();
    color->createEmptyTexture(w,h,GL_RGB,GL_RGB8,GL_UNSIGNED_BYTE);
    attachTexture(color);



    Texture* normal = new Texture();
    //    normal->createEmptyTexture(w,h,GL_RGB,GL_RGB8 ,GL_UNSIGNED_BYTE);
    //    normal->createEmptyTexture(w,h,GL_RGB,GL_RGB16 ,GL_UNSIGNED_SHORT);
    normal->createEmptyTexture(w,h,GL_RGB,GL_RGB32F ,GL_FLOAT);
    attachTexture(normal);


    Texture* position = new Texture();
    position->createEmptyTexture(w,h,GL_RGB,GL_RGB32F ,GL_FLOAT);
    attachTexture(position);


    Texture* colorFinal = new Texture();
    colorFinal->createEmptyTexture(w,h,GL_RGB,GL_RGB8,GL_UNSIGNED_BYTE);
    attachTexture(colorFinal);

   // Texture* depth = new Texture();
    //        depth->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT16,GL_UNSIGNED_SHORT);
   // depth->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32,GL_UNSIGNED_INT);
    //    depth->createEmptyTexture(w,h,GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32F,GL_FLOAT);
    //attachTextureDepth(depth);

    //depth and stencil texture combined
        Texture* depth_stencil = new Texture();
        depth_stencil->createEmptyTexture(w,h,GL_DEPTH_STENCIL, GL_DEPTH24_STENCIL8,GL_UNSIGNED_INT_24_8);
        attachTextureDepthStencil(depth_stencil);

    GLenum DrawBuffers[4] = {GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1,GL_COLOR_ATTACHMENT2,GL_COLOR_ATTACHMENT3};
    glDrawBuffers(4, DrawBuffers);


    check();
    unbind();


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
