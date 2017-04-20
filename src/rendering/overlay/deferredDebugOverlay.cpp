#include "saiga/rendering/overlay/deferredDebugOverlay.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/geometry/triangle_mesh.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/rendering/gbuffer.h"

DeferredDebugOverlay::DeferredDebugOverlay(int width, int height):width(width),height(height){
    proj = glm::ortho(0.0f,(float)width,0.0f,(float)height,1.0f,-1.0f);

    TriangleMesh<VertexNT, GLuint> tm;


    VertexNT verts[4];
    //bottom left
    verts[0] = VertexNT(vec3(-1,-1,0),
                        vec3(0,0,1),
                        vec2(0,0));
    //bottom right
    verts[1] = VertexNT(vec3(1,-1,0),
                        vec3(0,0,1),
                        vec2(1,0));
    //top right
    verts[2] = VertexNT(vec3(1,1,0),
                        vec3(0,0,1),
                        vec2(1,1));
    //top left
    verts[3] = VertexNT(vec3(-1,1,0),
                        vec3(0,0,1),
                        vec2(0,1));
    tm.addQuad(verts);


    tm.createBuffers(buffer);


    setScreenPosition(&color,0);
    setScreenPosition(&normal,1);
    setScreenPosition(&depth,2);
    setScreenPosition(&data,3);
    setScreenPosition(&light,4);

    loadShaders();

}

void DeferredDebugOverlay::loadShaders()
{
    shader = ShaderLoader::instance()->load<MVPTextureShader>("debug/gbuffer.glsl");
   depthShader = ShaderLoader::instance()->load<MVPTextureShader>("debug/gbuffer_depth.glsl");
    normalShader = ShaderLoader::instance()->load<MVPTextureShader>("debug/gbuffer_normal.glsl");
}

void DeferredDebugOverlay::setScreenPosition(GbufferTexture *gbt, int id)
{
    float images = 5;

    float s = 1.0f/images;
    gbt->setScale(vec3(s));

    float dy = -s*2.0f;
    float y = id*dy+dy*0.5f+1.0f;
    gbt->translateGlobal(vec3(1.0f-s,y,0));
    gbt->calculateModel();
}

void DeferredDebugOverlay::render(){
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    shader->bind();

    shader->uploadModel(color.model);
    shader->uploadTexture(color.texture);
    buffer.bindAndDraw();

    shader->uploadModel(data.model);
    shader->uploadTexture(data.texture);
    buffer.bindAndDraw();

    shader->uploadModel(light.model);
    shader->uploadTexture(light.texture);
    buffer.bindAndDraw();

    shader->unbind();


    normalShader->bind();
    normalShader->uploadModel(normal.model);
    normalShader->uploadTexture(normal.texture);
    buffer.bindAndDraw();
    normalShader->unbind();

    depthShader->bind();

    depthShader->uploadModel(depth.model);
    depthShader->uploadTexture(depth.texture);
    buffer.bindAndDraw();

    depthShader->unbind();
}

void DeferredDebugOverlay::setDeferredFramebuffer(GBuffer *gbuffer, std::shared_ptr<raw_Texture> light)
{
    color.texture = gbuffer->getTextureColor();
    normal.texture = gbuffer->getTextureNormal();
    depth.texture = gbuffer->getTextureDepth();
    data.texture = gbuffer->getTextureData();
    this->light.texture = light;
}
