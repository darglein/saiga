/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/opengl/indexedVertexBuffer.h>
#include <saiga/opengl/shader/basic_shaders.h>
#include <saiga/geometry/triangle_mesh.h>
#include <saiga/geometry/aabb.h>
#include <saiga/animation/boneVertex.h>
#include <saiga/animation/animation.h>

#include <saiga/camera/camera.h>

namespace Saiga {

class SAIGA_GLOBAL Asset{
public:
    virtual void render(Camera *cam, const mat4 &model) = 0;
    virtual void renderForward(Camera *cam, const mat4 &model) = 0;
    virtual void renderDepth(Camera *cam, const mat4 &model) = 0;
    virtual void renderWireframe(Camera *cam, const mat4 &model) = 0;
    virtual void renderRaw() = 0;
};


template<typename vertex_t, typename index_t>
class SAIGA_TEMPLATE BasicAsset : public Asset{
public:
    std::string name;
    AABB boundingBox;
    vec3 offset = vec3(0);

    std::shared_ptr<MVPShader> shader;
    std::shared_ptr<MVPShader> forwardShader;
    std::shared_ptr<MVPShader> depthshader;
    std::shared_ptr<MVPShader> wireframeshader;

    TriangleMesh<vertex_t,index_t> mesh;
    IndexedVertexBuffer<vertex_t,index_t> buffer;

    /**
     * Use these for simple inefficient rendering.
     * Every call binds and unbinds the shader and uploads the camera matrices again.
     */

    virtual void render(Camera *cam, const mat4 &model) override;
    virtual void renderForward(Camera *cam, const mat4 &model) override;
    virtual void renderDepth(Camera *cam, const mat4 &model) override;
    virtual void renderWireframe(Camera *cam, const mat4 &model) override;

    /**
     * Renders the mesh.
     * This maps to a single glDraw call and nothing else, so the shader
     * has to be setup before this renderRaw is called.
     */
    virtual void renderRaw() override;


    void create(std::string name,
                std::shared_ptr<MVPShader> shader, std::shared_ptr<MVPShader> forwardShader, std::shared_ptr<MVPShader> depthshader, std::shared_ptr<MVPShader> wireframeshader,
                bool normalizePosition=false, bool ZUPtoYUP=false);


    void normalizePosition();

    void normalizeScale();
    /**
     * Transforms the vertices and normals that the up axis is Y when before the up axis was Z.
     *
     * Many 3D CAD softwares (i.e. Blender) are using a right handed coordinate system with Z pointing upwards.
     * This frameworks uses a right haned system with Y pointing upwards.
     */


    void ZUPtoYUP();

};

template<typename vertex_t, typename index_t>
void BasicAsset<vertex_t,index_t>::render(Camera *cam, const mat4 &model)
{
	(void)cam;
    shader->bind();
//    shader->uploadAll(cam,model);
    shader->uploadModel(model);

//    glEnable(GL_POLYGON_OFFSET_FILL);
//    glPolygonOffset(1.0f,1.0f);

    buffer.bindAndDraw();

//    glDisable(GL_POLYGON_OFFSET_FILL);

    shader->unbind();
}

template<typename vertex_t, typename index_t>
void BasicAsset<vertex_t,index_t>::renderForward(Camera *cam, const mat4 &model)
{
	(void)cam;
    forwardShader->bind();
//    shader->uploadAll(cam,model);
    forwardShader->uploadModel(model);

//    glEnable(GL_POLYGON_OFFSET_FILL);
//    glPolygonOffset(1.0f,1.0f);

    buffer.bindAndDraw();

//    glDisable(GL_POLYGON_OFFSET_FILL);

    forwardShader->unbind();
}

template<typename vertex_t, typename index_t>
void BasicAsset<vertex_t,index_t>::renderDepth(Camera *cam, const mat4 &model)
{
	(void)cam;
    depthshader->bind();
    depthshader->uploadModel(model);
    buffer.bindAndDraw();
    depthshader->unbind();
}

template<typename vertex_t, typename index_t>
void BasicAsset<vertex_t,index_t>::renderWireframe(Camera *cam, const mat4 &model)
{
	(void)cam;
    wireframeshader->bind();
    wireframeshader->uploadModel(model);

//    glEnable(GL_POLYGON_OFFSET_LINE);

    //negative values shifts the wireframe towards the camera,
    //but a non zero factors does strange things for lines and increases
    //the depth on lines with high slope towards the camera by too much.
    //a visually better solution is to shift the triangles back a bit glPolygonOffset(1,1);
    //and draw the wireframe without polygon offset.
//    glPolygonOffset(0.0f,-500.0f);

    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
    buffer.bindAndDraw();
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
//    glDisable(GL_POLYGON_OFFSET_LINE);

    wireframeshader->unbind();
}

template<typename vertex_t, typename index_t>
void BasicAsset<vertex_t,index_t>::renderRaw()
{
    buffer.bindAndDraw();
}

template<typename vertex_t, typename index_t>
void BasicAsset<vertex_t,index_t>::normalizePosition()
{
    offset = boundingBox.getPosition();
    mat4 t = glm::translate(mat4(1),-offset);
    mesh.transform(t);
    boundingBox.setPosition(vec3(0));
}


template<typename vertex_t, typename index_t>
void BasicAsset<vertex_t,index_t>::normalizeScale()
{
    //TODO
    vec3 d = boundingBox.max - boundingBox.min;
   // int m = -1;
    //int mi = -1;

    for(int i = 0 ; i < 3 ; ++i){

    }


    mat4 t = glm::translate(mat4(1),-offset);
    mesh.transform(t);
    boundingBox.setPosition(vec3(0));
}



template<typename vertex_t, typename index_t>
void BasicAsset<vertex_t,index_t>::ZUPtoYUP()
{
    const mat4 m(
                1, 0, 0, 0,
                0, 0, -1, 0,
                0, 1, 0, 0,
                0, 0, 0, 1
                );
    mesh.transform(m);
    mesh.transformNormal(m);
}

template<typename vertex_t, typename index_t>
void BasicAsset<vertex_t,index_t>::create(std::string _name,
                                          std::shared_ptr<MVPShader> _shader, std::shared_ptr<MVPShader> _forwardShader, std::shared_ptr<MVPShader> _depthshader, std::shared_ptr<MVPShader> _wireframeshader,
                                          bool normalizePosition, bool ZUPtoYUP){

    this->name = _name;
    this->shader = _shader;
    this->forwardShader = _forwardShader;
    this->depthshader = _depthshader;
    this->wireframeshader = _wireframeshader;
    this->boundingBox = mesh.calculateAabb();

    if(ZUPtoYUP){
        this->ZUPtoYUP();
    }

    if(normalizePosition){
        this->normalizePosition();
    }
    mesh.createBuffers(buffer);
}

}
