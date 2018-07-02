/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/templatedBuffer.h"
#include "saiga/opengl/instancedBuffer.h"
#include "saiga/opengl/opengl.h"
#include "saiga/geometry/vertex.h"

#include <vector>

namespace Saiga {

/*
 *  A combination of a gl_array_buffer and a vertex array object (vao).
 * The VertexBuffer stores raw vertex data.
 *  If your mesh is indexed use IndexedVertexBuffer instead.
 *
 * You can add additional buffers to the vao like this:
 *
 * VertexBuffer<Particle> particleBuffer;
 * TemplatedBuffer<vec4> colorBuffer;
 *
 * //init both buffers with data
 * //...
 *
 * //Add the color buffer, which is a simple GL_ARRAY_BUFFER, to the location 3
 * //of the particleBuffer VAO.
 * particleBuffer.bind();
 * colorBuffer.bind();
 * glEnableVertexAttribArray( 3 );
 * glVertexAttribPointer(3,4, GL_FLOAT, GL_FALSE, sizeof(vec4), NULL );
 * particleBuffer.unbind();
 *
 *
 *
 */

template<class vertex_t>
class VertexBuffer : public TemplatedBuffer<vertex_t>{
private:
//    int vertex_count;
protected:
    GLenum draw_mode;
    GLuint  gl_vao = 0;

    /*
     *  Tells OpenGL how to handle the vertices.
     *  If you are not using the default Vertex types, you have to
     *  specialize this methode.
     *
     *  A specialized methode should look like this:
     *
     *  //Enable attributes
     *  glEnableVertexAttribArray( 0 );
     *  //Set Attribute Pointers
     *  glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), NULL );
     *  ...
     */

    void setVertexAttributes();


public:

    /*
     *  Create VertexBuffer object.
     *  Does not create any OpenGL buffers.
     *  use set() to initialized buffers.
     */

    VertexBuffer() : TemplatedBuffer<vertex_t>(GL_ARRAY_BUFFER){}
    ~VertexBuffer(){ deleteGLBuffer(); }


    /*
     *  Creates OpenGL buffers and uploads data.
     *  'vertices' can be deleted after that.
     *
     *  A VBO and a VAO will be created and initialized.
     */

    void set(std::vector<vertex_t> &vertices, GLenum usage);
    void set(vertex_t* vertices, int vertex_count, GLenum usage);

    /*
     *  Updates the existing OpenGL buffer.
     *  Have to be called after 'set()'.
     *
     *  Replaces the vertex at 'vertex_offset' and the following 'vertex_count'
     *  vertices in the current buffer by uploading them to OpenGL.
     *  'vertices' can be deleted after that.
     */

//    void updateVertexBuffer(vertex_t* vertices,int vertex_count, int vertex_offset);

    /*
     *  Deletes all OpenGL buffers.
     *  Will be called by the destructor automatically
     */

    void deleteGLBuffer();

    /*
     *  Binds/Unbinds OpenGL buffers.
     */

    void bind() const;
    void unbind() const;

    /**
     *  Adds an instanced buffer to this vertex array object.
     *  While rendering the instanced buffer does not have to be bound again.
     */

    template<typename data_t>
    void addInstancedBuffer(InstancedBuffer<data_t> &buffer, int location, int divisor=1);

    /*
     *  Draws the vertex array in the specified draw mode.
     *  Uses consecutive vertices to form the specified primitive.
     *  E.g. if the draw mode is GL_TRIANGLES
     *  1.Triangle =  (Vertex 0, Vertex 1, Vertex 2)
     *  2.Triangle =  (Vertex 3, Vertex 4, Vertex 5)
     *  ...
     *
     *  If you want to use one vertex for multiple faces use IndexedVertexBuffer instead.
     */

    void draw() const;

    /*
     * Like 'draw()' but only renders a part of the buffer
     */
    void draw(int startVertex, int count) const;

    void drawInstanced(int instances) const;
    void drawInstanced(int instances, int offset, int length) const;


    /*
     *  1. bind()
     *  2. draw()
     *  3. unbind()
     */

    void bindAndDraw() const;

    /*
     *  Set draw type of primitives.
     *
     *  Allowed values:
     *  GL_POINTS, GL_LINE_STRIP, GL_LINE_LOOP, GL_LINES, GL_LINE_STRIP_ADJACENCY,
     *  GL_LINES_ADJACENCY, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN, GL_TRIANGLES,
     *  GL_TRIANGLE_STRIP_ADJACENCY, GL_TRIANGLES_ADJACENCY, GL_PATCHES
     */

    void setDrawMode(GLenum draw_mode);


    int getVBO(){return TemplatedBuffer<vertex_t>::buffer;}
    int getVAO(){return gl_vao;}
};




template<class vertex_t>
void VertexBuffer<vertex_t>::bindAndDraw() const{
    bind();
    draw();
    unbind();
}

template<class vertex_t>
void VertexBuffer<vertex_t>::setDrawMode(GLenum _draw_mode){
    this->draw_mode = _draw_mode;
}

template<class vertex_t>
void VertexBuffer<vertex_t>::set(std::vector<vertex_t> &vertices, GLenum _usage){
    set(&vertices[0],(int)vertices.size(),_usage);
}

template<class vertex_t>
void VertexBuffer<vertex_t>::set(vertex_t* vertices,int _vertex_count, GLenum _usage){
//    this->vertex_count = _vertex_count;

    deleteGLBuffer();
    assert_no_glerror();

    TemplatedBuffer<vertex_t>::set(vertices,_vertex_count,_usage);
//    createGLBuffer(vertices,_vertex_count * sizeof(vertex_t),usage);

    //create VAO and init
    glGenVertexArrays(1, &gl_vao);
    glBindVertexArray(gl_vao);
    assert_no_glerror();

    Buffer::bind();

    setVertexAttributes();

    glBindVertexArray(0);
    glBindBuffer( TemplatedBuffer<vertex_t>::target, 0 );

    assert_no_glerror();
}

template<class vertex_t>
void VertexBuffer<vertex_t>::deleteGLBuffer(){
    //glDeleteBuffers silently ignores 0's and names that do not correspond to existing buffer objects
    TemplatedBuffer<vertex_t>::deleteGLBuffer();
    if(gl_vao)
    {
        glDeleteVertexArrays(1, &gl_vao);
        gl_vao = 0;
        assert_no_glerror();
    }

}


template<class vertex_t>
void VertexBuffer<vertex_t>::bind() const{
    glBindVertexArray(gl_vao);
    assert_no_glerror();
}

template<class vertex_t>
void VertexBuffer<vertex_t>::unbind() const{
    glBindVertexArray(0);
    assert_no_glerror();
}

template<class vertex_t>
template<typename data_t>
void VertexBuffer<vertex_t>::addInstancedBuffer(InstancedBuffer<data_t> &buffer, int location, int divisor)
{
    bind();
    buffer.setAttributes(location,divisor);
    unbind();
}

template<class vertex_t>
void VertexBuffer<vertex_t>::draw() const{
    draw(0,TemplatedBuffer<vertex_t>::getElementCount());
    assert_no_glerror();
}

template<class vertex_t>
void VertexBuffer<vertex_t>::draw(int startVertex, int count) const{
    glDrawArrays(draw_mode,startVertex,count);
    assert_no_glerror();
}

template<class vertex_t>
void VertexBuffer<vertex_t>::drawInstanced(int instances) const
{
    drawInstanced(instances,0,TemplatedBuffer<vertex_t>::getElementCount());
    assert_no_glerror();
}

template<class vertex_t>
void VertexBuffer<vertex_t>::drawInstanced(int instances, int offset, int length) const
{
    glDrawArraysInstanced(draw_mode,offset,length,instances);
    assert_no_glerror();
}


//=========================================================================

template<class vertex_t>
void VertexBuffer<vertex_t>::setVertexAttributes(){
    cerr<<"Warning: I don't know how to bind this Vertex Type. Please use the vertices in vertex.h or write your own bind function"<<endl;
    cerr<<"If you want to write your own bind function use this template:"<<endl;
    cerr<<"\ttemplate<>"<<endl;
    cerr<<"\tvoid VertexBuffer<YOUR_VERTEX_TYPE>::bindVertices(){"<<endl;
    cerr<<"\t\t//bind code"<<endl;
    cerr<<"\t}"<<endl;
    SAIGA_ASSERT(0);
}



template<>
SAIGA_GLOBAL void VertexBuffer<Vertex>::setVertexAttributes();
template<>
SAIGA_GLOBAL void VertexBuffer<VertexN>::setVertexAttributes();
template<>
SAIGA_GLOBAL void VertexBuffer<VertexNT>::setVertexAttributes();
template<>
SAIGA_GLOBAL void VertexBuffer<VertexNTD>::setVertexAttributes();
template<>
SAIGA_GLOBAL void VertexBuffer<VertexNC>::setVertexAttributes();


}
