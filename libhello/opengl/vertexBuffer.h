#pragma once

#include <GL/glew.h>
#include "libhello/opengl/vertex.h"
#include <iostream>

#include <vector>

using std::cerr;
using std::cout;
using std::endl;

/*
 *  The VertexBuffer stores raw vertex data.
 *  If your mesh is indexed use IndexedVertexBuffer instead.
 *
 */

template<class vertex_t>
class VertexBuffer{
protected:
    int vertex_count;
    int draw_mode;
    GLuint gl_vertex_buffer, gl_vao;

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

    VertexBuffer(){}
    ~VertexBuffer(){deleteGLBuffers();}


    /*
     *  Creates OpenGL buffers and uploads data.
     *  'vertices' can be deleted after that.
     *
     *  A VBO and a VAO will be created and initialized.
     */

    void set(std::vector<vertex_t> &vertices);
    void set(vertex_t* vertices,int vertex_count);

    /*
     *  Updates the existing OpenGL buffer.
     *  Have to be called after 'set()'.
     *
     *  Replaces the vertex at 'vertex_offset' and the following 'vertex_count'
     *  vertices in the current buffer by uploading them to OpenGL.
     *  'vertices' can be deleted after that.
     */

    void updateVertexBuffer(vertex_t* vertices,int vertex_count, int vertex_offset);

    /*
     *  Deletes all OpenGL buffers.
     *  Will be called by the destructor automatically
     */

    void deleteGLBuffers();

    /*
     *  Binds/Unbinds OpenGL buffers.
     */

    void bind() const;
    void unbind() const;

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

    void setDrawMode(int draw_mode);


    int getVBO(){return gl_vertex_buffer;}
    int getVAO(){return gl_vao;}
};


template<class vertex_t>
void VertexBuffer<vertex_t>::bindAndDraw() const{
    bind();
    draw();
    unbind();
}

template<class vertex_t>
void VertexBuffer<vertex_t>::setDrawMode(int draw_mode){
    this->draw_mode = draw_mode;
}

template<class vertex_t>
void VertexBuffer<vertex_t>::set(std::vector<vertex_t> &vertices){
    set(&vertices[0],vertices.size());
}

template<class vertex_t>
void VertexBuffer<vertex_t>::set(vertex_t* vertices,int vertex_count){
    this->vertex_count = vertex_count;


    //create VBO
    glGenBuffers( 1, &gl_vertex_buffer );
    glBindBuffer( GL_ARRAY_BUFFER, gl_vertex_buffer );
    glBufferData( GL_ARRAY_BUFFER, vertex_count * sizeof(vertex_t), vertices, GL_STATIC_DRAW );

    //create VAO and init
    glGenVertexArrays(1, &gl_vao);
    glBindVertexArray(gl_vao);

    glBindBuffer( GL_ARRAY_BUFFER, gl_vertex_buffer );
    setVertexAttributes();

    glBindVertexArray(0);
    glBindBuffer( GL_ARRAY_BUFFER, 0 );

}

template<class vertex_t>
void VertexBuffer<vertex_t>::deleteGLBuffers(){
    //glDeleteBuffers silently ignores 0's and names that do not correspond to existing buffer objects
    glDeleteBuffers( 1, &gl_vertex_buffer );
    gl_vertex_buffer = 0;
    glDeleteVertexArrays(1, &gl_vao);
   gl_vao = 0;

}

template<class vertex_t>
void VertexBuffer<vertex_t>::updateVertexBuffer(vertex_t* vertices,int vertex_count, int vertex_offset){
    glBindBuffer( GL_ARRAY_BUFFER, gl_vertex_buffer );
    glBufferSubData(GL_ARRAY_BUFFER,vertex_offset*sizeof(vertex_t),vertex_count*sizeof(vertex_t),vertices);
}

template<class vertex_t>
void VertexBuffer<vertex_t>::bind() const{
    glBindVertexArray(gl_vao);
}

template<class vertex_t>
void VertexBuffer<vertex_t>::unbind() const{
    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
    glBindVertexArray(0);
}

template<class vertex_t>
void VertexBuffer<vertex_t>::draw() const{
    glDrawArrays(draw_mode,0,vertex_count);
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
}

template<>
void VertexBuffer<Vertex>::setVertexAttributes();
template<>
void VertexBuffer<VertexN>::setVertexAttributes();
template<>
void VertexBuffer<VertexNT>::setVertexAttributes();

