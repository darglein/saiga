#pragma once

#include "libhello/util/glm.h"
#include "libhello/opengl/vertex.h"

#include <vector>
#include <glm/gtc/epsilon.hpp>

class aabb
{
public:
    vec3 min,max;

	aabb(void);

    aabb(const glm::vec3 &p, const glm::vec3 &s);
	~aabb(void);




    void makeNegative();
    void growBox(const vec3 &v);
    void growBox(const aabb &v);

    void transform(const mat4 &trafo);
	void translate(const glm::vec3 &v);
    void scale(const glm::vec3 &s);
    float height(){ return max.y-min.y;}
    void ensureValidity();

	int intersect(const aabb &other);
	bool intersect2(const aabb &other);
    bool intersectTouching(const aabb &other); //returns true if boxes are touching

    bool intersect2(const aabb &other, int side);
    int touching(const aabb &other); //used for visibility check (if air block is touching a cube the touched face is visible)




    int intersectAabb(const aabb &other);
    bool intersectAabb2(const aabb &other);
    void getMinimumAabb(aabb &box){ box = *this;}

    vec3 cornerPoint(int i) const;

    vec3 getPosition() const;
    void setPosition(const glm::vec3 &v);

    bool contains(const glm::vec3 &p);

//    void addOutlineToBuffer(std::vector<Vertex> &vertices,std::vector<GLuint> &indices);
//    void addToBuffer(std::vector<Vertex> &vertices,std::vector<GLuint> &indices);
//    void addToBuffer(std::vector<VertexN> &vertices,std::vector<GLuint> &indices);

//    void getDrawData(GLfloat* vert, GLuint *vertpointer, GLuint *facedata, GLuint *facepointer, char visibility, int id=0); //cube + id (7 floats per vertex)
//    void getDrawDataTx(GLfloat* vert, GLuint *vertpointer, GLuint *facedata, GLuint *facepointer, int visibility); //cube + texture coordinates (8 floats per  vertex)

	friend std::ostream& operator<<(std::ostream& os, const aabb& dt);
};

