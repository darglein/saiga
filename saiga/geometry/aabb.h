#pragma once

#include <saiga/config.h>
#include <saiga/util/glm.h>


class SAIGA_GLOBAL aabb
{
public:
    vec3 min,max;

	aabb(void);

    aabb(const glm::vec3 &p, const glm::vec3 &s);
	~aabb(void);


    int maxDimension(); //returns the axis with the maximum extend

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
    int touching(const aabb &other);

    vec3 getHalfExtends();


    int intersectAabb(const aabb &other);
    bool intersectAabb2(const aabb &other);
    void getMinimumAabb(aabb &box){ box = *this;}

    vec3 cornerPoint(int i) const;

    vec3 getPosition() const;
    void setPosition(const glm::vec3 &v);

    bool contains(const glm::vec3 &p);



	friend std::ostream& operator<<(std::ostream& os, const aabb& dt);
};

