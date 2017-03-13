#include "saiga/opengl/query/timeStampQuery.h"


TimeStampQuery::TimeStampQuery()
{
	//std::cout << "queeerryyyyyyyyyyyyyyy ......................." << std::endl;
}

TimeStampQuery::~TimeStampQuery()
{
    if(id)
        glDeleteQueries(1, &id);
}

void TimeStampQuery::create()
{
    if(!id)
        glGenQueries(1, &id);
    //prevent potential gl erros.
    record();
	waitTimestamp();
}

void TimeStampQuery::record()
{
    glQueryCounter(id, GL_TIMESTAMP);
}

bool TimeStampQuery::isAvailable()
{
    GLint res;
    glGetQueryObjectiv(id,GL_QUERY_RESULT_AVAILABLE,&res);
    return res!=0;
}

GLuint64 TimeStampQuery::getTimestamp()
{
    GLuint64 time;
    glGetQueryObjectui64v(id, GL_QUERY_RESULT, &time);
    return time;
}

GLuint64 TimeStampQuery::waitTimestamp()
{
    while(!isAvailable()){

    }
    return getTimestamp();
}

