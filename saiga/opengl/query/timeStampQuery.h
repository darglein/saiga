#pragma once

#include "saiga/opengl/opengl.h"

class SAIGA_GLOBAL TimeStampQuery{
private:

    GLuint id = 0;
public:
    TimeStampQuery();
    ~TimeStampQuery();
    void create();

    /**
     * Places this query in the command queue.
     * After all previous commands have been finished the timestamp becomes available.
     * This call return immediately.
     */
    void record();

    bool isAvailable();

    GLuint64 getTimestamp();

    /**
     * Blocks until the query gets available.
     */
    GLuint64 waitTimestamp();
};
