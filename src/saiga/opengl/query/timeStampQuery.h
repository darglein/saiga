/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/opengl.h"

namespace Saiga
{
class SAIGA_OPENGL_API QueryObject
{
   private:
    GLuint id = 0;

   public:
    QueryObject();
    ~QueryObject();

    // Note: The query state is not copied.
    QueryObject(const QueryObject& other);

    // Create/destroy the underlying OpenGL objects.
    void create(bool initial_record);
    void destroy();

    /**
     * Places this query in the command queue.
     * After all previous commands have been finished the timestamp becomes available.
     * This call return immediately.
     */
    void record();


    void begin();
    void end();

    bool isAvailable();

    GLuint64 getTimestamp();

    /**
     * Blocks until the query gets available.
     */
    GLuint64 waitTimestamp();
};

}  // namespace Saiga
