# Copyright (C) 2017,2018 Rodrigo Jose Hernandez Cordoba
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

find_program(GLSLANG_VALIDATOR_EXECUTABLE 
    NAMES
        glslangValidator
    PATHS
        /usr/local
)

find_path(GLSLANG_SPIRV_INCLUDE_DIR
    SPIRV/spirv.hpp
    PATHS
        /usr/local
        /mingw64
        /mingw32
)

find_library(GLSLANG_LIB
    NAMES
        glslang
        glslangd
    PATHS
        /usr/local
        /mingw64
        /mingw32
)

find_library(OGLCompiler_LIB
    NAMES
        OGLCompiler
    PATHS
        /usr/local
        /mingw64
        /mingw32
)

find_library(OSDependent_LIB
    NAMES
        OSDependent
    PATHS
        /usr/local
        /mingw64
        /mingw32
)

find_library(HLSL_LIB
    NAMES
        HLSL
    PATHS
        /usr/local
        /mingw64
        /mingw32
)

find_library(SPIRV_LIB
    NAMES
        SPIRV
    PATHS
        /usr/local
        /mingw64
        /mingw32
)

find_library(SPIRV_REMAPPER_LIB
    NAMES
        SPVRemapper
    PATHS
        /usr/local
        /mingw64
        /mingw32
)

# - Locate Debug Libraries if they exist -

find_library(GLSLANG_DEBUG_LIB
    NAMES
        glslangd
    PATHS
        /usr/local
        /mingw64
        /mingw32
)

find_library(OGLCompiler_DEBUG_LIB
    NAMES
        OGLCompilerd
    PATHS
        /usr/local
        /mingw64
        /mingw32
)

find_library(OSDependent_DEBUG_LIB
    NAMES
        OSDependentd
    PATHS
        /usr/local
        /mingw64
        /mingw32
)

find_library(HLSL_DEBUG_LIB
    NAMES
        HLSLd
    PATHS
        /usr/local
        /mingw64
        /mingw32
)

find_library(SPIRV_DEBUG_LIB
    NAMES
        SPIRVd
    PATHS
        /usr/local
        /mingw64
        /mingw32
)

find_library(SPIRV_REMAPPER_DEBUG_LIB
    NAMES
        SPVRemapperd
    PATHS
        /usr/local
        /mingw64
        /mingw32
)

if(GLSLANG_LIB AND GLSLANG_DEBUG_LIB)
    list(APPEND GLSLANG_LIBRARIES optimized ${GLSLANG_LIB} debug ${GLSLANG_DEBUG_LIB})
elseif(GLSLANG_LIB)
    list(APPEND GLSLANG_LIBRARIES ${GLSLANG_LIB})
elseif(GLSLANG_DEBUG_LIB)
    list(APPEND GLSLANG_LIBRARIES ${GLSLANG_DEBUG_LIB})
endif()

if(OGLCompiler_LIB AND OGLCompiler_DEBUG_LIB)
    list(APPEND GLSLANG_LIBRARIES optimized ${OGLCompiler_LIB} debug ${OGLCompiler_DEBUG_LIB}) 
elseif(OGLCompiler_LIB)
    list(APPEND GLSLANG_LIBRARIES ${OGLCompiler_LIB}) 
elseif(OGLCompiler_DEBUG_LIB)
    list(APPEND GLSLANG_LIBRARIES ${OGLCompiler_DEBUG_LIB}) 
endif()

if(OSDependent_LIB AND OSDependent_DEBUG_LIB)
    list(APPEND GLSLANG_LIBRARIES optimized ${OSDependent_LIB} debug ${OSDependent_DEBUG_LIB})
elseif(OSDependent_LIB)
    list(APPEND GLSLANG_LIBRARIES ${OSDependent_LIB})
elseif(OSDependent_DEBUG_LIB)
    list(APPEND GLSLANG_LIBRARIES ${OSDependent_DEBUG_LIB})
endif()

if(HLSL_LIB AND HLSL_DEBUG_LIB)
    list(APPEND GLSLANG_LIBRARIES optimized ${HLSL_LIB} debug ${HLSL_DEBUG_LIB})
elseif(HLSL_LIB)
    list(APPEND GLSLANG_LIBRARIES ${HLSL_LIB})
elseif(HLSL_DEBUG_LIB)
    list(APPEND GLSLANG_LIBRARIES ${HLSL_DEBUG_LIB})
endif()

if(SPIRV_LIB AND SPIRV_DEBUG_LIB)
    list(APPEND GLSLANG_LIBRARIES optimized ${SPIRV_LIB} debug ${SPIRV_DEBUG_LIB})
elseif(SPIRV_LIB)
    list(APPEND GLSLANG_LIBRARIES ${SPIRV_LIB})
elseif(SPIRV_DEBUG_LIB)
    list(APPEND GLSLANG_LIBRARIES ${SPIRV_DEBUG_LIB})
endif()

if(SPIRV_REMAPPER_LIB AND SPIRV_REMAPPER_DEBUG_LIB)
    list(APPEND GLSLANG_LIBRARIES optimized ${SPIRV_REMAPPER_LIB} debug ${SPIRV_REMAPPER_DEBUG_LIB})
elseif(SPIRV_REMAPPER_LIB)
    list(APPEND GLSLANG_LIBRARIES ${SPIRV_REMAPPER_LIB})
elseif(SPIRV_REMAPPER_DEBUG_LIB)
    list(APPEND GLSLANG_LIBRARIES ${SPIRV_REMAPPER_DEBUG_LIB})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GLslang 
    DEFAULT_MSG
    GLSLANG_LIB
    OGLCompiler_LIB
    OSDependent_LIB
    HLSL_LIB
    SPIRV_LIB
    SPIRV_REMAPPER_LIB    
    GLSLANG_VALIDATOR_EXECUTABLE
    GLSLANG_SPIRV_INCLUDE_DIR
    GLSLANG_LIBRARIES
)

mark_as_advanced(
    GLSLANG_LIB
    OGLCompiler_LIB
    OSDependent_LIB
    HLSL_LIB
    SPIRV_LIB
    SPIRV_REMAPPER_LIB
    GLSLANG_LIBRARIES
GLSLANG_VALIDATOR_EXECUTABLE
GLSLANG_SPIRV_INCLUDE_DIR
GLSLANG_DEBUG_LIB
HLSL_DEBUG_LIB
OGLCompiler_DEBUG_LIB
OSDependent_DEBUG_LIB
SPIRV_DEBUG_LIB
SPIRV_REMAPPER_DEBUG_LIB
)
