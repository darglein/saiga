
list(APPEND required_modules "saiga_core")

# Make sure this always compiles with optimizations
set(CMAKE_BUILD_TYPE Release)

if(SAIGA_CXX_CLANG OR SAIGA_CXX_GNU)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()
if(SAIGA_CXX_MSVC)
    #todo check if avx is present
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Oi /Ot /Oy /GL /fp:fast /Gy /arch:AVX2")
    set(CMAKE_LD_FLAGS "${CMAKE_LD_FLAGS} /LTCG")
    add_definitions(-D__FMA__)
endif()

saiga_make_sample(required_modules)
