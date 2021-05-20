############# Check which compiler is used ############

set(SAIGA_CXX_WCLANG 0)
set(SAIGA_CXX_CLANG 0)
set(SAIGA_CXX_GNU 0)
set(SAIGA_CXX_INTEL 0)
set(SAIGA_CXX_MSVC 0)

unset(SAIGA_COMPILER_STRING)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(WIN32)
    set(SAIGA_CXX_WCLANG 1)
    set(SAIGA_COMPILER_STRING "MS-Clang")
  else()
    set(SAIGA_CXX_CLANG 1)
    set(SAIGA_COMPILER_STRING "Clang")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(SAIGA_CXX_GNU 1)
  set(SAIGA_COMPILER_STRING "GNU")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(SAIGA_CXX_INTEL 1)
  set(SAIGA_COMPILER_STRING "Intel")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(SAIGA_CXX_MSVC 1)
  set(SAIGA_COMPILER_STRING "MSVC")
else()
  message(FATAL_ERROR "Unknown CXX Compiler. '${CMAKE_CXX_COMPILER_ID}'")
endif()


message(STATUS "Detected Compiler:  ${SAIGA_COMPILER_STRING}")
message(STATUS "Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}" )


######### warnings #########


if(SAIGA_CXX_MSVC)
  # Force to always compile with W1
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W1" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W1")
  endif()
endif()

if(SAIGA_CXX_CLANG OR SAIGA_CXX_GNU)
  list(APPEND SAIGA_CXX_FLAGS "-Wall")
  list(APPEND SAIGA_CXX_FLAGS "-Werror=return-type")
  list(APPEND SAIGA_CXX_FLAGS "-Wno-strict-aliasing")
  list(APPEND SAIGA_CXX_FLAGS "-Wno-sign-compare")

endif()

if (SAIGA_CXX_CLANG)
  list(APPEND SAIGA_CXX_FLAGS "-Wno-missing-braces")
endif()

######### basic #########

if(SAIGA_LIBSTDCPP AND SAIGA_CXX_CLANG)
  #list(APPEND SAIGA_CXX_FLAGS "-stdlib=libstdc++")
  #set(CMAKE_LD_FLAGS "${CMAKE_LD_FLAGS} -stdlib=libstdc++")
  #SET(LIBS ${LIBS} "-lstdc++")
else()
  #SET(SAIGA_CXX_FLAGS "${SAIGA_CXX_FLAGS} -stdlib=libc++")
endif()

if(SAIGA_CXX_CLANG OR SAIGA_CXX_GNU)
  list(APPEND SAIGA_PRIVATE_CXX_FLAGS "-fvisibility=hidden")
endif()

if(SAIGA_CXX_WCLANG)
  # Fixes: cannot use 'throw' with exceptions disabled
  list(APPEND SAIGA_CXX_FLAGS "-Xclang -fcxx-exceptions")
  # some eigen header generates this warning
  add_definitions(-D_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING)
endif()

if(SAIGA_CXX_MSVC OR SAIGA_CXX_WCLANG)
  # required for some crazy eigen stuff
  list(APPEND SAIGA_CXX_FLAGS "/bigobj")
  #multiprocessor compilation for visual studio
  list(APPEND SAIGA_CXX_FLAGS "/MP")
  set(CMAKE_CXX_FLAGS "/MP ${CMAKE_CXX_FLAGS}")
  add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
endif()

######### floating point #########

if(SAIGA_STRICT_FP)
  if(SAIGA_CXX_CLANG OR SAIGA_CXX_GNU)
    list(APPEND SAIGA_CXX_FLAGS "-msse2" "-mfpmath=sse")
  endif()
  if(SAIGA_CXX_MSVC)
    list(APPEND SAIGA_CXX_FLAGS "/fp:strict")
  endif()
endif()

if(SAIGA_FAST_MATH)
  if(SAIGA_CXX_CLANG OR SAIGA_CXX_GNU)
    list(APPEND SAIGA_CXX_FLAGS "-ffast-math")
  endif()
endif()

######### debug and optimization #########

if(SAIGA_CUDA_DEBUG)
  add_definitions(-DCUDA_DEBUG)
else()
  add_definitions(-DCUDA_NDEBUG)
endif()

if(SAIGA_FULL_OPTIMIZE OR SAIGA_ARCHNATIVE)
  if(SAIGA_CXX_CLANG OR SAIGA_CXX_GNU)
    list(APPEND SAIGA_CXX_FLAGS "-march=native")
  endif()
endif()

if(SAIGA_FULL_OPTIMIZE)
  if(SAIGA_CXX_MSVC)
    #todo check if avx is present
    set(SAIGA_CXX_FLAGS "${SAIGA_CXX_FLAGS} /Oi /Ot /Oy /GL /fp:fast /Gy")
    set(CMAKE_LD_FLAGS "${CMAKE_LD_FLAGS} /LTCG")
    add_definitions(-D__FMA__)
  endif()
endif()

# Sanitizers
# https://github.com/google/sanitizers
if(SAIGA_DEBUG_ASAN)
  if(SAIGA_CXX_CLANG OR SAIGA_CXX_GNU)
    SET(CMAKE_BUILD_TYPE Debug)
    list(APPEND SAIGA_CXX_FLAGS "-fsanitize=address")
    list(APPEND SAIGA_LD_FLAGS "-fsanitize=address")
    list(APPEND SAIGA_CXX_FLAGS "-fno-omit-frame-pointer")
  else()
    message(FATAL_ERROR "ASAN not supported for your compiler.")
  endif()
endif()

if(SAIGA_DEBUG_MSAN)
  if(SAIGA_CXX_CLANG OR SAIGA_CXX_GNU)
    SET(CMAKE_BUILD_TYPE Debug)
    list(APPEND SAIGA_CXX_FLAGS "-fsanitize=memory")
    list(APPEND SAIGA_LD_FLAGS "-fsanitize=memory")
    list(APPEND SAIGA_CXX_FLAGS "-fno-omit-frame-pointer")
  else()
    message(FATAL_ERROR "MSAN not supported for your compiler.")
  endif()
endif()

if(SAIGA_DEBUG_TSAN)
  if(SAIGA_CXX_CLANG OR SAIGA_CXX_GNU)
    list(APPEND SAIGA_CXX_FLAGS "-fsanitize=thread -static-libasan -fno-omit-frame-pointer -g")
  else()
    message(FATAL_ERROR "TSAN not supported for your compiler.")
  endif()
endif()


if(SAIGA_DEBIAN_BUILD)
  if( NOT SAIGA_CXX_CLANG AND NOT SAIGA_CXX_GNU)
    message(FATAL_ERROR "Only GCC and Clang is supported for a debian build.")
  endif()
  list(APPEND SAIGA_CXX_FLAGS "-mavx2 -mfma")
endif()
