
if(UNIX)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fvisibility=hidden")
endif(UNIX)

if(MSVC)
	#multiprocessor compilation for visual studio
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP") 
endif()

#strict fp behaviour flags if determinism is required
if(STRICT_FP)
	if(UNIX)
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mfpmath=sse")
	endif()
	if(MSVC)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:strict")
	endif()
endif()

if(CUDA_DEBUG)
	add_definitions(-DCUDA_DEBUG)
else()
	add_definitions(-DCUDA_NDEBUG)
endif()


if(FULL_OPTIMIZE)
	if(UNIX)
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
	endif()
	if(MSVC)
		#todo check if avx is present
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Oi /GL /fp:fast /Gy /arch:AVX2")
	endif()
endif()