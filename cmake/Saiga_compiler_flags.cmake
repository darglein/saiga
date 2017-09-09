######### enable warnings #########

if(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W1")
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long")
endif()



if(UNIX)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fvisibility=hidden")
endif(UNIX)

if(MSVC)
	#multiprocessor compilation for visual studio
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP") 
endif()

#strict fp behaviour flags if determinism is required
if(SAIGA_STRICT_FP)
	if(UNIX)
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mfpmath=sse")
	endif()
	if(MSVC)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:strict")
	endif()
endif()

if(SAIGA_CUDA_DEBUG)
	add_definitions(-DCUDA_DEBUG)
else()
	add_definitions(-DCUDA_NDEBUG)
endif()

if(SAIGA_FULL_OPTIMIZE)
	if(UNIX)
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
	endif()
	if(MSVC)
		#todo check if avx is present
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Oi /GL /fp:fast /Gy /arch:AVX2")
	endif()
endif()
