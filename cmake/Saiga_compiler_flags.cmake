######### warnings #########

if(MSVC)
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W1")
	# Force to always compile with W1
	if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
		string(REGEX REPLACE "/W[0-4]" "/W1" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	else()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W1")
	endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long")
endif()

######### basic #########

if(UNIX)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
endif(UNIX)

if(MSVC)
	#multiprocessor compilation for visual studio
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP") 
endif()

######### floating point #########

if(SAIGA_STRICT_FP)
	if(UNIX)
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mfpmath=sse")
	endif()
	if(MSVC)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:strict")
	endif()
endif()

######### debug and optimization #########

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
