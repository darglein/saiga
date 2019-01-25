######### warnings #########

if(MSVC)
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W1")
	# Force to always compile with W1
	if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
		string(REGEX REPLACE "/W[0-4]" "/W1" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	else()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W1")
	endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-aliasing")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-braces")
endif()

######### basic #########

if(UNIX)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
endif(UNIX)

if(MSVC)
	#multiprocessor compilation for visual studio
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP") 
	add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
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
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Oi /Ot /Oy /GL /fp:fast /Gy /arch:AVX2")
		set(CMAKE_LD_FLAGS "${CMAKE_LD_FLAGS} /LTCG")
		add_definitions(-D__FMA__)
	endif()
endif()
