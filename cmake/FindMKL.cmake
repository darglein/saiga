
# This can be specified by the user to point to the install dir
# For windows this might be C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2019.2.190/windows/mkl
set(MKL_DIR "/usr/" CACHE PATH "The directory where mkl was installed.")


find_path(MKL_INCLUDE_DIR
    mkl.h
    PATHS
    /usr/local
    ${MKL_DIR}/include
    PATH_SUFFIXES
    include/mkl
    include
    mkl
    )


find_library(MKL_LIBRARIES_CORE
    mkl_core
    PATHS
    ${MKL_DIR}/lib/intel64
    )

find_library(MKL_LIBRARIES_LP64
    mkl_intel_lp64
    PATHS
    ${MKL_DIR}/lib/intel64
    )

find_library(MKL_LIBRARIES_SEQUENTIAL
    mkl_sequential
    PATHS
    ${MKL_DIR}/lib/intel64
    )

set(MKL_LIBRARIES ${MKL_LIBRARIES_CORE} ${MKL_LIBRARIES_LP64} ${MKL_LIBRARIES_SEQUENTIAL})

if(UNIX)
	set(MKL_LIBRARIES ${MKL_LIBRARIES} pthread m)
	set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} "-Wl,--no-as-needed")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL DEFAULT_MSG MKL_INCLUDE_DIR MKL_LIBRARIES)

mark_as_advanced(MKL_LIBRARIES MKL_INCLUDE_DIR)
