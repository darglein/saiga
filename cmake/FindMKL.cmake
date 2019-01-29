


if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")

    find_path(MKL_INCLUDE_DIR
        mkl/mkl.h
        PATHS
            /usr/local
    )

find_library(MKL_LIBRARIES
  mkl_core
  PATHS
  $ENV{MKLLIB}
  /opt/intel/mkl/*/lib/em64t
  /opt/intel/Compiler/*/*/mkl/lib/em64t
  ${LIB_INSTALL_DIR}
)

find_library(MKL_GUIDE
  guide
  PATHS
  $ENV{MKLLIB}
  /opt/intel/mkl/*/lib/em64t
  /opt/intel/Compiler/*/*/mkl/lib/em64t
  /opt/intel/Compiler/*/*/lib/intel64
  ${LIB_INSTALL_DIR}
)

if(MKL_LIBRARIES AND MKL_GUIDE)
  set(MKL_LIBRARIES ${MKL_LIBRARIES} mkl_intel_lp64 mkl_sequential ${MKL_GUIDE} pthread)
endif()

set(MKL_LIBRARIES ${MKL_LIBRARIES} mkl_intel_lp64 mkl_sequential)

endif()


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL DEFAULT_MSG MKL_INCLUDE_DIR MKL_LIBRARIES)

mark_as_advanced(MKL_LIBRARIES)
