# Synopsis:
#   CUDA_SELECT_NVCC_ARCH_FLAGS(out_variable [target_CUDA_architectures])
#   -- Selects GPU arch flags for nvcc based on target_CUDA_architectures
#      target_CUDA_architectures : Auto | All | LIST(ARCH_AND_PTX ...)
#       - "Auto" detects the local machine GPU compute arch at runtime.
#       - "All" covers all architectures the current CUDA toolkit supports.
#      ARCH_AND_PTX : NUM | NUMa | NUMf | NUM+PTX | NUMa+PTX
#      NUM: The architecture number without dot, exactly matching the nvcc
#           flag suffix. Examples: 52, 75, 80, 86, 89, 90, 100, 120
#      Suffixes (optional):
#       - 'a'  architecture-specific variant (e.g. 90a, 100a)
#       - 'f'  family-specific variant (e.g. 100f, CUDA >= 12.9)
#       - +PTX emit PTX (code=compute_XX) instead of a binary (code=sm_XX)
#      Returns LIST of flags to be added to CUDA_NVCC_FLAGS in ${out_variable}
#      Additionally, sets ${out_variable}_readable to the resulting readable list
#      Example:
#       CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 75 86 89+PTX)
#        list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
#
#      More info on CUDA architectures: https://en.wikipedia.org/wiki/CUDA
#      See also the NVCC "GPU Feature List" in the CUDA Compiler Driver docs.

if(CMAKE_CUDA_COMPILER_LOADED) # CUDA as a language
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA"
      AND CMAKE_CUDA_COMPILER_VERSION MATCHES "^([0-9]+\\.[0-9]+)")
    set(CUDA_VERSION "${CMAKE_MATCH_1}")
  endif()
endif()

# Version-gated list of all supported (base) architectures per toolkit release.
# Gated at the exact CUDA releases that introduced the architecture:
#   11.8: 89 (Ada), 90 (Hopper)      12.0: 87
#   12.8: 100, 101, 120 (Blackwell)  12.9: 103, 121 (Blackwell)
#   13.0: 88, 110, 107 (Rubin); 101 was dropped again
if(CUDA_VERSION VERSION_GREATER_EQUAL "13.0")
  set(CUDA_ALL_GPU_ARCHITECTURES
      "50" "52" "53" "60" "61" "62" "70" "72" "75"
      "80" "86" "87" "88" "89" "90"
      "100" "103" "107" "110" "120" "121")
elseif(CUDA_VERSION VERSION_GREATER_EQUAL "12.9")
  set(CUDA_ALL_GPU_ARCHITECTURES
      "50" "52" "53" "60" "61" "62" "70" "72" "75"
      "80" "86" "87" "89" "90"
      "100" "101" "103" "120" "121")
elseif(CUDA_VERSION VERSION_GREATER_EQUAL "12.8")
  set(CUDA_ALL_GPU_ARCHITECTURES
      "50" "52" "53" "60" "61" "62" "70" "72" "75"
      "80" "86" "87" "89" "90"
      "100" "101" "120")
elseif(CUDA_VERSION VERSION_GREATER_EQUAL "12.0")
  set(CUDA_ALL_GPU_ARCHITECTURES
      "50" "52" "53" "60" "61" "62" "70" "72" "75"
      "80" "86" "87" "89" "90")
elseif(CUDA_VERSION VERSION_GREATER_EQUAL "11.8")
  set(CUDA_ALL_GPU_ARCHITECTURES
      "50" "52" "53" "60" "61" "62" "70" "72" "75"
      "80" "86" "89" "90")
elseif(CUDA_VERSION VERSION_GREATER_EQUAL "11.1")
  set(CUDA_ALL_GPU_ARCHITECTURES
      "50" "52" "53" "60" "61" "62" "70" "72" "75" "80" "86")
elseif(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
  set(CUDA_ALL_GPU_ARCHITECTURES
      "50" "52" "53" "60" "61" "62" "70" "72" "75" "80")
else()
  set(CUDA_ALL_GPU_ARCHITECTURES
      "35" "50" "52" "53" "60" "61" "62" "70" "72" "75")
endif()
list(GET CUDA_ALL_GPU_ARCHITECTURES -1 CUDA_MAX_GPU_ARCHITECTURE)

# Check with: cmake -DCUDA_VERSION=12.8 -P select_compute_arch.cmake
if(DEFINED CMAKE_SCRIPT_MODE_FILE)
  include(CMakePrintHelpers)
  cmake_print_variables(CUDA_VERSION)
  cmake_print_variables(CUDA_ALL_GPU_ARCHITECTURES)
  cmake_print_variables(CUDA_MAX_GPU_ARCHITECTURE)
endif()


################################################################################################
# A function for automatic detection of GPUs installed  (if autodetection is enabled)
# Usage:
#   CUDA_DETECT_INSTALLED_GPUS(OUT_VARIABLE)
#
function(CUDA_DETECT_INSTALLED_GPUS OUT_VARIABLE)
  # V2: dotless output; invalidates stale "12.0"-style cache entries from the
  # legacy (dotted) detection program in existing build directories.
  if(NOT CUDA_GPU_DETECT_OUTPUT_V2)
    if(CMAKE_CUDA_COMPILER_LOADED) # CUDA as a language
      set(file "${PROJECT_BINARY_DIR}/detect_cuda_compute_capabilities.cu")
    else()
      set(file "${PROJECT_BINARY_DIR}/detect_cuda_compute_capabilities.cpp")
    endif()

    file(WRITE ${file} ""
      "#include <cuda_runtime.h>\n"
      "#include <cstdio>\n"
      "int main()\n"
      "{\n"
      "  int count = 0;\n"
      "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
      "  if (count == 0) return -1;\n"
      "  for (int device = 0; device < count; ++device)\n"
      "  {\n"
      "    cudaDeviceProp prop;\n"
      "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
      "      std::printf(\"%d%d \", prop.major, prop.minor);\n"
      "  }\n"
      "  return 0;\n"
      "}\n")

    if(CMAKE_CUDA_COMPILER_LOADED) # CUDA as a language
      try_run(run_result compile_result SOURCES ${file}
              RUN_OUTPUT_VARIABLE compute_capabilities)
    else()
      try_run(run_result compile_result SOURCES ${file}
              CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CUDA_INCLUDE_DIRS}"
              LINK_LIBRARIES ${CUDA_LIBRARIES}
              RUN_OUTPUT_VARIABLE compute_capabilities)
    endif()

    # Filter unrelated content out of the output (arch numbers are dotless, e.g. 86, 120).
    string(REGEX MATCHALL "[0-9]+" compute_capabilities "${compute_capabilities}")

    if(run_result EQUAL 0)
      set(CUDA_GPU_DETECT_OUTPUT_V2 ${compute_capabilities}
        CACHE INTERNAL "Returned GPU architectures from detect_gpus tool" FORCE)
    endif()
  endif()

  if(NOT CUDA_GPU_DETECT_OUTPUT_V2)
    message(WARNING "Automatic GPU detection failed. Building for all supported architectures.")
    set(${OUT_VARIABLE} ${CUDA_ALL_GPU_ARCHITECTURES} PARENT_SCOPE)
  else()
    # Keep only archs the current toolkit actually supports. Anything newer is
    # replaced by the highest supported arch with PTX, so it still runs via JIT.
    set(CUDA_GPU_DETECT_OUTPUT_FILTERED "")
    set(_detect_archs "${CUDA_GPU_DETECT_OUTPUT_V2}")
    separate_arguments(_detect_archs)
    foreach(ITEM IN ITEMS ${_detect_archs})
      if(ITEM GREATER CUDA_MAX_GPU_ARCHITECTURE)
        list(APPEND CUDA_GPU_DETECT_OUTPUT_FILTERED "${CUDA_MAX_GPU_ARCHITECTURE}+PTX")
      else()
        list(APPEND CUDA_GPU_DETECT_OUTPUT_FILTERED "${ITEM}")
      endif()
    endforeach()

    set(${OUT_VARIABLE} ${CUDA_GPU_DETECT_OUTPUT_FILTERED} PARENT_SCOPE)
  endif()
endfunction()


################################################################################################
# Function for selecting GPU arch flags for nvcc based on CUDA architectures from parameter list
# Usage:
#   SELECT_NVCC_ARCH_FLAGS(out_variable [list of CUDA compute archs])
function(CUDA_SELECT_NVCC_ARCH_FLAGS out_variable)
  set(CUDA_ARCH_LIST "${ARGN}")

  if("X${CUDA_ARCH_LIST}" STREQUAL "X" )
    set(CUDA_ARCH_LIST "Auto")
  endif()

  if("${CUDA_ARCH_LIST}" STREQUAL "All")
    set(CUDA_ARCH_LIST ${CUDA_ALL_GPU_ARCHITECTURES})
  elseif("${CUDA_ARCH_LIST}" STREQUAL "Auto")
    CUDA_DETECT_INSTALLED_GPUS(CUDA_ARCH_LIST)
    message(STATUS "Autodetected CUDA architecture(s): ${CUDA_ARCH_LIST}")
  endif()

  # Now process the list.
  string(REGEX REPLACE "[ \t]+" ";" CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")
  list(REMOVE_DUPLICATES CUDA_ARCH_LIST)

  set(cuda_arch_bin)
  set(cuda_arch_ptx)

  foreach(arch_name ${CUDA_ARCH_LIST})
    # Accepted tokens: NUM, NUMa, NUMf, NUM+PTX, NUMa+PTX (NUM is 2-3 digits, no dot).
    if(arch_name MATCHES "^([0-9][0-9][0-9]?)([af])?(\\+PTX)?$")
      set(arch_bin ${CMAKE_MATCH_1}${CMAKE_MATCH_2})
      set(add_ptx FALSE)
      if(CMAKE_MATCH_3 STREQUAL "+PTX")
        set(add_ptx TRUE)
      endif()
    else()
      message(SEND_ERROR "Invalid CUDA architecture '${arch_name}'. Expected a dotless number matching the nvcc arch flag, e.g. 75, 86, 90a, 100f, 120, 89+PTX (arch names like 'Ampere' are not supported).")
      set(cuda_arch_bin "")
      set(cuda_arch_ptx "")
      return()
    endif()
    list(APPEND cuda_arch_bin ${arch_bin})
    if(add_ptx)
      list(APPEND cuda_arch_ptx ${arch_bin})
    endif()
  endforeach()

  if(cuda_arch_bin)
    list(REMOVE_DUPLICATES cuda_arch_bin)
  endif()
  if(cuda_arch_ptx)
    list(REMOVE_DUPLICATES cuda_arch_ptx)
  endif()

  set(nvcc_flags "")
  set(nvcc_archs_readable "")

  # Tell NVCC to add binaries for the specified GPUs
  foreach(arch ${cuda_arch_bin})
    list(APPEND nvcc_flags -gencode arch=compute_${arch},code=sm_${arch})
    list(APPEND nvcc_archs_readable sm_${arch})
  endforeach()

  # Tell NVCC to add PTX intermediate code for the specified architectures
  foreach(arch ${cuda_arch_ptx})
    list(APPEND nvcc_flags -gencode arch=compute_${arch},code=compute_${arch})
    list(APPEND nvcc_archs_readable compute_${arch})
  endforeach()

  string(REPLACE ";" " " nvcc_archs_readable "${nvcc_archs_readable}")
  set(${out_variable}          ${nvcc_flags}          PARENT_SCOPE)
  set(${out_variable}_readable ${nvcc_archs_readable} PARENT_SCOPE)
endfunction()
