add_subdirectory(imageProcessing)

if(OpenCV_FOUND)
	add_subdirectory(transformCompareOpenCV)
endif()
