if(CUDA_FOUND)
	add_subdirectory(runTests)
	add_subdirectory(cudaImageProcessing)
endif ()

if (GLFW_FOUND)
	add_subdirectory(simpleGLFWWindow)
endif ()

if (SDL2_FOUND)
	add_subdirectory(simpleSDLWindow)
	add_subdirectory(advancedSDLWindow)
	add_subdirectory(cascadedShadowMaps)
	add_subdirectory(lighting)
	add_subdirectory(twoViewOrientation)
	add_subdirectory(volumetricLights)
	if (FFMPEG_FOUND)
		add_subdirectory(videoRecording)
	endif ()
endif ()

if (EGL_FOUND)
	add_subdirectory(offscreenRendering)
endif ()


if (EIGEN3_FOUND)
	add_subdirectory(eigenTest)
endif ()
