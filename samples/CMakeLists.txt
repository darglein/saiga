if(CUDA_FOUND)
	add_subdirectory(runTests)
	add_subdirectory(cudaImageProcessing)
	set_target_properties (runTests cudaImageProcessing PROPERTIES FOLDER samples/cuda)
endif ()

if (GLFW_FOUND)
	add_subdirectory(simpleGLFWWindow)
	set_target_properties (simpleGLFWWindow PROPERTIES FOLDER samples)
endif ()

if (SDL2_FOUND)
	add_subdirectory(simpleSDLWindow)
	add_subdirectory(advancedSDLWindow)
	add_subdirectory(cascadedShadowMaps)
	add_subdirectory(lighting)
	add_subdirectory(twoViewOrientation)
	add_subdirectory(volumetricLights)
	set_target_properties (simpleSDLWindow advancedSDLWindow cascadedShadowMaps lighting twoViewOrientation volumetricLights PROPERTIES FOLDER samples)
	if (FFMPEG_FOUND)
		add_subdirectory(videoRecording)
		set_target_properties (runTests cudaImageProcessing PROPERTIES FOLDER samples)
	endif ()
endif ()

if (EGL_FOUND)
	add_subdirectory(offscreenRendering)
	set_target_properties (offscreenRendering PROPERTIES FOLDER samples)
endif ()


if (EIGEN3_FOUND)
	add_subdirectory(eigenTest)
	set_target_properties (eigenTest PROPERTIES FOLDER samples)
endif ()


