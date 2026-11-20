




if (NOT GTEST_FOUND)
    message(STATUS "GTest not found. Tests disabled.")
    return()
else ()
endif ()

function(saiga_test FILE_NAME)
    string(REGEX REPLACE "\\.[^.]*$" "" NAME ${FILE_NAME})

    set(TARGET_NAME ${NAME})


    # cuda architecture
    #cmake_policy(SET CMP0104 OLD)
    add_executable(${TARGET_NAME} ${FILE_NAME})
    message(STATUS "Test enabled:      ${NAME}")


    target_link_libraries(${TARGET_NAME} PUBLIC saiga_core ${GTEST_TARGETS})
    # gtest is a shared library (submodules/CMakeLists.txt): the tests import its symbols, e.g. the GTEST_FLAG variables
    # on Windows (not set on gtest itself, gtest_main would then import its own main)
    target_compile_definitions(${TARGET_NAME} PRIVATE GTEST_LINKED_AS_SHARED_LIBRARY=1)
    target_link_libraries(${TARGET_NAME} PUBLIC ${ARGN})

    add_test(
            NAME ${TARGET_NAME}
            COMMAND ${TARGET_NAME}
    )
    # A binary whose tests all skipped (e.g. the Vulkan tests without a driver) is reported as skipped, not passed.
    # The pattern is gtest's summary when no test ran and none failed: "[  PASSED  ] 0 tests." followed only by the
    # list of skipped tests up to the end of the output (a failure list after it does not match, so failures stay
    # failures; binaries where only some tests skip count as passed)
    set_tests_properties(${TARGET_NAME} PROPERTIES SKIP_REGULAR_EXPRESSION
            "\\[  PASSED  \\] 0 tests\\.\n\\[  SKIPPED \\] [0-9]+ tests?, listed below:\n(\\[  SKIPPED \\] [^\n]+\n?)+$")


    #set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES ${SAIGA_CUDA_ARCH})
    target_compile_options(${TARGET_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${SAIGA_CUDA_FLAGS}>)

    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER tests/${PREFIX})
endfunction()


if (MODULE_CORE)
    saiga_test(test_core_align.cpp)
    if (NOT SAIGA_WITH_TINY_EIGEN)
        saiga_test(test_core_normal_packing.cpp)
        saiga_test(test_core_clusterer.cpp)
        saiga_test(test_core_math.cpp)
        saiga_test(test_core_vectorization.cpp)
        saiga_test(test_vision_derivative_chain_rule.cpp)
    else ()
        saiga_test(test_core_tiny_eigen.cpp)
    endif ()
    saiga_test(test_core_frustum.cpp)
    saiga_test(test_core_kdtree.cpp)
    saiga_test(test_core_math_functions.cpp)
    saiga_test(test_core_file_io.cpp)
    saiga_test(test_core_ini.cpp)
    saiga_test(test_core_image_decode.cpp)
    saiga_test(test_core_image_view.cpp)
    saiga_test(test_core_thread_util.cpp)
    saiga_test(test_core_util.cpp)
    saiga_test(test_core_mcp_server.cpp)
    saiga_test(test_core_shader_parser.cpp)
    saiga_test(test_core_tiny_exif.cpp)
    if (SAIGA_USE_ZLIB)
        saiga_test(test_core_zlib.cpp)
    endif ()
    saiga_test(test_core_progressbar.cpp)
    saiga_test(test_core_rectangular_decomposition.cpp)
    saiga_test(test_core_plane_intersecting_circle.cpp)
    saiga_test(test_core_imgui_cursors.cpp)
    saiga_test(test_core_geometry.cpp)
    saiga_test(test_core_model.cpp)

    if (OpenCV_FOUND AND MODULE_EXTRA)
        saiga_test(test_core_image_load_store.cpp ${EXTRA_LIBS})
    else ()
        saiga_test(test_core_image_load_store.cpp)
    endif ()
endif ()
if (MODULE_VISION)
    if (CERES_FOUND)
        saiga_test(test_vision_bundle_adjustment.cpp "saiga_vision")
        saiga_test(test_vision_bundle_adjustment_relpose.cpp "saiga_vision")
        saiga_test(test_vision_pose_graph_optimization.cpp "saiga_vision")
        saiga_test(test_vision_pose_alignment.cpp "saiga_vision")
        saiga_test(test_vision_imu_decoupled_solver.cpp "saiga_vision")
        saiga_test(test_vision_pose_estimation.cpp "saiga_vision")
    endif ()
    saiga_test(test_vision_bow.cpp "saiga_vision")
    saiga_test(test_vision_nearest_neighbor.cpp "saiga_vision")
    saiga_test(test_vision_poisson_surface.cpp "saiga_vision")
    saiga_test(test_vision_distortion.cpp "saiga_vision")
    saiga_test(test_vision_motion_model.cpp "saiga_vision")
    saiga_test(test_vision_derivative_ba.cpp "saiga_vision")
    saiga_test(test_vision_derivative_relpose.cpp "saiga_vision")
    saiga_test(test_vision_derivative_numeric.cpp "saiga_vision")
    saiga_test(test_vision_derivative_ocam.cpp "saiga_vision")
    saiga_test(test_vision_derivative_dualquat.cpp "saiga_vision")
    saiga_test(test_vision_sophus.cpp "saiga_vision")
    saiga_test(test_vision_two_view_reconstruction.cpp "saiga_vision")
    saiga_test(test_vision_feature_grid.cpp "saiga_vision")
    saiga_test(test_vision_five_eight_point.cpp "saiga_vision")
    saiga_test(test_vision_imu.cpp "saiga_vision")
    saiga_test(test_vision_imu_derivatives.cpp "saiga_vision")
    saiga_test(test_vision_robust_cost_function.cpp "saiga_vision")
    saiga_test(test_vision_tsdf.cpp "saiga_vision")
    saiga_test(test_vision_tsdf_fuse.cpp "saiga_vision")
    saiga_test(test_vision_recursive_linear_systems.cpp "saiga_vision")
    if (K4A_FOUND)
        saiga_test(test_vision_azure.cpp "saiga_vision")
    endif ()
endif ()

if (MODULE_OPENGL AND EGL_FOUND)
    # Headless: an EGL context without a window (see opengl_test_context.h)
    saiga_test(test_opengl_context.cpp "saiga_opengl")
    saiga_test(test_opengl_texture.cpp "saiga_opengl")
    saiga_test(test_opengl_framebuffer.cpp "saiga_opengl")
    saiga_test(test_opengl_buffer.cpp "saiga_opengl")
    saiga_test(test_opengl_shader.cpp "saiga_opengl")
    saiga_test(test_opengl_query.cpp "saiga_opengl")
    # A StandaloneWindow with the headless backend (OffscreenWindow with ImGui)
    saiga_test(test_opengl_window.cpp "saiga_opengl")
endif ()

if (MODULE_VULKAN)
    # Headless: a HeadlessWindow without a window system or swap chain (see vulkan_test_context.h)
    saiga_test(test_vulkan_device.cpp "saiga_vulkan")
    saiga_test(test_vulkan_buffer.cpp "saiga_vulkan")
    saiga_test(test_vulkan_texture.cpp "saiga_vulkan")
    saiga_test(test_vulkan_shader.cpp "saiga_vulkan")
    saiga_test(test_vulkan_compute.cpp "saiga_vulkan")
    saiga_test(test_vulkan_headless_renderer.cpp "saiga_vulkan")
    saiga_test(test_vulkan_sampler.cpp "saiga_vulkan")
    saiga_test(test_vulkan_render_target.cpp "saiga_vulkan")
    saiga_test(test_vulkan_draw.cpp "saiga_vulkan")
    saiga_test(test_vulkan_memory.cpp "saiga_vulkan")
    saiga_test(test_vulkan_world.cpp "saiga_vulkan")
    saiga_test(test_vulkan_standalone_window.cpp "saiga_vulkan")
endif ()

if (MODULE_ENGINE)
    # Engine tests: test_engine_<area>.cpp (docs/engine_design.md §18.1)
    saiga_test(test_engine_smoke.cpp "saiga_engine")
    saiga_test(test_engine_core.cpp "saiga_engine")
    saiga_test(test_engine_ecs.cpp "saiga_engine")
    saiga_test(test_engine_ecs_benchmark.cpp "saiga_engine")
    saiga_test(test_engine_world.cpp "saiga_engine")
    # A second translation unit: types of the same name in unnamed namespaces (event type check)
    target_sources(test_engine_world PRIVATE engine_world_second_tu.cpp)
    saiga_test(test_engine_spatial_grid.cpp "saiga_engine")
    saiga_test(test_engine_gameobject.cpp "saiga_engine")
    saiga_test(test_engine_world_api.cpp "saiga_engine")
    saiga_test(test_engine_save.cpp "saiga_engine")
    saiga_test(test_engine_data.cpp "saiga_engine")
    saiga_test(test_engine_scenarios.cpp "saiga_engine")
    saiga_test(test_engine_app.cpp "saiga_engine")

    # Compile-fail tests of the visit field rules (docs/engine_design.md §7.2): case 0 must compile, the others must
    # fail with their static_assert message. The targets are not built by default; ctest builds them one at a time.
    set(ECS_FIELD_RULE_MESSAGES
        "ecs_field_rules_ok"
        "float fields only in components with allowFloat"
        "enum fields need EnumNames"
        "unsupported field type"
        "unsupported field type"
        "only in components with allowFloat"
        "float fields only in components with allowFloat"
        "is not supported \\(use std::vector<uint8_t>\\)"
        "allowFloat only in Render"
        "unsupported field type")
    foreach (FIELD_CASE RANGE 0 9)
        set(FIELD_TARGET compile_fail_ecs_field_${FIELD_CASE})
        add_executable(${FIELD_TARGET} EXCLUDE_FROM_ALL compile_fail/ecs_field_rules.cpp)
        target_link_libraries(${FIELD_TARGET} saiga_engine)
        target_compile_definitions(${FIELD_TARGET} PRIVATE FIELD_CASE=${FIELD_CASE})
        set_target_properties(${FIELD_TARGET} PROPERTIES FOLDER tests/compile_fail)
        list(GET ECS_FIELD_RULE_MESSAGES ${FIELD_CASE} FIELD_MESSAGE)
        if (FIELD_CASE EQUAL 0)
            # The control case must build (exit code 0)
            add_test(NAME ${FIELD_TARGET}
                     COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${FIELD_TARGET} --config $<CONFIG>)
        else ()
            # The build must fail and print the static_assert message
            add_test(NAME ${FIELD_TARGET}
                     COMMAND ${CMAKE_COMMAND} -DBUILD_DIR=${CMAKE_BINARY_DIR} -DTARGET=${FIELD_TARGET}
                             -DCONFIG=$<CONFIG> -DEXPECT=${FIELD_MESSAGE}
                             -P ${CMAKE_CURRENT_SOURCE_DIR}/compile_fail/expect_build_failure.cmake)
        endif ()
        # One build at a time in the build directory
        set_tests_properties(${FIELD_TARGET} PROPERTIES RESOURCE_LOCK cmake_build_dir)
    endforeach ()
endif ()

if (MODULE_CUDA)
    saiga_test(test_cuda_image.cu "saiga_cuda")
    saiga_test(test_cuda_simple.cu "saiga_cuda")
    saiga_test(test_cuda_thrust.cu "saiga_cuda")
    saiga_test(test_cuda_bvh.cu "saiga_cuda")
    saiga_test(test_cuda_algorithms.cu "saiga_cuda")
    saiga_test(test_cuda_image_processing.cu "saiga_cuda")
    if (MODULE_VISION)
        saiga_test(test_cuda_orb.cu "saiga_cuda" "saiga_vision")
    endif ()
    if (MODULE_OPENGL AND EGL_FOUND)
        saiga_test(test_cuda_opengl_interop.cpp "saiga_cuda" "saiga_opengl")
    endif ()
endif ()








