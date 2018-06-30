/*
* Vulkan Example - imGui (https://github.com/ocornut/imgui)
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <array>
#include <algorithm>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gli/gli.hpp>

#include <saiga/imgui/imgui.h>

#include <vulkan/vulkan.h>
#include "vulkanexamplebase.h"
#include "VulkanDevice.hpp"
#include "VulkanBuffer.hpp"
#include "VulkanModel.hpp"
#include "VulkanImgui.h"

#define ENABLE_VALIDATION true

// ----------------------------------------------------------------------------
// VulkanExample
// ----------------------------------------------------------------------------

class VulkanExample : public VulkanExampleBase
{
public:
    std::shared_ptr<ImGUI> imGui;

	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_NORMAL,
		vks::VERTEX_COMPONENT_COLOR,
	});

	struct Models {
		vks::Model models;
		vks::Model logos;
		vks::Model background;
	} models;

	vks::Buffer uniformBufferVS;

	struct UBOVS {
		glm::mat4 projection;
		glm::mat4 modelview;
		glm::vec4 lightPos;
	} uboVS;

	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSet descriptorSet;

    VulkanExample();

    ~VulkanExample();
	
    void buildCommandBuffers();

    void setupLayoutsAndDescriptors();

    void preparePipelines();

	// Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers();

    void updateUniformBuffers();

    void draw();

    void loadAssets();

    void prepareImGui();

    void prepare();

    virtual void render();

	virtual void viewChanged()
	{
		updateUniformBuffers();
	}

	virtual void mouseMoved(double x, double y, bool &handled)
	{
		ImGuiIO& io = ImGui::GetIO();	
		handled = io.WantCaptureMouse;
	}

};

