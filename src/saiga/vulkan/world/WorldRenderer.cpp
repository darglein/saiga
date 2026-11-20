/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/world/WorldRenderer.h"

#include "saiga/core/imgui/imgui.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace Saiga
{
namespace Vulkan
{
using Stage  = vk::PipelineStageFlagBits2;
using Access = vk::AccessFlagBits2;

//==================================================================================================
// Settings

namespace
{
template <typename S, typename F>
void forEachSetting(S& s, F f)
{
    f("shading", s.shading);
    f("freeze_time", s.freezeTime);
    f("terrain", s.terrain);
    f("trees", s.trees);
    f("grass", s.grass);
    f("water", s.water);
    f("sky", s.sky);
    f("wireframe", s.wireframe);
    f("debug_mode", s.debugMode);
    f("flat_shading", s.flatShading);
    f("freeze_culling", s.freezeCulling);
    f("fovy", s.fovy);
    f("fog", s.fog);
    f("fog_density", s.fogDensity);
    f("fog_falloff", s.fogFalloff);
    f("sun_azimuth", s.sunAzimuth);
    f("sun_elevation", s.sunElevation);
    f("sun_intensity", s.sunIntensity);
    f("ambient", s.ambient);
    f("albedo", s.albedo);
    f("wind", s.wind);
    f("terrain_lod_range", s.terrainLodRange);
    f("morph_start", s.morphStart);
    f("tree_lod1", s.treeLod1);
    f("tree_lod2", s.treeLod2);
    f("tree_max_distance", s.treeMaxDistance);
    f("grass_radius", s.grassRadius);
    f("grass_cell_size", s.grassCellSize);
    f("grass_density", s.grassDensity);
    f("alpha_cutoff", s.alphaCutoff);
}
}  // namespace

json11::Json WorldSettings::toJson() const
{
    json11::Json::object o;
    forEachSetting(*this,
                   [&](const char* name, const auto& v)
                   {
                       using T = std::decay_t<decltype(v)>;
                       if constexpr (std::is_same_v<T, WorldShading>)
                           o[name] = json11::Json("clay");
                       else
                           o[name] = json11::Json(v);
                   });
    return o;
}

std::vector<std::string> WorldSettings::fromJson(const json11::Json& json)
{
    std::vector<std::string> unknown;
    for (auto& kv : json.object_items())
    {
        bool found = false;
        forEachSetting(*this,
                       [&](const char* name, auto& v)
                       {
                           if (kv.first != name) return;
                           found   = true;
                           using T = std::decay_t<decltype(v)>;
                           if constexpr (std::is_same_v<T, WorldShading>)
                           {
                               // Clay is the only shading mode
                               if (kv.second.string_value() == "clay")
                                   v = WorldShading::Clay;
                               else
                                   found = false;
                           }
                           else if constexpr (std::is_same_v<T, bool>)
                           {
                               // A value of the wrong type is reported like an unknown key, not converted
                               if (kv.second.is_bool())
                                   v = kv.second.bool_value();
                               else
                                   found = false;
                           }
                           else if constexpr (std::is_same_v<T, int>)
                           {
                               // int_value() is undefined outside the int range: bound the number first, clamp()
                               // then limits it to the GUI range (debug_mode 1001 -> 1, not 0)
                               if (kv.second.is_number())
                                   v = int(std::max(-1e6, std::min(kv.second.number_value(), 1e6)));
                               else
                                   found = false;
                           }
                           else
                           {
                               // The float conversion of a double outside the float range is undefined
                               if (kv.second.is_number())
                                   v = float(std::max(-1e30, std::min(kv.second.number_value(), 1e30)));
                               else
                                   found = false;
                           }
                       });
        if (!found) unknown.push_back(kv.first);
    }
    clamp();
    return unknown;
}

void WorldSettings::clamp()
{
    // The ranges of the GUI sliders. Values from MCP are not limited otherwise; e.g. a tiny grass cell size would draw
    // billions of instances (device lost). NaN becomes the lower bound.
    auto c    = [](float& v, float lo, float hi) { v = v >= lo ? std::min(v, hi) : lo; };
    debugMode = std::min(std::max(debugMode, 0), 1);
    c(fovy, 20, 110);
    c(fogDensity, 0, 0.002f);
    c(fogFalloff, 0.0001f, 0.01f);
    c(sunAzimuth, -180, 180);
    c(sunElevation, -10, 90);
    c(sunIntensity, 0, 2);
    c(ambient, 0, 1);
    c(albedo, 0, 1);
    c(wind, 0, 4);
    c(terrainLodRange, 100, 2000);
    c(morphStart, 0.3f, 0.95f);
    c(treeLod1, 10, 500);
    c(treeLod2, 50, 3000);
    c(treeMaxDistance, 500, 50000);
    c(grassRadius, 0, 200);
    c(grassCellSize, 0.1f, 2.0f);
    c(grassDensity, 0, 1);
    c(alphaCutoff, 0.05f, 0.95f);
}

//==================================================================================================

void WorldRenderer::init(VulkanBase& base, const RenderingFormats& formats, const WorldParameters& parameters)
{
    initWith(base, formats, parameters, nullptr);
}

void WorldRenderer::init(VulkanBase& base, const RenderingFormats& formats, const WorldParameters& parameters,
                         std::shared_ptr<const HeightField> heightField)
{
    if (!heightField) SAIGA_EXIT_ERROR("WorldRenderer::init: no height field");
    initWith(base, formats, parameters, std::move(heightField));
}

void WorldRenderer::setOutputLinear(bool linear, float preExposure)
{
    SAIGA_ASSERT(preExposure > 0, "WorldRenderer::setOutputLinear: the pre-exposure must be positive");
    outputLinear      = linear;
    outputPreExposure = preExposure;
}

const HeightField& WorldRenderer::heightField() const
{
    static const HeightField empty;
    return hf ? *hf : empty;
}

void WorldRenderer::initWith(VulkanBase& _base, const RenderingFormats& _formats, const WorldParameters& parameters,
                             std::shared_ptr<const HeightField> external)
{
    destroy();
    base         = &_base;
    formats      = _formats;
    params       = parameters;
    hasCullState = false;
    SAIGA_ASSERT(params.reversedZ, "WorldRenderer: the pass must use reversed depth (WorldParameters::reversedZ)");

    auto t0           = std::chrono::steady_clock::now();
    const bool shared = external != nullptr;
    if (shared)
    {
        hf             = std::move(external);
        params.terrain = hf->parameters();
    }
    else
    {
        auto generated = std::make_shared<HeightField>();
        generated->generate(params.terrain);
        hf = std::move(generated);
    }
    auto t1 = std::chrono::steady_clock::now();

    uniformBuffer.init(*base, nullptr, sizeof(WorldUniforms));

    {
        auto layers = FoliageGenerator::foliageTextures(512, params.seed);
        std::vector<const Image*> images;
        for (auto& l : layers) images.push_back(&l);
        foliageTexture.fromImages(*base, images, vk::ImageUsageFlagBits::eSampled, false, true);
    }

    terrainRenderer.init(*base, formats, uniformBuffer, *hf, 64.0f);
    forestRenderer.init(*base, formats, uniformBuffer, foliageTexture, *hf, params.treeCount, params.seed);
    grassRenderer.init(*base, formats, uniformBuffer, foliageTexture, terrainRenderer);
    auto t2 = std::chrono::steady_clock::now();

    // Sky: a full screen triangle at the far plane (depth 0), only where nothing was drawn
    {
        DescriptorSetLayout layout(
            {{0, {0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment}}});
        PipelineInfo info                       = worldPipelineInfo();
        info.rasterizationState.cullMode        = vk::CullModeFlagBits::eNone;
        info.depthStencilState.depthCompareOp   = vk::CompareOp::eEqual;
        info.depthStencilState.depthWriteEnable = VK_FALSE;
        createWorldPipeline(skyPipeline, *base, layout, 0, {"vulkan/world/sky.vert", "vulkan/world/sky.frag"}, formats,
                            info);
        skySet = skyPipeline.createDescriptorSet();
        skySet.assign(0, &uniformBuffer);
    }

    // GPU timestamps
    {
        auto props          = base->physicalDevice.getProperties();
        auto queueProps     = base->physicalDevice.getQueueFamilyProperties()[base->mainQueue.getQueueFamilyIndex()];
        timestampsSupported = props.limits.timestampComputeAndGraphics && queueProps.timestampValidBits > 0;
        timestampPeriod     = props.limits.timestampPeriod;
        // The slots refer to the queries of the previous pool (a re-init): the new pool's queries are not reset yet
        timestampSlots = {};
        gpuQueue.clear();
        timestampIndex = 0;
        if (timestampsSupported)
        {
            queryPool = base->device.createQueryPool(vk::QueryPoolCreateInfo(
                {}, vk::QueryType::eTimestamp, uint32_t(timestampSlots.size() * timestampsPerFrame)));
        }
        statistics.gpuPassMs.fill(-1);
    }
    cpuHistory.assign(historySize, 0.0f);
    gpuHistory.assign(historySize, 0.0f);
    historyIndex = 0;

    std::cout << "World: terrain " << hf->size() << "^2 (" << hf->worldSize() / 1000.0f << " km) "
              << (shared ? "shared" : "generated") << " in " << std::chrono::duration<double>(t1 - t0).count() << " s, "
              << forestRenderer.treeCount() << " trees, GPU resources in "
              << std::chrono::duration<double>(t2 - t1).count() << " s" << std::endl;
    initialized = true;
}

void WorldRenderer::destroy()
{
    if (!initialized) return;
    terrainRenderer.destroy();
    forestRenderer.destroy();
    grassRenderer.destroy();
    skyPipeline.destroy();
    foliageTexture.destroy();
    uniformBuffer.destroy();
    if (queryPool)
    {
        auto device = base->device;
        auto pool   = queryPool;
        base->deferDelete([device, pool]() { device.destroyQueryPool(pool); });
        queryPool = nullptr;
    }
    hf.reset();  // the shared height field
    initialized = false;
}

mat4 WorldRenderer::projection(float aspect) const
{
    return reversedZInfinitePerspective(radians(settings.fovy), aspect, 0.1f);
}

const char* WorldRenderer::gpuPassName(int pass)
{
    static const char* names[PASS_COUNT] = {"upload + cull", "terrain + water", "trees", "grass", "sky", "wireframe"};
    return pass >= 0 && pass < PASS_COUNT ? names[pass] : "?";
}

void WorldRenderer::smooth(double& average, double value, int frames)
{
    double a = frames < 10 ? 0.5 : 0.05;
    average  = average < 0 ? value : average * (1 - a) + value * a;
}

void WorldRenderer::writeTimestamp(vk::CommandBuffer cmd, uint32_t marker, vk::PipelineStageFlags2 stage)
{
    if (timestampsSupported) cmd.writeTimestamp2(stage, queryPool, currentSlot * timestampsPerFrame + marker);
}

double WorldRenderer::readGpuTimes(std::array<double, PASS_COUNT>& passes, int& frameNumber)
{
    if (!timestampsSupported) return -1;
    uint64_t completed = base->completedFrameValue();
    // All completed slots, oldest first
    std::vector<int> ready;
    for (int i = 0; i < (int)timestampSlots.size(); ++i)
    {
        auto& s = timestampSlots[i];
        if (s.written && s.frameValue <= completed) ready.push_back(i);
    }
    std::sort(ready.begin(), ready.end(),
              [&](int a, int b) { return timestampSlots[a].frameNumber < timestampSlots[b].frameNumber; });
    for (int slot : ready)
    {
        std::array<uint64_t, timestampsPerFrame> ts;
        auto result =
            base->device.getQueryPoolResults(queryPool, uint32_t(slot * timestampsPerFrame), timestampsPerFrame,
                                             sizeof(ts), ts.data(), sizeof(uint64_t), vk::QueryResultFlagBits::e64);
        timestampSlots[slot].written = false;
        if (result != vk::Result::eSuccess) continue;
        GpuSample g;
        g.frame = timestampSlots[slot].frameNumber;
        for (int p = 0; p < PASS_COUNT; ++p) g.passes[p] = double(ts[p + 1] - ts[p]) * timestampPeriod * 1e-6;
        g.total = double(ts[PASS_COUNT] - ts[0]) * timestampPeriod * 1e-6;
        gpuQueue.push_back(g);
    }
    if (gpuQueue.empty()) return -1;
    GpuSample g = gpuQueue.front();
    gpuQueue.pop_front();
    passes      = g.passes;
    frameNumber = g.frame;
    return g.total;
}

void WorldRenderer::transfer(vk::CommandBuffer cmd, const mat4& view, const vec3& cameraPosition, float aspect)
{
    SAIGA_ASSERT(initialized);
    auto transferStart = std::chrono::steady_clock::now();
    // Without line rasterization there are no wireframe pipelines (get_state reports it as off)
    if (!wireframeSupported(*base)) settings.wireframe = false;

    // Frame timing
    auto now     = std::chrono::steady_clock::now();
    double cpuMs = 0;
    if (hasLastFrame)
    {
        cpuMs                 = std::chrono::duration<double, std::milli>(now - lastFrame).count();
        time                  = settings.freezeTime ? 0.0 : time + cpuMs * 0.001;
        float a               = statistics.frames < 10 ? 0.5f : 0.05f;
        statistics.cpuFrameMs = statistics.cpuFrameMs * (1 - a) + cpuMs * a;
        statistics.fps        = statistics.cpuFrameMs > 0 ? 1000.0 / statistics.cpuFrameMs : 0;
    }
    lastFrame    = now;
    hasLastFrame = true;
    statistics.frames++;

    std::array<double, PASS_COUNT> passMs;
    int gpuFrame = 0;
    double gpuMs = readGpuTimes(passMs, gpuFrame);
    if (gpuMs >= 0)
    {
        smooth(statistics.gpuMs, gpuMs, statistics.frames);
        for (int p = 0; p < PASS_COUNT; ++p) smooth(statistics.gpuPassMs[p], passMs[p], statistics.frames);
        lastGpuMs = gpuMs;
    }
    if (cpuMs > 0)
    {
        cpuHistory[historyIndex] = float(cpuMs);
        gpuHistory[historyIndex] = float(std::max(lastGpuMs, 0.0));
        historyIndex             = (historyIndex + 1) % historySize;
    }
    lastTimes           = FrameTimes();
    lastTimes.cpuMs     = cpuMs > 0 ? cpuMs : -1;
    lastTimes.cpuFrame  = statistics.frames - 1;
    lastTimes.cpuWorkMs = frameWorkMs;
    frameWorkMs         = -1;
    if (gpuMs >= 0)
    {
        lastTimes.gpuMs     = gpuMs;
        lastTimes.gpuPassMs = passMs;
        lastTimes.gpuFrame  = gpuFrame;
    }
    if (hasLastFrame && cpuMs > 0)
    {
        for (size_t i = 0; i < frameCallbacks.size();)
        {
            if (frameCallbacks[i](lastTimes))
                frameCallbacks.erase(frameCallbacks.begin() + i);
            else
                ++i;
        }
    }

    if (timestampsSupported)
    {
        currentSlot    = timestampIndex;
        timestampIndex = (timestampIndex + 1) % timestampSlots.size();
        cmd.resetQueryPool(queryPool, currentSlot * timestampsPerFrame, timestampsPerFrame);
        writeTimestamp(cmd, 0, Stage::eTopOfPipe);
        timestampSlots[currentSlot].frameValue  = base->currentFrameSignalValue();
        timestampSlots[currentSlot].frameNumber = statistics.frames;
        timestampSlots[currentSlot].written     = true;
    }

    mat4 proj     = projection(aspect);
    mat4 viewProj = proj * view;
    // Frozen culling keeps the last state; it is captured once if culling was frozen before the first frame
    if (!settings.freezeCulling || !hasCullState)
    {
        cullPosition = cameraPosition;
        cullFrustum  = frustumPlanes(viewProj);
        hasCullState = true;
    }

    // Terrain node selection (records the instance upload)
    auto t0 = std::chrono::steady_clock::now();
    terrainRenderer.update(cmd, cullPosition, cullFrustum, settings.terrainLodRange, settings.morphStart,
                           settings.water);
    auto t1 = std::chrono::steady_clock::now();
    grassRenderer.update(cullPosition, cullFrustum, settings.grass ? settings.grassRadius : 0, settings.grassCellSize);
    auto t2 = std::chrono::steady_clock::now();
    smooth(statistics.cpuTerrainSelectMs, std::chrono::duration<double, std::milli>(t1 - t0).count(),
           statistics.frames);
    smooth(statistics.cpuGrassCullMs, std::chrono::duration<double, std::milli>(t2 - t1).count(), statistics.frames);

    // Uniforms
    {
        float az = radians(settings.sunAzimuth), el = radians(settings.sunElevation);
        vec3 sun(std::cos(el) * std::sin(az), std::sin(el), -std::cos(el) * std::cos(az));

        // The view without the translation: the camera is at the origin of the relative positions
        mat4 viewRel            = view;
        viewRel(0, 3)           = 0;
        viewRel(1, 3)           = 0;
        viewRel(2, 3)           = 0;
        mat4 viewProjRel        = proj * viewRel;
        uniforms.viewProjRel    = viewProjRel;
        uniforms.invViewProjRel = inverse(viewProjRel);
        uniforms.cameraPos      = vec4(cameraPosition(0), cameraPosition(1), cameraPosition(2), float(time));
        uniforms.lodCameraPos   = vec4(cullPosition(0), cullPosition(1), cullPosition(2), 0);
        uniforms.sunDir         = vec4(sun(0), sun(1), sun(2), settings.sunIntensity);
        // w: 0 display values (clamped, encoded), > 0 linear values times the pre-exposure (setOutputLinear)
        uniforms.shading  = vec4(settings.albedo, settings.ambient, settings.flatShading ? 1 : 0,
                                outputLinear ? outputPreExposure : 0.0f);
        uniforms.fog      = vec4(settings.fogDensity, settings.fogFalloff, settings.fog ? 1 : 0, hf->waterLevel());
        uniforms.fogColor = vec4(0.42f, 0.42f, 0.42f, 1);
        uniforms.skyColor = vec4(0.2f, 0.2f, 0.2f, 1);
        for (int i = 0; i < 6; ++i) uniforms.frustum[i] = cullFrustum[i];
        uniforms.terrain = vec4(float(hf->size()), hf->spacing(), terrainRenderer.range0(), settings.morphStart);
        uniforms.terrain2 =
            vec4(float(terrainRenderer.maxLevel()), float(settings.debugMode), settings.wind, settings.alphaCutoff);
        uniforms.trees = vec4(settings.treeLod1, settings.treeLod2, settings.trees ? settings.treeMaxDistance : 0,
                              float(forestRenderer.treeCount()));
        uniforms.grass =
            vec4(settings.grassRadius, settings.grassCellSize, settings.grassDensity, float(int(formats.samples)));
        uniforms.viewport = vec4(aspect, 1, 1.0f / aspect, 1);

        // The culling of the previous frame read the uniforms
        vk::MemoryBarrier2 war(Stage::eComputeShader, {}, Stage::eAllTransfer, {});
        cmd.pipelineBarrier2(vk::DependencyInfo({}, war, nullptr, nullptr));
        uniformBuffer.update(cmd, sizeof(WorldUniforms), &uniforms);
        vk::MemoryBarrier2 raw(Stage::eAllTransfer, Access::eTransferWrite, Stage::eComputeShader,
                               Access::eUniformRead);
        cmd.pipelineBarrier2(vk::DependencyInfo({}, raw, nullptr, nullptr));
    }

    forestRenderer.cull(cmd);
    writeTimestamp(cmd, PASS_UPLOAD_CULL + 1, Stage::eAllCommands);

    // Statistics
    auto& ts                    = terrainRenderer.stats();
    statistics.terrainNodes     = ts.nodes + ts.quarterNodes;
    statistics.terrainTriangles = settings.terrain ? ts.triangles : 0;
    statistics.treesVisible     = forestRenderer.stats().visible;
    statistics.treeTriangles    = forestRenderer.stats().triangles;
    statistics.grassClumps      = grassRenderer.stats().clumps;
    statistics.grassTriangles   = grassRenderer.stats().triangles;
    transferWorkMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - transferStart).count();
    smooth(statistics.cpuTransferMs, transferWorkMs, statistics.frames);
}

void WorldRenderer::render(vk::CommandBuffer cmd)
{
    auto start = std::chrono::steady_clock::now();
    int calls  = 0;
    auto mark  = [&](int pass) { writeTimestamp(cmd, uint32_t(pass + 1), Stage::eAllCommands); };
    if (settings.terrain || settings.water)
    {
        // Without terrain only the water plane
        terrainRenderer.render(cmd, false, settings.terrain);
        calls += terrainRenderer.drawCalls(settings.terrain);
    }
    mark(PASS_TERRAIN);
    if (settings.trees)
    {
        forestRenderer.render(cmd, false);
        calls += forestRenderer.drawCalls();
    }
    mark(PASS_TREES);
    if (settings.grass)
    {
        grassRenderer.render(cmd, false);
        calls += grassRenderer.drawCalls();
    }
    mark(PASS_GRASS);
    if (settings.sky && skyPipeline.bind(cmd))
    {
        skyPipeline.bindDescriptorSet(cmd, skySet);
        cmd.draw(3, 1, 0, 0);
        calls += 1;
    }
    mark(PASS_SKY);
    if (settings.wireframe && wireframeSupported(*base))
    {
        if (settings.terrain) terrainRenderer.render(cmd, true);
        if (settings.trees) forestRenderer.render(cmd, true);
        if (settings.grass) grassRenderer.render(cmd, true);
        calls += (settings.terrain ? terrainRenderer.drawCalls() : 0) +
                 (settings.trees ? forestRenderer.drawCalls() : 0) + (settings.grass ? grassRenderer.drawCalls() : 0);
    }
    mark(PASS_WIREFRAME);

    statistics.drawCalls = calls;
    double renderMs      = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    smooth(statistics.cpuRenderMs, renderMs, statistics.frames);
    // Delivered with the frame interval of this frame by the next transfer()
    frameWorkMs = transferWorkMs + renderMs;
}

std::vector<WorldRenderer::MemoryEntry> WorldRenderer::memoryUsage() const
{
    // R8 array with a full mip chain (~4/3 of level 0)
    size_t foliage = size_t(foliageTexture.width) * foliageTexture.height * foliageTexture.layerCount * 4 / 3;
    return {{"terrain (height map, grids, nodes)", terrainRenderer.memoryBytes()},
            {"trees (meshes, instances, culling)", forestRenderer.memoryBytes()},
            {"grass (clump mesh)", grassRenderer.memoryBytes()},
            {"foliage alpha texture", foliage},
            {"uniforms", uniformBuffer.size()}};
}

json11::Json WorldRenderer::statsJson() const
{
    auto& s = statistics;
    json11::Json::object gpuPasses;
    for (int p = 0; p < PASS_COUNT; ++p) gpuPasses[gpuPassName(p)] = s.gpuPassMs[p];
    size_t memory = 0;
    for (auto& m : memoryUsage()) memory += m.bytes;
    return json11::Json::object{
        {"fps", s.fps},
        {"cpu_frame_ms", s.cpuFrameMs},
        {"gpu_world_ms", s.gpuMs},
        {"terrain_nodes", s.terrainNodes},
        {"terrain_triangles", s.terrainTriangles},
        {"trees_total", forestRenderer.treeCount()},
        {"trees_visible_lod0", s.treesVisible[0]},
        {"trees_visible_lod1", s.treesVisible[1]},
        {"trees_visible_lod2", s.treesVisible[2]},
        {"tree_triangles", s.treeTriangles},
        {"grass_clumps_submitted", s.grassClumps},
        {"grass_triangles_submitted", s.grassTriangles},
        {"msaa_samples", int(formats.samples)},
        {"draw_calls", s.drawCalls},
        {"gpu_pass_ms", gpuPasses},
        {"cpu_ms", json11::Json::object{{"terrain_select", s.cpuTerrainSelectMs},
                                        {"grass_cull", s.cpuGrassCullMs},
                                        {"transfer", s.cpuTransferMs},
                                        {"render", s.cpuRenderMs}}},
        {"gpu_memory_mb", double(memory) / (1024.0 * 1024.0)},
    };
}

void WorldRenderer::imgui()
{
    if (!ImGui::Begin("World"))
    {
        ImGui::End();
        return;
    }
    auto& s = statistics;
    ImGui::Text("%.1f fps  cpu %.2f ms  gpu(world) %.2f ms", s.fps, s.cpuFrameMs, s.gpuMs);
    ImGui::Text("terrain: %d nodes, %.2f M tris", s.terrainNodes, s.terrainTriangles / 1e6);
    ImGui::Text("trees: %d / %d / %d of %d, %.2f M tris", s.treesVisible[0], s.treesVisible[1], s.treesVisible[2],
                forestRenderer.treeCount(), s.treeTriangles / 1e6);
    ImGui::Text("grass: %d clumps, %.2f M tris (submitted)", s.grassClumps, s.grassTriangles / 1e6);

    ImGui::Checkbox("statistics window", &showStatistics);

    auto& st = settings;
    if (ImGui::CollapsingHeader("Passes", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("terrain", &st.terrain);
        ImGui::SameLine();
        ImGui::Checkbox("trees", &st.trees);
        ImGui::SameLine();
        ImGui::Checkbox("grass", &st.grass);
        ImGui::Checkbox("water", &st.water);
        ImGui::SameLine();
        ImGui::Checkbox("sky", &st.sky);
        ImGui::SameLine();
        ImGui::Checkbox("fog", &st.fog);
        if (wireframeSupported(*base))
            ImGui::Checkbox("wireframe", &st.wireframe);
        else
            ImGui::TextDisabled("wireframe (not supported)");
        ImGui::SameLine();
        ImGui::Checkbox("flat shading", &st.flatShading);
        ImGui::Combo("debug", &st.debugMode, "none\0lod tint\0\0");
        ImGui::Checkbox("freeze culling", &st.freezeCulling);
        ImGui::SameLine();
        ImGui::Checkbox("freeze time", &st.freezeTime);
    }
    if (ImGui::CollapsingHeader("Level of detail"))
    {
        ImGui::SliderFloat("terrain lod range", &st.terrainLodRange, 100, 2000, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("morph start", &st.morphStart, 0.3f, 0.95f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("tree lod1", &st.treeLod1, 10, 500, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("tree lod2", &st.treeLod2, 50, 3000, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("tree max distance", &st.treeMaxDistance, 500, 50000, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("grass radius", &st.grassRadius, 0, 200, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("grass cell size", &st.grassCellSize, 0.1f, 2.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("grass density", &st.grassDensity, 0, 1, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("alpha cutoff", &st.alphaCutoff, 0.05f, 0.95f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
    }
    if (ImGui::CollapsingHeader("Lighting"))
    {
        ImGui::SliderFloat("fovy", &st.fovy, 20, 110, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("sun azimuth", &st.sunAzimuth, -180, 180, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("sun elevation", &st.sunElevation, -10, 90, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("sun intensity", &st.sunIntensity, 0, 2, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("ambient", &st.ambient, 0, 1, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("albedo", &st.albedo, 0, 1, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("wind", &st.wind, 0, 4, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("fog density", &st.fogDensity, 0, 0.002f, "%.5f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("fog falloff", &st.fogFalloff, 0.0001f, 0.01f, "%.5f", ImGuiSliderFlags_AlwaysClamp);
    }
    ImGui::End();

    if (showStatistics) imguiStatistics();
}

}  // namespace Vulkan
}  // namespace Saiga
