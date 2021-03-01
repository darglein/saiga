/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */



#include "saiga/core/image/imageTransformations.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/imath.h"
#include "saiga/core/sdl/sdl_camera.h"
#include "saiga/core/sdl/sdl_eventhandler.h"
#include "saiga/core/time/timer.h"
#include "saiga/core/util/color.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/vulkan/VulkanForwardRenderer.h"
#include "saiga/vulkan/memory/BufferChunkAllocator.h"
#include "saiga/vulkan/memory/VulkanMemory.h"
#include "saiga/vulkan/renderModules/AssetRenderer.h"
#include "saiga/vulkan/renderModules/LineAssetRenderer.h"
#include "saiga/vulkan/renderModules/PointCloudRenderer.h"
#include "saiga/vulkan/renderModules/TextureDisplay.h"
#include "saiga/vulkan/renderModules/TexturedAssetRenderer.h"
#include "saiga/vulkan/window/SDLSample.h"

#include <algorithm>
#include <iterator>
#include <random>
#include <saiga/core/imgui/imgui.h>
#include <utility>
#include <vector>
using namespace Saiga;
using Saiga::Vulkan::Memory::BufferMemoryLocation;
class VulkanExample : public VulkanSDLExampleBase
{
    std::array<std::string, 5> image_names{"cat.png", "red-panda.png", "dog.png", "pika.png", "ludi.png"};
    std::array<std::shared_ptr<Saiga::Image>, 5> images;
    std::vector<std::shared_ptr<Saiga::Vulkan::Buffer>> test_allocs;
    std::vector<std::pair<std::shared_ptr<Saiga::Vulkan::Buffer>, uint32_t>> allocations;
    std::vector<std::tuple<std::shared_ptr<Saiga::Vulkan::Texture2D>, Saiga::Vulkan::DynamicDescriptorSet, int32_t>>
        tex_allocations;
    std::vector<std::tuple<std::shared_ptr<Saiga::Vulkan::Texture2D>, Saiga::Vulkan::DynamicDescriptorSet, int32_t>>
        to_delete_tex;
    std::vector<std::pair<std::shared_ptr<Saiga::Vulkan::Buffer>, uint32_t>> num_allocations;

    std::mt19937 mersenne_twister, auto_mersenne, defrag_test_mersenne;


    //    std::array<vk::DeviceSize, 4> tex_sizes{256, 512, 1024, 2048};
    std::array<vk::DeviceSize, 4> sizes{256 * 256, 512 * 512, 1024 * 1024, 16 * 1024 * 1024};

    Saiga::Vulkan::Memory::BufferType buffer_type{vk::BufferUsageFlagBits::eTransferDst,
                                                  vk::MemoryPropertyFlagBits::eDeviceLocal};
    Saiga::Vulkan::Memory::ImageType image_type{vk::ImageUsageFlagBits::eSampled,
                                                vk::MemoryPropertyFlagBits::eDeviceLocal};

    Saiga::Vulkan::TextureDisplay textureDisplay;
    vk::DescriptorSet textureDes = nullptr;


    bool enable_auto_index = false;
    bool enable_defragger  = false;
    int auto_allocs        = 0;

   public:
    VulkanExample();
    ~VulkanExample() override;

    void init(Saiga::Vulkan::VulkanBase& base);


    void update(float dt) override;
    void transfer(vk::CommandBuffer cmd) override;
    void render(vk::CommandBuffer cmd) override;
    void renderGUI() override;

   private:
    int defrag_tests          = 1000;
    int defrag_rounds         = 50;
    int defrag_current_test   = -1;
    int defrag_current_round  = 0;
    float defrag_current_loc  = -1;
    float defrag_current_free = -1;
    std::vector<unsigned int> defrag_seeds;

    bool show_textures = false;
    Saiga::SDLCamera<Saiga::PerspectiveCamera> camera;


    //    bool displayModels = true;


    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;

    void alloc_index(int index);

    std::pair<std::shared_ptr<Saiga::Vulkan::Buffer>, uint32_t> allocate(Saiga::Vulkan::Memory::BufferType type,
                                                                         unsigned long long int size);
    std::pair<std::shared_ptr<Saiga::Vulkan::Buffer>, uint32_t> allocateEmpty(Saiga::Vulkan::Memory::BufferType type,
                                                                              unsigned long long int size);

    std::tuple<std::shared_ptr<Saiga::Vulkan::Texture2D>, Saiga::Vulkan::DynamicDescriptorSet, int32_t> allocate(
        Saiga::Vulkan::Memory::ImageType type, unsigned long long int size);
    void cleanup();
    void speedProfiling() const;
    void fragmentationProfiling();
    void defragProfiling();
};


VulkanExample::VulkanExample()
{
    auto_mersenne = std::mt19937();

    defrag_test_mersenne = std::mt19937(666);
    init(renderer->base());
}

VulkanExample::~VulkanExample() {}

void VulkanExample::init(Saiga::Vulkan::VulkanBase& base)
{
    for (auto i = 0U; i < image_names.size(); ++i)
    {
        auto image = std::make_shared<Saiga::Image>(image_names[i]);

        if (image->type == Saiga::UC3)
        {
            auto img2 = std::make_shared<Saiga::TemplatedImage<ucvec4>>(image->height, image->width);
            Saiga::ImageTransformation::addAlphaChannel(image->getImageView<ucvec3>(), img2->getImageView(), 255);
            image = img2;
        }

        images[i] = image;

        VLOG(3) << image_names[i] << " " << image.get()->size();
    }
    num_allocations.resize(10, std::make_pair(nullptr, 0));

    textureDisplay.init(base, renderer->renderPass);
}



void VulkanExample::update(float dt)
{
    static std::uniform_int_distribution<> size_dist(1, 64);
    camera.update(dt);
    camera.interpolate(dt, 0);

    static int frameCount = 0;

    static std::uniform_int_distribution<unsigned long> alloc_dist(1, 5), image_dist(0, 4), dealloc_dist(0, 5);

    frameCount++;

    static std::ofstream testFile;

    if (defrag_current_free >= 0 && defrag_current_free <= 1 && defrag_current_loc >= 0 && defrag_current_loc <= 1 &&
        defrag_current_test >= 0 && frameCount % 10 == 0)
    {
        if (defrag_current_test == 0 && defrag_current_round == 0)
        {
            auto now       = std::chrono::system_clock::now();
            auto in_time_t = std::chrono::system_clock::to_time_t(now);

            auto time = std::put_time(localtime(&in_time_t), "%Y%m%d_%H%M%S_");

            std::stringstream timestr;
            timestr << time;
            timestr << defrag_tests << "_" << defrag_rounds << "_" << (enable_defragger ? "defrag" : "nodefrag") << "_l"
                    << defrag_current_loc << "_f" << defrag_current_free << ".txt";
            testFile.open(timestr.str(), std::ios::out);

            auto& defragger                    = *renderer->base().memory.getAllocator(buffer_type).defragger;
            defragger.config.weight_location   = defrag_current_loc;
            defragger.config.weight_small_free = defrag_current_free;
            renderer->base().memory.enable_defragmentation(buffer_type, enable_defragger);
            if (enable_defragger) renderer->base().memory.start_defrag(buffer_type);
        }
        if (defrag_current_round == 0)
        {
            defrag_test_mersenne.seed(defrag_seeds[defrag_current_test]);
        }
        //        renderer->base().memory.enable_defragmentation(image_type, false);
        renderer->base().memory.stop_defrag(buffer_type);
        auto& chunks = renderer->base().memory.getAllocator(buffer_type).allocator->chunks;

        for (auto& chunk : chunks)
        {
            testFile << chunk.getFragmentation() << std::endl;
        }
        testFile << std::endl;
        renderer->base().memory.start_defrag(buffer_type);


        auto num_allocs = alloc_dist(defrag_test_mersenne);

        for (auto i = 0U; i < num_allocs; ++i)
        {
            test_allocs.push_back(allocateEmpty(buffer_type, size_dist(defrag_test_mersenne) * 128 * 1024).first);
        }
        //        renderer->base().memory.enable_defragmentation(buffer_type, enable_defragger);
        //        renderer->base().memory.start_defrag(buffer_type);
        //
        //        renderer->base().memory.enable_defragmentation(image_type, false);
        //        renderer->base().memory.stop_defrag(image_type);

        num_allocs = std::min<size_t>(dealloc_dist(defrag_test_mersenne), test_allocs.size());

        if (defrag_current_round != defrag_rounds)
        {
            for (auto i = 0U; i < num_allocs; ++i)
            {
                auto index = defrag_test_mersenne() % test_allocs.size();

                test_allocs.erase(test_allocs.begin() + index);
            }
        }
        //        renderer->base().memory.enable_defragmentation(image_type, enable_defragger);
        //        renderer->base().memory.start_defrag(image_type);
        defrag_current_round++;

        if (defrag_current_round == defrag_rounds)
        {
            defrag_current_test++;
            defrag_current_round = 0;

            //            std::move(tex_allocations.begin(), tex_allocations.end(), std::back_inserter(to_delete_tex));
            test_allocs.clear();

            if (defrag_current_test == defrag_tests)
            {
                defrag_current_test = 0;
                //                testFile.close();
                defrag_current_loc += 0.25f;
                if (defrag_current_loc > 1)
                {
                    defrag_current_loc = 0;
                    defrag_current_free += 0.25f;
                }
                if (defrag_current_free > 1)
                {
                    defrag_current_loc  = -1;
                    defrag_current_free = -1;
                }
                testFile.close();
            }
        }
    }

    if (enable_auto_index)
    {
        static std::uniform_int_distribution<> alloc_dist(0, 9);

        // std::cout << alloc_dist(auto_mersenne) << std::endl;
        alloc_index(alloc_dist(auto_mersenne));
        auto_allocs++;
    }

    std::for_each(to_delete_tex.begin(), to_delete_tex.end(), [](auto& entry) { std::get<2>(entry) -= 1; });

    auto end = std::remove_if(to_delete_tex.begin(), to_delete_tex.end(),
                              [](const auto& entry) { return std::get<2>(entry) < 0; });

    to_delete_tex.erase(end, to_delete_tex.end());
    //    renderer->base().memory.vertexIndexAllocator.deallocate(m_location3);
    //    m_location3 = renderer->base().memory.vertexIndexAllocator.allocate(1025);
}

void VulkanExample::transfer(vk::CommandBuffer cmd) {}


void VulkanExample::render(vk::CommandBuffer cmd)
{
    if (show_textures && !tex_allocations.empty())
    {
        if (textureDisplay.bind(cmd))
        {
            const int width = 16;
            int index       = 0;

            for (auto& texture : tex_allocations)
            {
                vec2 position((index % width) * 64, (index / width) * 64);
                textureDisplay.renderTexture(cmd, std::get<1>(texture), position, vec2(64, 64));
                index++;
            }
        }
    }
}

void writeToFile(bool chunk_allocator, const std::string& baseName, const std::string& time, bool singleAllocs,
                 int allocCount, int allocSizeKB, std::vector<double>& times, bool device)
{
    std::ofstream file;

    std::stringstream file_name;
    file_name << baseName << "_" << time << "_";

    if (device)
    {
        file_name << "device_";
    }
    else
    {
        file_name << "host_";
    }

    if (chunk_allocator)
    {
        file_name << "chunk_";
    }
    else
    {
        file_name << "unique_";
    }
    if (singleAllocs)
    {
        file_name << "dealloc_";
    }
    else
    {
        file_name << "nodealloc_";
    }
    file_name << allocCount << "_" << allocSizeKB << ".txt";
    file.open(file_name.str(), std::ios::out | std::ios::app);

    for (auto& time : times)
    {
        file << time << std::endl;
    }
    file.close();
}

inline Saiga::Vulkan::Memory::BufferMemoryLocation* performAlloc(Saiga::Vulkan::Memory::BufferType& type,
                                                                 Saiga::Vulkan::Memory::UniqueAllocator& allocator,
                                                                 vk::DeviceSize size)
{
    return allocator.allocate(type, size);
}

inline Saiga::Vulkan::Memory::BufferMemoryLocation* performAlloc(Saiga::Vulkan::Memory::BufferType& type,
                                                                 Saiga::Vulkan::Memory::BufferChunkAllocator& allocator,
                                                                 vk::DeviceSize size)
{
    return allocator.allocate(size);
}



template <typename Allocator>
inline void performSingleAllocs(int repetitions, int allocCount, int allocSizeKB, std::vector<double>& times,
                                std::vector<double>& times_dealloc, Saiga::Vulkan::Memory::BufferType type,
                                Allocator& allocator)
{
    for (int rep = 0; rep < repetitions; ++rep)
    {
        for (int i = 0; i < allocCount; i++)
        {
            BufferMemoryLocation* location = nullptr;

            {
                times.push_back(0.0);
                //                Saiga::ScopedTimer<std::chrono::microseconds, double> timer(times.back());
                auto timer = Saiga::make_scoped_timer<std::chrono::microseconds>(times.back());
                location   = performAlloc(type, allocator, allocSizeKB * 1024);
            }
            if (location)
            {
                {
                    times_dealloc.push_back(0.0);
                    auto timer = Saiga::make_scoped_timer<std::chrono::microseconds>(times_dealloc.back());
                    //                    Saiga::ScopedTimer<std::chrono::microseconds, double>
                    //                    timer(times_dealloc.back());
                    allocator.deallocate(location);
                }
            }
        }
    }
}
inline void performMassAllocs(int repetitions, int allocCount, int allocSizeKB, std::vector<double>& times,
                              std::vector<double>& times_dealloc, Saiga::Vulkan::Memory::BufferType type, bool chunk,
                              Saiga::Vulkan::VulkanBase& base)
{
    std::vector<BufferMemoryLocation*> locations;
    for (int rep = 0; rep < repetitions; ++rep)
    {
        Saiga::Vulkan::Memory::FirstFitStrategy<BufferMemoryLocation> ffs;
        Saiga::Vulkan::Memory::BufferChunkAllocator bca(base.physicalDevice, base.device, type, ffs,
                                                        base.transferQueue);
        Saiga::Vulkan::Memory::UniqueAllocator ua(base.device, base.physicalDevice);
        if (chunk)
        {
            {
                times.push_back(0.0);
                auto timer = Saiga::make_scoped_timer<std::chrono::microseconds>(times.back());
                for (int i = 0; i < allocCount; i++)
                {
                    locations.push_back(bca.allocate(allocSizeKB * 1024U));
                }
            }
            {
                times_dealloc.push_back(0.0);
                auto timer = Saiga::make_scoped_timer<std::chrono::microseconds>(times_dealloc.back());
                for (auto* location : locations)
                {
                    bca.deallocate(location);
                }
            }
        }
        else
        {
            {
                times.push_back(0.0);
                //                Saiga::ScopedTimer<std::chrono::microseconds, double> timer(times.back());
                auto timer = Saiga::make_scoped_timer<std::chrono::microseconds>(times.back());
                for (int i = 0; i < allocCount; i++)
                {
                    locations.push_back(ua.allocate(type, (allocSizeKB * 1024U)));
                }
            }
            {
                times_dealloc.push_back(0.0);
                //                Saiga::ScopedTimer<std::chrono::microseconds, double> timer(times_dealloc.back());
                auto timer = Saiga::make_scoped_timer<std::chrono::microseconds>(times_dealloc.back());
                for (auto* location : locations)
                {
                    ua.deallocate(location);
                }
            }
        }
        locations.clear();
    }
}

void VulkanExample::renderGUI()
{
    static std::uniform_int_distribution<unsigned long> alloc_dist(1, 5), size_dist(0UL, 3UL), image_dist(0, 4);

    ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiCond_FirstUseEver);
    ImGui::Begin("Example settings");

    bool old_enable = enable_defragger;
    ImGui::Checkbox("Defragger enabled", &old_enable);
    if (old_enable != enable_defragger)
    {
        enable_defragger = old_enable;
        renderer->base().memory.enable_defragmentation(buffer_type, enable_defragger);
    }

    ImGui::Checkbox("Auto allocate indexed", &enable_auto_index);
    ImGui::Text("%d", auto_allocs);

    //    static bool singleAllocs = true;
    speedProfiling();
    fragmentationProfiling();
    defragProfiling();


    if (ImGui::Button("Allocate Image"))
    {
        renderer->base().memory.enable_defragmentation(image_type, false);
        renderer->base().memory.stop_defrag(image_type);
        auto num_allocs = alloc_dist(mersenne_twister);

        for (auto i = 0U; i < num_allocs; ++i)
        {
            auto index = image_dist(mersenne_twister);
            tex_allocations.push_back(allocate(image_type, index));
        }
        renderer->base().memory.enable_defragmentation(buffer_type, enable_defragger);
        renderer->base().memory.start_defrag(buffer_type);
    }

    if (ImGui::Button("Deallocate Image"))
    {
        renderer->base().memory.enable_defragmentation(image_type, false);
        renderer->base().memory.stop_defrag(image_type);

        auto num_allocs = std::min<size_t>(alloc_dist(mersenne_twister), tex_allocations.size());


        for (auto i = 0U; i < num_allocs; ++i)
        {
            auto index = mersenne_twister() % tex_allocations.size();

            std::move(tex_allocations.begin() + index, tex_allocations.begin() + index + 1,
                      std::back_inserter(to_delete_tex));
            // auto& alloc = tex_allocations[index];
            // to_delete_tex.push_back(std::make_tuple(std::move(alloc.first), std::move(alloc.second), 12));
            tex_allocations.erase(tex_allocations.begin() + index);
        }
        renderer->base().memory.enable_defragmentation(image_type, enable_defragger);
        renderer->base().memory.start_defrag(image_type);
    }

    ImGui::Checkbox("Show textures", &show_textures);
    ImGui::End();

    window->renderImGui();
}
void VulkanExample::speedProfiling() const
{
    static int allocCount  = 100;
    static int allocSizeKB = 1024;
    static int repetitions = 1;
    static bool chunk      = true;
    static bool device_mem = true;

    ImGui::Text("Profiling");
    ImGui::Checkbox("Enable chunk alloc", &chunk);
    ImGui::Checkbox("Use Device memory", &device_mem);
    //    ImGui::Checkbox("Single allocs", &singleAllocs);
    ImGui::InputInt("repetitions", &repetitions);
    ImGui::InputInt("allocCount", &allocCount, 10);
    ImGui::InputInt("allocSizeKB", &allocSizeKB, 128, 1024);
    ImGui::Indent();
    if (ImGui::Button("*2"))
    {
        allocSizeKB *= 2;
    }
    ImGui::SameLine();
    if (ImGui::Button("/2"))
    {
        allocSizeKB /= 2;
    }
    ImGui::Unindent();


    const Saiga::Vulkan::Memory::BufferType device_type{
        {vk::BufferUsageFlagBits ::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal}};

    const Saiga::Vulkan::Memory::BufferType host_type{
        {vk::BufferUsageFlagBits ::eVertexBuffer, vk::MemoryPropertyFlagBits::eHostVisible}};

    const auto type = device_mem ? device_type : host_type;

    if (ImGui::Button("Profile single allocations"))
    {
        renderer->base().device.waitIdle();

        std::vector<double> times;
        std::vector<double> times_dealloc;


        auto now       = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);

        auto time = std::put_time(localtime(&in_time_t), "%Y%m%d_%H%M%S_");

        std::stringstream timestr;
        timestr << time;
        Saiga::Vulkan::Memory::FirstFitStrategy<BufferMemoryLocation> ffs;
        Saiga::Vulkan::Memory::BufferChunkAllocator bca(renderer->base().physicalDevice, renderer->base().device, type,
                                                        ffs, renderer->base().transferQueue);
        Saiga::Vulkan::Memory::UniqueAllocator ua(renderer->base().device, renderer->base().physicalDevice);

        if (chunk)
        {
            bca.deallocate(bca.allocate(1024));
            performSingleAllocs(repetitions, allocCount, allocSizeKB, times, times_dealloc, type, bca);
        }
        else
        {
            performSingleAllocs(repetitions, allocCount, allocSizeKB, times, times_dealloc, type, ua);
        }
        writeToFile(chunk, "allocate", timestr.str(), true, allocCount, allocSizeKB, times, type == device_type);
        writeToFile(chunk, "deallocate", timestr.str(), true, allocCount, allocSizeKB, times_dealloc,
                    type == device_type);
    }


    if (ImGui::Button("Profile mass allocation"))
    {
        renderer->base().device.waitIdle();

        std::vector<double> times;
        std::vector<double> times_dealloc;


        auto now       = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);

        auto time = std::put_time(localtime(&in_time_t), "%Y%m%d_%H%M%S_");

        std::stringstream timestr;
        timestr << time;

        if (chunk)
        {
            performMassAllocs(repetitions, allocCount, allocSizeKB, times, times_dealloc, type, true, renderer->base());
        }
        else
        {
            performMassAllocs(repetitions, allocCount, allocSizeKB, times, times_dealloc, type, false,
                              renderer->base());
        }

        writeToFile(chunk, "allocate", timestr.str(), false, allocCount, allocSizeKB, times, type == device_type);
        writeToFile(chunk, "deallocate", timestr.str(), false, allocCount, allocSizeKB, times_dealloc,
                    type == device_type);
    }

    if (ImGui::Button("Multi complete"))
    {
        std::vector<Saiga::Vulkan::Memory::BufferType> types{host_type, device_type};
        std::vector<bool> use_chunk{false, true};
        std::vector<vk::DeviceSize> sizes{128, 256, 512, 1024, 2048, 4096, 4096 * 2, 4096 * 4, 4096 * 8, 4096 * 16};
        renderer->base().device.waitIdle();

        for (const auto& type : types)
        {
            for (const auto& chunk : use_chunk)
            {
                for (const auto& size : sizes)
                {
                    {
                        using namespace std::chrono_literals;
                        std::this_thread::sleep_for(1s);
                    }
                    std::vector<double> times;
                    std::vector<double> times_dealloc;


                    auto now       = std::chrono::system_clock::now();
                    auto in_time_t = std::chrono::system_clock::to_time_t(now);

                    auto time = std::put_time(localtime(&in_time_t), "%Y%m%d_%H%M%S_");

                    std::stringstream timestr;
                    timestr << time;

                    performMassAllocs(repetitions, allocCount, size, times, times_dealloc, type, chunk,
                                      renderer->base());

                    writeToFile(chunk, "allocate", timestr.str(), false, allocCount, size, times, type == device_type);
                    writeToFile(chunk, "deallocate", timestr.str(), false, allocCount, size, times_dealloc,
                                type == device_type);
                }
            }
        }
    }
}

void VulkanExample::fragmentationProfiling()
{
    ImGui::Text("Fragmentation tests");
    static int testCount = 1000;
    static int rounds = 10, roundsMax = 100;
    static int maxAllocCount = 100;
    ImGui::InputInt("Count", &testCount);
    ImGui::DragIntRange2("Rounds MinMax", &rounds, &roundsMax);
    ImGui::InputInt("Max (de)allocs", &maxAllocCount);

    if (ImGui::Button("Profile fit strategies"))
    {
        const Saiga::Vulkan::Memory::BufferType device_type{
            {vk::BufferUsageFlagBits ::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal}};

        Saiga::Vulkan::Memory::FirstFitStrategy<BufferMemoryLocation> firstFit;
        Saiga::Vulkan::Memory::BestFitStrategy<BufferMemoryLocation> bestFit;
        Saiga::Vulkan::Memory::WorstFitStrategy<BufferMemoryLocation> worstFit;

        auto strategies =
            std::vector<Saiga::Vulkan::Memory::FitStrategy<BufferMemoryLocation>*>{&bestFit, &firstFit, &worstFit};

        std::mt19937 seed_rand(666);

        std::vector<unsigned int> seeds;


        // auto sizes = std::vector<vk::DeviceSize>{128, 256, 512, 1024, 1024 * 2, 1024 * 4, 1024 * 8, 1024 * 16};

        for (int i = 0; i < testCount; ++i)
        {
            seeds.push_back(seed_rand());
        }


        // for (int i = 0; i < testCount; ++i)
        //{
        //    std::cout << " " << seeds[i];
        //}
        auto& base = renderer->base();

        std::uniform_int_distribution<> size_dist(1, 128);


        auto now       = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);

        auto time = std::put_time(localtime(&in_time_t), "%Y%m%d_%H%M%S_");



        for (auto* strat : strategies)
        {
            std::stringstream fileName, speedFileName;

            fileName << time;

            fileName << testCount << "_" << rounds << "_" << roundsMax << "_" << maxAllocCount << "_";

            if (strat == &firstFit) fileName << "first";
            if (strat == &bestFit) fileName << "besty";
            if (strat == &worstFit) fileName << "worst";

            speedFileName << fileName.str() << "_speed.txt";

            fileName << ".txt";


            std::ofstream file, speedFile;
            file.open(fileName.str(), std::ios::out);
            speedFile.open(speedFileName.str(), std::ios::out);

            for (const auto& seed : seeds)
            {
                std::vector<double> times;
                std::mt19937 test_rand(seed);

                Saiga::Vulkan::Memory::BufferChunkAllocator bca(base.physicalDevice, base.device, device_type, *strat,
                                                                base.transferQueue);

                bca.totalTime = 0.0;
                std::uniform_int_distribution<> rounds_dist(rounds, roundsMax);
                std::uniform_int_distribution<> alloc_dist(0, maxAllocCount);
                int currRounds = rounds_dist(test_rand);

                std::vector<BufferMemoryLocation*> locations;
                for (auto round = 0; round < currRounds; round++)
                {
                    int numAllocs   = alloc_dist(test_rand);
                    int numDeallocs = alloc_dist(test_rand);


                    for (int dealloc = 0; dealloc < numDeallocs; ++dealloc)
                    {
                        if (locations.empty())
                        {
                            break;
                        }
                        std::uniform_int_distribution<> dealloc_dist(0, locations.size() - 1);

                        auto dealloc_index = dealloc_dist(test_rand);
                        bca.deallocate(locations[dealloc_index]);
                        locations.erase(locations.begin() + dealloc_index);
                    }

                    for (int alloc = 0; alloc < numAllocs; ++alloc)
                    {
                        times.push_back(0.0);
                        //                        Saiga::ScopedTimer<std::chrono::microseconds, double>
                        //                        timer(times.back());
                        auto timer = Saiga::make_scoped_timer<std::chrono::microseconds>(times.back());
                        locations.push_back(bca.allocate(size_dist(test_rand) * 128U * 1024U));
                    }
                }

                // if (bca.chunks.empty())
                //{
                //    // std::cout << "empty" << std::endl;
                //    continue;
                //}
                // float sum = 0.0f;
                for (auto& chunk : bca.chunks)
                {
                    // sum += chunk.getFragmentation();
                    file << chunk.getFragmentation() << std::endl;
                }

                auto times_sum = 0.0;
                for (const auto& time : times)
                {
                    times_sum += time;
                }

                times_sum /= times.size();

                speedFile << bca.totalTime / times.size() << std::endl;
            }
            file.close();
            speedFile.close();
        }
    }
}


void VulkanExample::defragProfiling()
{
    ImGui::Text("Defrag profiling");

    ImGui::InputInt("Tests", &defrag_tests);
    ImGui::InputInt("Rounds", &defrag_rounds);

    ImGui::Text("Current test %d", defrag_current_test);
    ImGui::Text("Current round %d", defrag_current_round);
    ImGui::Text("Current loc %f", defrag_current_loc);
    ImGui::Text("Current free %f", defrag_current_free);

    if (ImGui::Button("Begin"))
    {
        defrag_seeds.resize(defrag_tests);
        defrag_test_mersenne.seed(666);

        for (auto& seed : defrag_seeds)
        {
            seed = defrag_test_mersenne();
        }

        defrag_current_test  = 0;
        defrag_current_round = 0;
        defrag_current_loc   = 0;
        defrag_current_free  = 0;
    }
}

void VulkanExample::keyPressed(SDL_Keysym key)
{
    static std::uniform_int_distribution<unsigned long> alloc_dist(1UL, 8UL), size_dist(0UL, 3UL), image_dist(0, 4);



    int single_unassign = -1;
    unsigned long num_allocs;
    switch (key.scancode)
    {
        case SDL_SCANCODE_0:
            single_unassign = 10;
            break;
        case SDL_SCANCODE_9:
            single_unassign = 9;
            break;
        case SDL_SCANCODE_8:
            single_unassign = 8;
            break;
        case SDL_SCANCODE_7:
            single_unassign = 7;
            break;
        case SDL_SCANCODE_6:
            single_unassign = 6;
            break;
        case SDL_SCANCODE_5:
            single_unassign = 5;
            break;
        case SDL_SCANCODE_4:
            single_unassign = 4;
            break;
        case SDL_SCANCODE_3:
            single_unassign = 3;
            break;
        case SDL_SCANCODE_2:
            single_unassign = 2;
            break;
        case SDL_SCANCODE_1:
            single_unassign = 1;
            break;

        case SDL_SCANCODE_A:
            renderer->base().memory.enable_defragmentation(buffer_type, false);
            renderer->base().memory.stop_defrag(buffer_type);
            num_allocs = alloc_dist(mersenne_twister);

            for (auto i = 0U; i < num_allocs; ++i)
            {
                auto size = sizes[size_dist(mersenne_twister)];
                // allocations.push_back(renderer->base().memory.allocate(buffer_type, size));
                allocations.push_back(allocate(buffer_type, size));
            }
            renderer->base().memory.enable_defragmentation(buffer_type, enable_defragger);
            renderer->base().memory.start_defrag(buffer_type);
            break;
        case SDL_SCANCODE_D:

            renderer->base().memory.enable_defragmentation(buffer_type, false);
            renderer->base().memory.stop_defrag(buffer_type);

            num_allocs = std::min<size_t>(alloc_dist(mersenne_twister), allocations.size());

            for (auto i = 0U; i < num_allocs; ++i)
            {
                auto index = mersenne_twister() % allocations.size();

                // renderer->base().memory.deallocateBuffer(buffer_type, allocations[index].first);
                allocations.erase(allocations.begin() + index);
            }
            renderer->base().memory.enable_defragmentation(buffer_type, enable_defragger);
            renderer->base().memory.start_defrag(buffer_type);

            break;
        case SDL_SCANCODE_Z:
        {
            renderer->base().memory.enable_defragmentation(image_type, false);
            renderer->base().memory.stop_defrag(image_type);
            num_allocs = alloc_dist(mersenne_twister);

            for (auto i = 0U; i < num_allocs; ++i)
            {
                auto index = image_dist(mersenne_twister);
                // allocations.push_back(renderer->base().memory.allocate(buffer_type, size));
                tex_allocations.push_back(allocate(image_type, index));
            }
            renderer->base().memory.enable_defragmentation(image_type, enable_defragger);
            renderer->base().memory.start_defrag(image_type);
        }
        break;
        case SDL_SCANCODE_C:
        {
            renderer->base().memory.enable_defragmentation(image_type, false);
            renderer->base().memory.stop_defrag(image_type);

            num_allocs = std::min<size_t>(alloc_dist(mersenne_twister), tex_allocations.size());


            for (auto i = 0U; i < num_allocs; ++i)
            {
                auto index = mersenne_twister() % tex_allocations.size();
                VLOG(3) << "Dealloc image " << index;
                std::move(tex_allocations.begin() + index, tex_allocations.begin() + index + 1,
                          std::back_inserter(to_delete_tex));
                tex_allocations.erase(tex_allocations.begin() + index);
            }
            renderer->base().memory.enable_defragmentation(image_type, enable_defragger);
            renderer->base().memory.start_defrag(image_type);
        }
        break;
        case SDL_SCANCODE_R:
            renderer->base().memory.enable_defragmentation(buffer_type, false);
            renderer->base().memory.stop_defrag(buffer_type);

            for (int i = 0; i < 10; ++i)
            {
                alloc_index(i);
            }
            renderer->base().memory.enable_defragmentation(buffer_type, enable_defragger);
            renderer->base().memory.start_defrag(buffer_type);
            break;
        case SDL_SCANCODE_V:
        {
            for (auto& numAlloc : num_allocations)
            {
                if (numAlloc.first)
                {
                    auto& buffer      = *numAlloc.first.get();
                    auto num_elements = buffer.size() / sizeof(uint32_t);
                    std::vector<uint32_t> copy(num_elements);

                    buffer.stagedDownload(copy.data());


                    if (copy[0] != numAlloc.second)
                    {
                        LOG(ERROR) << "Wrong value " << copy[0] << " != " << numAlloc.second;
                    }
                    else
                    {
                        VLOG(3) << "Verified " << copy[0];
                    }
                }
            }
        }
        break;
        case SDL_SCANCODE_ESCAPE:
            cleanup();
            window->close();
            break;

        case SDL_SCANCODE_J:
            renderer->base().memory.full_defrag();
            break;
        default:
            break;
    }

    if (single_unassign > 0)
    {
        auto index = single_unassign - 1;
        alloc_index(index);
    }
}

void VulkanExample::alloc_index(int index)
{
    if (num_allocations[index].first)
    {
        // num_allocations[index].first->destroy();
        // BufferMemoryLocation* loc = num_allocations[index].first;
        // allocations.erase(allocations.begin() + index);
        // renderer->base().memory.deallocateBuffer(buffer_type, loc);
        num_allocations[index] = std::make_pair(nullptr, 0);
    }
    else
    {
        num_allocations[index] = allocate(buffer_type, sizes[3]);
        // num_allocations[index] = renderer->base().memory.allocate(buffer_type, sizes[3]);
    }
}

void VulkanExample::keyReleased(SDL_Keysym key) {}

std::pair<std::shared_ptr<Saiga::Vulkan::Buffer>, uint32_t> VulkanExample::allocate(
    Saiga::Vulkan::Memory::BufferType type, unsigned long long int size)
{
    static std::uniform_int_distribution<unsigned long> init_dist(0UL, 1024UL);

    auto start = init_dist(mersenne_twister);

    VLOG(3) << "Creating buffer of size " << size << " beginning at " << start;
    std::vector<uint32_t> mem;
    mem.resize(size / sizeof(uint32_t));
    std::iota(mem.begin(), mem.end(), start);
    std::shared_ptr<Saiga::Vulkan::Buffer> buffer = std::make_shared<Saiga::Vulkan::Buffer>();
    buffer->createBuffer(renderer->base(), size, type.usageFlags, type.memoryFlags);

    buffer->stagedUpload(renderer->base(), size, mem.data());

    buffer->mark_dynamic();

    return std::make_pair(buffer, start);
}

std::pair<std::shared_ptr<Saiga::Vulkan::Buffer>, uint32_t> VulkanExample::allocateEmpty(
    Saiga::Vulkan::Memory::BufferType type, unsigned long long int size)
{
    static std::uniform_int_distribution<unsigned long> init_dist(0UL, 1024UL);

    auto start = init_dist(mersenne_twister);

    VLOG(3) << "Creating buffer of size " << size << " beginning at " << start;

    std::shared_ptr<Saiga::Vulkan::Buffer> buffer = std::make_shared<Saiga::Vulkan::Buffer>();
    buffer->createBuffer(renderer->base(), size, type.usageFlags, type.memoryFlags);

    buffer->mark_dynamic();

    return std::make_pair(buffer, start);
}

std::tuple<std::shared_ptr<Saiga::Vulkan::Texture2D>, Saiga::Vulkan::DynamicDescriptorSet, int32_t>
VulkanExample::allocate(Saiga::Vulkan::Memory::ImageType type, unsigned long long int index)
{
    std::shared_ptr<Saiga::Vulkan::Texture2D> texture = std::make_shared<Saiga::Vulkan::Texture2D>();

    texture->fromImage(renderer->base(), *images[index]);
    texture->mark_dynamic();

    auto descriptor = textureDisplay.createDynamicDescriptorSet();
    descriptor.assign(0, texture.get());
    return std::make_tuple(texture, std::move(descriptor), 12);
}

void VulkanExample::cleanup()
{
    renderer->base().device.waitIdle();

    VLOG(3) << allocations.size();
    // if (!allocations.empty())
    //{
    //    allocations.resize(0);
    //}
    // if (!num_allocations.empty())
    //{
    //    num_allocations.resize(0);
    //}
    // if (!tex_allocations.empty())
    //{
    //    tex_allocations.resize(0);
    //}
}


#undef main
int main(const int argc, const char* argv[])
{
    initSaigaSample();
    VulkanExample example;
    example.run();
    return 0;
}
