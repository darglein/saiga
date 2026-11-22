/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/scene/RenderSceneTypes.h"
#include "saiga/vulkan/texture/Texture.h"

#include <array>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
// The camera and the light of a frame for CascadedShadows::fit (double, world coordinates of the camera's space)
struct CascadeInput
{
    Vec3 camera         = Vec3::Zero();
    Mat3 cameraRotation = Mat3::Identity();  // the camera looks along -z of it
    double fovy = 1, aspect = 1, nearPlane = 0.1;
    Vec3 towardsLight = Vec3(0, 1, 0);
    uint64_t frame    = 0;
    // The camera speed of the last second (m/s, at least 50) and the frame time (s): the margin of staggered cascades
    double maxSpeed = 50, dt = 1.0 / 60.0;
};

/**
 * The sun / moon cascaded shadow maps of RenderScene (S36a2, A5 of docs/stages/S36_lighting.plan.md, §10.7):
 * - Splits: lambda * log + (1 - lambda) * linear between the camera's near plane and shadowDistance (cascadeSplits).
 * - Light space (lightBasis): z towards the light, x = normalize(cross(north, z)) with north = (0, 0, -1) (falling back
 *   to (1, 0, 0) near the zenith), y = cross(z, x); fixed for a light direction, so the texel snap is stable.
 * - Fitting: a cascade updated every frame (interval 1) is fitted to the bounding sphere of its slice of the camera
 *   frustum (rotation invariant: no shimmer when the camera turns); a staggered cascade (interval > 1) is fitted around
 *   the camera position with the distance of the slice's far corners + interval * maxSpeed * dt (quantized up to 1/16
 *   of the far distance), so it covers every receiver of its slice after any turn and move until its next update (and
 *   it is refit early when the camera leaves that margin, e.g. a teleport). The rectangle
 *   is the sphere's square enlarged by the blend band (the slice stays inside the band-free part), its centre snapped
 *   to the texel grid in light space and kept in double (world); the camera-relative matrix is rebuilt every frame.
 * - One light-space depth range (D23): the cascades rendered in a frame share it, the union of their spheres' depth
 *   extents; casters nearer to the light are pancaked in the shadow vertex shaders. A cascade is its xy rectangle, so
 *   the receivers' selection and the casters' de-duplication by the rectangle are exact.
 * - Depth: D32, orthographic, not reversed (clear 1, LESS_OR_EQUAL); one Texture2DArray layer per cascade.
 * Golden mode (LightingParameters::forceFullUpdates) updates every cascade every frame (interval 1: no camera speed
 * margin, deterministic images). Freezing keeps every cascade's world data (and the camera they were fitted to); the
 * frozen cascades are still rendered every frame (the casters may move), from the current camera's position.
 */
class SAIGA_VULKAN_API CascadedShadows
{
   public:
    struct Cascade
    {
        bool valid          = false;
        uint64_t lastUpdate = 0;
        // Rendered in the frame of the last fit
        bool updated = false;
        int interval = 1;
        // The light basis of its fit (each cascade keeps its own: a staggered cascade is sampled with its matrix)
        Vec3 axisX = Vec3(1, 0, 0), axisY = Vec3(0, 1, 0), axisZ = Vec3(0, 0, 1);
        // The snapped rectangle's centre and half extent in light space (absolute, world)
        double cx = 0, cy = 0, halfExtent = 0;
        // The depth range in absolute light z: the receivers' near end (towards the light) and far end
        double zNear = 0, zFar = 0;
        double splitNear = 0, splitFar = 0, texelSize = 0;
        int resolution = 0;
        // A staggered cascade: the camera position of its fit and the distance the camera may move from it (it is
        // refit when the camera leaves it, e.g. a teleport)
        Vec3 fitCamera = Vec3::Zero();
        double margin  = 0;
    };

    // ---- pure helpers (tests)
    // count + 1 distances from nearPlane to farPlane
    static std::vector<double> cascadeSplits(double nearPlane, double farPlane, int count, double lambda);
    static void lightBasis(const Vec3& towardsLight, Vec3& x, Vec3& y, Vec3& z);
    // The shadow light (D13): 0 none, 1 the sun, 2 the moon by their ground illuminance, with a hysteresis of a factor
    // 2 (the other body takes over only when it is twice as bright)
    static int selectShadowLight(int previous, double sunGround, double moonGround);
    // The ground illuminance of a body (lux) of the top-of-atmosphere illuminance at the elevation: an air-mass model
    // (Kasten-Young) of the clear atmosphere's attenuation, 0 below the horizon
    static double groundIlluminance(double topIlluminance, double elevationRad);
    // The bounding sphere of a slice [n, f] of a frustum with the half-diagonal tangent k: the centre's distance along
    // the view axis and the radius
    static void sliceSphere(double n, double f, double k, double& centreDistance, double& radius);

    // ---- CPU: the cascades of a frame
    // Fits the due cascades (none while frozen; every cascade on a parameter or count change) for the light; the
    // updated ones are rendered this frame
    void fit(const CascadeInput& in, const LightingParameters& p, bool frozen);
    // Drops every cascade (no shadow light): the next fit refits all
    void invalidate();
    const std::array<Cascade, kMaxCascades>& cascades() const { return state; }
    int count() const { return activeCount; }
    // The camera-relative shadow view of cascade c (the de-duplication against c - 1, the caps, the small-caster rule
    // and the LOD bias of the parameters) and its camera-relative light view-projection
    ShadowView view(int c, const Vec3& camera, const LightingParameters& p, double elevationRad) const;
    mat4 matrix(int c, const Vec3& camera) const;
    // The camera the cascades were fitted to (frozen: the frozen camera)
    const CascadeInput& fittedCamera() const { return lastInput; }

    // ---- GPU
    void init(VulkanBase& base);
    void destroy();
    ~CascadedShadows() { destroy(); }
    // Recreates the depth array when the count or the resolution changed (the old one after the frames in flight);
    // every layer cleared to 1 and in eShaderReadOnlyOptimal
    void ensureImages(int count, int resolution);
    bool hasImages() const { return depthArray != nullptr; }
    // Releases the depth array (after the frames in flight; shadows off)
    void releaseImages();
    Texture2DArray* depth() { return depthArray.get(); }
    vk::ImageView layerView(int c) const { return layerViews[size_t(c)]; }
    int resolution() const { return imageResolution; }
    int layers() const { return imageLayers; }
    uint64_t memoryBytes() const;

   private:
    std::array<Cascade, kMaxCascades> state;
    int activeCount = 0;
    LightingParameters fittedFor;
    bool anyFit = false;
    CascadeInput lastInput;

    VulkanBase* base = nullptr;
    std::unique_ptr<Texture2DArray> depthArray;
    std::array<vk::ImageView, kMaxCascades> layerViews = {};
    int imageResolution = 0, imageLayers = 0;
};

}  // namespace Vulkan
}  // namespace Saiga
