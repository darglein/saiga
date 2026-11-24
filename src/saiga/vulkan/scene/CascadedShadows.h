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
 * - One light-space depth range (D23): the cascades rendered in a frame get the union of the depth extents of every
 *   valid cascade's sphere (also the staggered ones not rendered this frame: the range depends on the cascade layout,
 *   not on which cascades are due), so it holds every receiver; casters nearer to the light are pancaked in the shadow
 *   vertex shaders. A cascade is its xy rectangle, so the receivers' selection by the rectangle is exact.
 * - De-duplication: a cascade c skips the casters inside its stored de-duplication volume (xy footprint in a rectangle,
 *   light depth in an interval), set when it is rendered: cascade c - 1's rectangle shrunk by the blend band times c's
 *   caster depth range (its depth range up to the instance cap), for a staggered cascade both shrunk by a guard of 1/16
 *   of c - 1's half extent, and only if c - 1's caster volume contains it (in the other light basis with the
 *   rotation's slack).
 *   In every later frame in which c is not rendered the containment is checked against c - 1's current volume (the map
 *   sampled in that frame); if it fails c is rendered again. So a skipped caster is always in the map of c - 1 that is
 *   sampled with c's, and for a static view the caster sets are the same every frame.
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
        // The fitted sphere's centre (world) and the caster volume's end towards the light (absolute light z: zNear
        // plus the instance cap of the frame it was rendered in)
        Vec3 centre      = Vec3::Zero();
        double casterTop = 0;
        // The de-duplication volume (absolute light space of this cascade's basis: the rectangle's centre and half
        // extent, the light depth interval; dedup false: none)
        bool dedup     = false;
        double dedupCx = 0, dedupCy = 0, dedupHalf = 0, dedupZ0 = 0, dedupZ1 = 0;
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
    // The instances' caster cap towards the light beyond a cascade's receivers (maxCasterHeight / sin of the elevation,
    // at least minShadowElevationDeg)
    static double instanceCap(const LightingParameters& p, double elevationRad);
    // The guard of a staggered cascade's de-duplication rectangle (a fraction of the previous cascade's half extent)
    static constexpr double kDedupGuard = 1.0 / 16.0;
    // The resolution of the depth images and of the fit (the parameter clamped to 256..4096; S36a2 review, finding 10:
    // the fit's texel matches the image's)
    static int clampedResolution(const LightingParameters& p);
    // The receivers' edge margin and the blend band of a cascade, fractions of its half extent (S36a2 review, finding
    // 12): within the PCF footprint (kPcfMarginTexels) plus the normal offset (normalOffsetTexels, up to its full
    // length; CodeRabbit #85 round 2) of the rectangle's edge the taps would read beyond the layer (clamp to edge), so
    // the cascade's weight is 0 there (the shaders blend over band - margin); the band is at least the margin plus
    // kMinBlendTexels (also with cascadeBlend 0)
    // (one constant per declaration: MSVC C2487 in a dll interface class)
    static constexpr double kPcfMarginTexels = 2;
    static constexpr double kMinBlendTexels  = 4;
    static double edgeMarginTexels(const LightingParameters& p);
    static double edgeMargin(const LightingParameters& p, int resolution)
    {
        return 2 * edgeMarginTexels(p) / double(std::max(resolution, 1));
    }
    static double blendBand(const LightingParameters& p, int resolution);
    // The shadow light's direction the cascades are fitted to moves in steps of this angle (S36a2 review, finding 13:
    // the light basis refit to every small move of a moving sun rotated the texel grid in every frame, and every shadow
    // edge crawled); golden mode (forceFullUpdates) uses the exact light
    static constexpr double kLightStepDeg = 0.05;

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
    // The light direction the cascades were fitted to (in steps of kLightStepDeg)
    const Vec3& fittedLight() const { return fitLight; }

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
    Vec3 fitLight = Vec3(0, 1, 0);

    // fit's steps: one cascade's sphere and rectangle; cascade c's de-duplication against c - 1 (rendered cascades) and
    // whether the stored one still holds (cascades not rendered this frame)
    // The due cascades of a fit (in.towardsLight: the stepped light), the shared depth range, the de-duplication
    void fitDue(const CascadeInput& in, const LightingParameters& p);
    void fitCascade(int c, const CascadeInput& in, const LightingParameters& p, const Vec3& x, const Vec3& y,
                    const Vec3& z, const std::vector<double>& splits, double k, const Vec3& forward);
    void setDedup(int c, const LightingParameters& p);
    bool dedupHolds(int c, const LightingParameters& p) const;

    VulkanBase* base = nullptr;
    std::unique_ptr<Texture2DArray> depthArray;
    std::array<vk::ImageView, kMaxCascades> layerViews = {};
    int imageResolution = 0, imageLayers = 0;
};

}  // namespace Vulkan
}  // namespace Saiga
