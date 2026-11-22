/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/CascadedShadows.h"

#include <algorithm>
#include <cmath>

namespace Saiga
{
namespace Vulkan
{
using Stage  = vk::PipelineStageFlagBits2;
using Access = vk::AccessFlagBits2;

namespace
{
constexpr double kPi = 3.14159265358979323846;

// The fields whose change refits every cascade
bool sameFit(const LightingParameters& a, const LightingParameters& b)
{
    return a.cascadeCount == b.cascadeCount && a.cascadeResolution == b.cascadeResolution &&
           a.shadowDistance == b.shadowDistance && a.splitLambda == b.splitLambda &&
           a.cascadeInterval == b.cascadeInterval && a.cascadeBlend == b.cascadeBlend;
}
}  // namespace

std::vector<double> CascadedShadows::cascadeSplits(double nearPlane, double farPlane, int count, double lambda)
{
    count     = std::max(count, 1);
    nearPlane = std::max(nearPlane, 1e-3);
    farPlane  = std::max(farPlane, nearPlane * 1.001);
    std::vector<double> s(size_t(count) + 1);
    for (int i = 0; i <= count; ++i)
    {
        const double t   = double(i) / double(count);
        const double lg  = nearPlane * std::pow(farPlane / nearPlane, t);
        const double lin = nearPlane + (farPlane - nearPlane) * t;
        s[size_t(i)]     = lambda * lg + (1 - lambda) * lin;
    }
    s.front() = nearPlane;
    s.back()  = farPlane;
    return s;
}

void CascadedShadows::lightBasis(const Vec3& towardsLight, Vec3& x, Vec3& y, Vec3& z)
{
    z          = towardsLight.normalized();
    Vec3 north = Vec3(0, 0, -1);
    if (std::abs(north.dot(z)) > 0.999) north = Vec3(1, 0, 0);
    x = north.cross(z).normalized();
    y = z.cross(x);
}

int CascadedShadows::selectShadowLight(int previous, double sunGround, double moonGround)
{
    const bool sun = sunGround > 0, moon = moonGround > 0;
    if (!sun && !moon) return 0;
    if (!moon) return 1;
    if (!sun) return 2;
    if (previous == 1) return moonGround > 2 * sunGround ? 2 : 1;
    if (previous == 2) return sunGround > 2 * moonGround ? 1 : 2;
    return sunGround >= moonGround ? 1 : 2;
}

double CascadedShadows::groundIlluminance(double topIlluminance, double elevationRad)
{
    if (!(elevationRad > 0) || !(topIlluminance > 0)) return 0;
    const double deg     = elevationRad * 180.0 / kPi;
    const double airMass = 1.0 / (std::sin(elevationRad) + 0.50572 * std::pow(deg + 6.07995, -1.6364));
    // About 0.21 optical depth (visible band) for the clear default atmosphere, times the cosine on the ground
    return topIlluminance * std::sin(elevationRad) * std::exp(-0.21 * airMass);
}

void CascadedShadows::sliceSphere(double n, double f, double k, double& centreDistance, double& radius)
{
    // The corners at n and f with lateral radii n k and f k: (z - n)^2 + (n k)^2 = (f - z)^2 + (f k)^2
    double z = 0.5 * (n + f) * (1 + k * k);
    if (z > f) z = f;
    centreDistance = z;
    radius         = std::sqrt((f - z) * (f - z) + f * f * k * k);
}

void CascadedShadows::invalidate()
{
    for (Cascade& c : state) c = Cascade();
    activeCount = 0;
    anyFit      = false;
}

void CascadedShadows::fit(const CascadeInput& in, const LightingParameters& p, bool frozen)
{
    const int count = std::clamp(p.cascadeCount, 1, kMaxCascades);
    for (Cascade& c : state) c.updated = false;
    if (frozen && anyFit)
    {
        // Frozen: the world data and the camera stay; every valid cascade is rendered (the casters may move)
        for (int c = 0; c < activeCount; ++c) state[size_t(c)].updated = state[size_t(c)].valid;
        return;
    }
    const bool refitAll = !anyFit || !sameFit(p, fittedFor);
    if (refitAll)
        for (Cascade& c : state) c.valid = false;
    activeCount = count;
    fittedFor   = p;
    anyFit      = true;
    lastInput   = in;

    Vec3 x, y, z;
    lightBasis(in.towardsLight, x, y, z);
    const std::vector<double> splits = cascadeSplits(in.nearPlane, p.shadowDistance, count, p.splitLambda);
    const double tanY                = std::tan(in.fovy * 0.5);
    const double tanX                = tanY * in.aspect;
    const double k                   = std::sqrt(tanX * tanX + tanY * tanY);
    const Vec3 forward               = in.cameraRotation * Vec3(0, 0, -1);
    const double blend               = std::clamp(double(p.cascadeBlend), 0.0, 0.5);
    const int res                    = std::max(p.cascadeResolution, 16);

    // The spheres of the due cascades, then their shared depth range
    double zNear = -1e300, zFar = 1e300;
    bool any = false;
    for (int c = 0; c < count; ++c)
    {
        Cascade& s         = state[size_t(c)];
        const int interval = c == 0 ? 1 : std::max(p.cascadeInterval[size_t(c)], 1);
        const bool due     = !s.valid || interval == 1 || in.frame < s.lastUpdate ||
                         in.frame - s.lastUpdate >= uint64_t(interval) || s.resolution != res ||
                         (in.camera - s.fitCamera).norm() > s.margin;
        if (!due) continue;
        const double n = splits[size_t(c)], f = splits[size_t(c) + 1];
        Vec3 centre;
        double r;
        if (interval == 1)
        {
            double zc;
            sliceSphere(n, f, k, zc, r);
            centre   = in.camera + forward * zc;
            s.margin = 0;
        }
        else
        {
            // Around the camera: the slice's far corners plus the move until the next update; quantized up (a stable
            // texel size)
            centre              = in.camera;
            const double corner = f * std::sqrt(1 + k * k);
            r                   = corner + double(interval) * std::max(in.maxSpeed, 50.0) * std::max(in.dt, 0.0);
            const double q      = std::max(f / 16.0, 1.0);
            r                   = std::ceil(r / q) * q;
            s.margin            = r - corner;
        }
        s.fitCamera        = in.camera;
        const double h     = r / (1 - blend);
        const double texel = 2 * h / double(res);
        s.valid            = true;
        s.updated          = true;
        s.lastUpdate       = in.frame;
        s.interval         = interval;
        s.axisX            = x;
        s.axisY            = y;
        s.axisZ            = z;
        s.cx               = std::round(x.dot(centre) / texel) * texel;
        s.cy               = std::round(y.dot(centre) / texel) * texel;
        s.halfExtent       = h;
        s.splitNear        = n;
        s.splitFar         = f;
        s.texelSize        = texel;
        s.resolution       = res;
        const double lz    = z.dot(centre);
        zNear              = std::max(zNear, lz + h);
        zFar               = std::min(zFar, lz - h);
        any                = true;
    }
    if (!any) return;
    for (int c = 0; c < count; ++c)
    {
        Cascade& s = state[size_t(c)];
        if (!s.updated) continue;
        s.zNear = zNear;
        s.zFar  = zFar;
    }
    for (int c = count; c < kMaxCascades; ++c) state[size_t(c)] = Cascade();
}

mat4 CascadedShadows::matrix(int c, const Vec3& camera) const
{
    const Cascade& s = state[size_t(c)];
    mat4 m           = mat4::Zero();
    if (!s.valid) return mat4::Identity();
    const double h = s.halfExtent, range = std::max(s.zNear - s.zFar, 1e-3);
    // NDC x = (x.(p + camera) - cx) / h, y negated (Vulkan's y down, as the camera's projection: the winding stays),
    // depth (zNear - z.(p + camera)) / range; the translations in double
    const double tx = (s.axisX.dot(camera) - s.cx) / h;
    const double ty = -(s.axisY.dot(camera) - s.cy) / h;
    const double tz = (s.zNear - s.axisZ.dot(camera)) / range;
    for (int j = 0; j < 3; ++j)
    {
        m(0, j) = float(s.axisX(j) / h);
        m(1, j) = float(-s.axisY(j) / h);
        m(2, j) = float(-s.axisZ(j) / range);
    }
    m(0, 3) = float(tx);
    m(1, 3) = float(ty);
    m(2, 3) = float(tz);
    m(3, 3) = 1;
    return m;
}

ShadowView CascadedShadows::view(int c, const Vec3& camera, const LightingParameters& p, double elevationRad) const
{
    const Cascade& s = state[size_t(c)];
    ShadowView v;
    v.axisX            = s.axisX;
    v.axisY            = s.axisY;
    v.axisZ            = s.axisZ;
    const Vec2 rel     = Vec2(s.cx - s.axisX.dot(camera), s.cy - s.axisY.dot(camera));
    v.rectMin          = rel - Vec2(s.halfExtent, s.halfExtent);
    v.rectMax          = rel + Vec2(s.halfExtent, s.halfExtent);
    const double cz    = s.axisZ.dot(camera);
    v.zMin             = s.zFar - cz;
    v.zMax             = s.zNear - cz;
    const double minEl = std::max(elevationRad, double(std::clamp(p.minShadowElevationDeg, 0.5f, 90.0f)) * kPi / 180.0);
    v.instanceCap      = double(p.maxCasterHeight) / std::sin(minEl);
    v.terrainCap       = double(p.cascadeCasterDistance) / std::max(std::cos(std::max(elevationRad, 0.0)), 0.25);
    // De-duplication against the previous cascade's rectangle shrunk by its band (the same light basis only)
    if (c > 0)
    {
        const Cascade& prev = state[size_t(c) - 1];
        if (prev.valid && (prev.axisZ - s.axisZ).norm() < 1e-9)
        {
            const double band = std::clamp(double(p.cascadeBlend), 0.0, 0.5) * prev.halfExtent;
            const Vec2 pr     = Vec2(prev.cx - s.axisX.dot(camera), prev.cy - s.axisY.dot(camera));
            const double in   = prev.halfExtent - band;
            v.dedup           = true;
            v.dedupMin        = pr - Vec2(in, in);
            v.dedupMax        = pr + Vec2(in, in);
        }
    }
    v.smallRadius = double(p.smallCasterTexels) * s.texelSize;
    v.lodBias     = c >= 2 ? std::max(p.shadowLodBias, 0) : 0;
    v.viewProj    = matrix(c, camera);
    return v;
}

void CascadedShadows::init(VulkanBase& b)
{
    destroy();
    base = &b;
}

void CascadedShadows::releaseImages()
{
    if (!base) return;
    auto device = base->device;
    for (vk::ImageView& v : layerViews)
    {
        auto view = v;
        if (view) base->deferDelete([device, view]() { device.destroyImageView(view); });
        v = nullptr;
    }
    // Texture::destroy releases the image after the frames in flight
    if (depthArray) depthArray->destroy();
    depthArray.reset();
    imageResolution = imageLayers = 0;
}

void CascadedShadows::destroy()
{
    releaseImages();
    base = nullptr;
}

void CascadedShadows::ensureImages(int count, int resolution)
{
    if (depthArray && imageLayers == count && imageResolution == resolution) return;
    releaseImages();
    depthArray = std::make_unique<Texture2DArray>();
    depthArray->create(*base, uint32_t(resolution), uint32_t(resolution), uint32_t(count), vk::Format::eD32Sfloat,
                       vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled);
    imageResolution = resolution;
    imageLayers     = count;
    for (int c = 0; c < count; ++c)
    {
        vk::ImageViewCreateInfo info({}, depthArray->image(), vk::ImageViewType::e2D, vk::Format::eD32Sfloat, {},
                                     vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, uint32_t(c), 1));
        layerViews[size_t(c)] = base->device.createImageView(info);
    }
    // Every layer lit (1) and readable
    auto& q               = base->mainQueue;
    vk::CommandBuffer cmd = q.commandPool.createAndBeginOneTimeBuffer();
    const vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, uint32_t(count));
    vk::ImageMemoryBarrier2 toDst(Stage::eNone, Access::eNone, Stage::eClear, Access::eTransferWrite,
                                  vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                                  VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, depthArray->image(), range);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toDst));
    cmd.clearDepthStencilImage(depthArray->image(), vk::ImageLayout::eTransferDstOptimal,
                               vk::ClearDepthStencilValue(1.0f, 0), range);
    vk::ImageMemoryBarrier2 toRead(Stage::eClear, Access::eTransferWrite, Stage::eAllCommands,
                                   Access::eShaderSampledRead, vk::ImageLayout::eTransferDstOptimal,
                                   vk::ImageLayout::eShaderReadOnlyOptimal, VK_QUEUE_FAMILY_IGNORED,
                                   VK_QUEUE_FAMILY_IGNORED, depthArray->image(), range);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toRead));
    cmd.end();
    q.submitAndWait(cmd);
    q.commandPool.freeCommandBuffer(cmd);
    depthArray->setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    // The cascades refit (their images are new)
    for (Cascade& c : state) c.valid = false;
}

uint64_t CascadedShadows::memoryBytes() const
{
    return uint64_t(imageResolution) * uint64_t(imageResolution) * uint64_t(imageLayers) * 4;
}

}  // namespace Vulkan
}  // namespace Saiga
