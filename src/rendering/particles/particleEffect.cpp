#include "libhello/rendering/particles/particleEffect.h"

void ImpactParticleEffect::apply(ParticleSystem &ps)
{
    for(int i=0;i<10;++i){
        Particle p;
        p.position = vec3(this->getPosition());
        p.velocity = sampleCone(coneDirection,glm::radians(coneAngle));
        p.color = color;
        p.radius = radius;
        p.lifetime = lifetime;
        p.image = rand()%4;
        ps.addParticle(p);
    }
    ps.flush();
}
