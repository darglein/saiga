
#include <glbinding/Binding.h>
#include <glbinding/State.h>


namespace glbinding
{
void State::resolve(const char* name)
{
    if (m_initialized)
    {
        return;
    }

    m_address     = Binding::resolveFunction(name);
    m_initialized = true;
}


}  // namespace glbinding
