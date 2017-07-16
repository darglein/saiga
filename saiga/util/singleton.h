#pragma once

#include <saiga/config.h>

namespace Saiga {

template <typename C>
class SAIGA_GLOBAL Singleton
{
public:
    static C* instance ()
    {
        static C _instance;
        return &_instance;
    }
    virtual
    ~Singleton ()
    {

    }
private:

protected:
    Singleton () { }
};

}
