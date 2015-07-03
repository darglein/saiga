#pragma once

#include <libhello/config.h>

template <typename C>
 class SAIGA_GLOBAL Singleton
 {
 public:
    static C* instance ()
    {
//       if (!_instance)
//          _instance = new C ();
        static C _instance;
       return &_instance;
    }
    virtual
    ~Singleton ()
    {
//       _instance = nullptr;
    }
 private:

 protected:
    Singleton () { }
 };
// template <typename C> C Singleton <C>::_instance;
