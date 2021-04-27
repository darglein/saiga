/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <vector>

namespace Saiga
{
class SAIGA_CORE_API Subject
{
    std::vector<class Observer*> views;

   public:
    virtual ~Subject() {}
    void attach(Observer* obs) { views.push_back(obs); }
    void notify();
};

class SAIGA_CORE_API Observer
{
    Subject* model;

   public:
    Observer() {}
    virtual ~Observer() {}
    virtual void notify() = 0;
    void setSubject(Subject* _model)
    {
        this->model = _model;
        model->attach(this);
    }

   protected:
    Subject* getSubject() { return model; }
};

}  // namespace Saiga
