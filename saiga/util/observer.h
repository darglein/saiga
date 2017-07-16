#pragma once

#include <saiga/config.h>
#include <vector>

namespace Saiga {

class SAIGA_GLOBAL Subject {
    std::vector < class Observer * > views;
  public:
    virtual ~Subject(){}
    void attach(Observer *obs) {
        views.push_back(obs);
    }
    void notify();
};

class SAIGA_GLOBAL Observer {
    Subject *model;
  public:
    Observer() {
    }
    virtual ~Observer(){}
    virtual void notify() = 0;
    void setSubject(Subject *_model){
    this->model = _model;
    model->attach(this);
}

  protected:
    Subject *getSubject() {
        return model;
    }
};

}
