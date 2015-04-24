#pragma once

#include <vector>


class Subject {
    std::vector < class Observer * > views;
  public:
    virtual ~Subject(){}
    void attach(Observer *obs) {
        views.push_back(obs);
    }
    void notify();
};

class Observer {
    Subject *model;
  public:
    Observer() {
    }
    virtual ~Observer(){}
    virtual void notify() = 0;
    void setSubject(Subject *model){
    this->model = model;
    model->attach(this);
}

  protected:
    Subject *getSubject() {
        return model;
    }
};
