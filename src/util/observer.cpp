#include "libhello/util/observer.h"

void Subject::notify() {
  for (Observer* &view : views)
    view->notify();
}
