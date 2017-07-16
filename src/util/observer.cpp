#include "saiga/util/observer.h"

namespace Saiga {

void Subject::notify() {
  for (Observer* &view : views)
    view->notify();
}

}
