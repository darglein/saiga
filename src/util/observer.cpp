/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/observer.h"

namespace Saiga {

void Subject::notify() {
  for (Observer* &view : views)
    view->notify();
}

}
