#pragma once

#include <torch/torch.h>


// This scheduler reduces the learning rate by 'factor' once the loss hasn't improved for 'patience' amount of
// iterations.
class LRSchedulerPlateau
{
   public:
    LRSchedulerPlateau(double reduce_factor = 0.5, int patience = 5) : reduce_factor(reduce_factor), patience(patience)
    {
    }


    // Returns the LR-reduction factor based on the passed loss value
    // This factor should then be multiplied to the optimizer's current learning rate.
    double step(double value)
    {
        if (best_loss.has_value())
        {
            if (value < best_loss.value())
            {
                best_loss                  = value;
                _num_epochs_no_improvement = 0;
            }
            else
            {
                _num_epochs_no_improvement++;
            }
        }else
        {
            best_loss = value;
            return 1;
        }

        if (_num_epochs_no_improvement == patience)
        {
            _num_epochs_no_improvement = 0;
            return reduce_factor;
        }
        return 1;
    }

   private:
    double reduce_factor;


    int patience;

    int _num_epochs_no_improvement = 0;
    std::optional<double> best_loss;
};
