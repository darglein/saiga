/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/config.h"
#include "saiga/core/math/imath.h"
#include "saiga/core/time/all.h"
#include "saiga/core/util/Thread/SpinLock.h"
#include "saiga/core/util/Thread/threadName.h"
#include "saiga/core/util/assert.h"
#include "saiga/core/util/tostring.h"

#include <atomic>
#include <iostream>
#include <mutex>
#include <string>

#include <condition_variable>
namespace Saiga
{

struct ProgressBarBase
{
   public:
    ProgressBarBase(int64_t _end, std::string prefix) : end(_end), prefix(prefix) {}
    virtual ~ProgressBarBase() {}

    void addProgress(int64_t i) { current += i; }


    void SetPostfix(const std::string& str)
    {
        std::unique_lock l(lock);
        postfix = str;
    }
    int64_t end;
    std::atomic_int64_t current = 0;

    std::string prefix;
    std::string postfix;
    std::mutex lock;
};


/**
 * A synchronized progress bar for console output.
 * You must not write to the given stream while the progress bar is active.
 *
 * Usage Parallel Image loading:
 *
 * ProgressBar loadingBar(std::cout, "Loading " + to_string(N) + " images ", N);
 * #pragma omp parallel for
 * for (int i = 0; i < N; ++i)
 * {
 *     images[i].load("...");
 *     loadingBar.addProgress(1);
 * }
 *
 */
struct ProgressBar : public ProgressBarBase
{
    ProgressBar(std::ostream& strm, const std::string header, int64_t end, int length = 30,
                bool show_remaining_time = false, int update_time_ms = 100, std::string element_name = "e",
                float element_factor = 1)
        : ProgressBarBase(end, header),
          strm(strm),
          length(length),
          show_remaining_time(show_remaining_time),
          update_time_ms(update_time_ms),
          element_name(element_name),
          element_factor(element_factor)
    {
        SAIGA_ASSERT(end >= 0);
        timer.start();
        print();
        if (end > 0)
        {
            run();
        }
    }

    ~ProgressBar() { Quit(); }


    void Quit()
    {
        running = false;
        cv.notify_one();
        if (st.joinable())
        {
            st.join();
        }
    }

   private:
    TimerBase timer;
    std::ostream& strm;
    ScopedThread st;
    std::atomic_bool running = true;
    std::condition_variable cv;

    int length;
    bool show_remaining_time;
    int update_time_ms;
    std::string element_name;
    float element_factor;

    void run()
    {
        st = ScopedThread(
            [this]()
            {
                while (running && current.load() < end)
                {
                    print();
                    std::unique_lock<std::mutex> l(lock);
                    cv.wait_for(l, std::chrono::milliseconds(update_time_ms));
                }
                print();
                strm << std::endl;
            });
    }

    void print()
    {
        auto f = strm.flags();


        //        SAIGA_ASSERT(current <= end);
        double progress  = end == 0 ? 0 : double(current) / end;
        auto time        = timer.stop();
        int progress_pro = iRound(progress * 100);
        int barLength    = progress * length;

        strm << "\r" << prefix << " ";

        strm << std::setw(3) << progress_pro << "%";

        {
            // bar
            strm << " |";
            for (auto i = 0; i < barLength; ++i)
            {
                strm << "#";
            }
            for (auto i = barLength; i < length; ++i)
            {
                strm << " ";
            }
            strm << "| ";
        }


        {
            // element count
            auto end_str = to_string(end);
            strm << std::setw(end_str.size()) << current << "/" << end << " ";
        }


        {
            // Time
            strm << "[" << DurationToString(time);

            if (show_remaining_time)
            {
                auto remaining_time = time * (1 / progress) - time;
                strm << "<" << DurationToString(remaining_time);
            }
            strm << "] ";
        }

        {
            // performance stats
            double elapsed_time   = std::chrono::duration_cast<std::chrono::duration<double>>(time).count();
            double ele_per_second = elapsed_time > 0 ? current * element_factor / elapsed_time : 0;
            strm << "[" << std::setprecision(2) << std::fixed << ele_per_second << " " << element_name << "/s]";
        }

        {
            std::unique_lock l(lock);
            strm << " " << postfix;
        }
        strm << std::flush;
        strm << std::setprecision(6);
        strm.flags(f);
    }
};


class SAIGA_CORE_API ProgressBarManager
{
   public:
    ProgressBarManager() {}

    std::shared_ptr<ProgressBarBase> ScopedProgressBar(std::string name, int64_t end);

    bool Imgui();


    std::shared_ptr<ProgressBarBase> current_bar;
    std::mutex lock;
};


#define SAIGA_OPTIONAL_PROGRESS_BAR(progress_bar_pointer, name, end) \
    (progress_bar_pointer) ? progress_bar_pointer->ScopedProgressBar(name, end) : nullptr

}  // namespace Saiga
