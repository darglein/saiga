#include <mutex>
#include <condition_variable>

class Semaphore {
public:
    Semaphore (int count_ = 0)
        : count(count_) {}

    inline void notify()
    {
        std::unique_lock<std::mutex> lock(mtx);
        count++;
        cv.notify_one();
    }

    inline void wait()
    {
        std::unique_lock<std::mutex> lock(mtx);

        while(count == 0){
            cv.wait(lock);
        }
        count--;
    }

    inline bool trywait()
    {
        std::unique_lock<std::mutex> lock(mtx);
        if(count)
        {
            --count;
            return true;
        }
        else
        {
            return false;
        }
    }

private:
    std::mutex mtx;
    std::condition_variable cv;
    int count;
};
