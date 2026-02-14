#include "init.hpp"
#include <random>
#include <cmath>

static std::mt19937 *global_gen = nullptr;

void set_random_seed(int seed)
{
    if (global_gen)
        delete global_gen;
    global_gen = new std::mt19937(seed);
}

std::shared_ptr<Tensor> xavier_init(std::vector<int> shape) {
    if (!global_gen)
    {
        global_gen = new std::mt19937(42); // Default seed
    }

    int fan_in = shape[0];
    int fan_out = (shape.size() > 1) ? shape[1] : 1;
    float range = std::sqrt(6.0f / (fan_in + fan_out));

    std::uniform_real_distribution<float> dist(-range, range);

    int total_size = 1;
    for (int s : shape) total_size *= s;

    std::vector<float> data(total_size);
    for (int i = 0; i < total_size; i++)
    {
        data[i] = dist(*global_gen);
    }

    return std::make_shared<Tensor>(std::move(data), shape);
}