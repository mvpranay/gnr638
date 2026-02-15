#include "init.hpp"
#include <random>
#include <cmath>

static std::unique_ptr<std::mt19937> global_gen;

void set_random_seed(int seed) {
    if (!global_gen) global_gen = std::make_unique<std::mt19937>(seed);
    else global_gen->seed(seed);
}

std::shared_ptr<Tensor> xavier_init(std::vector<int> shape) {
    if (!global_gen) {
        global_gen = std::make_unique<std::mt19937>(42);
    }

    int shape_size = shape.size();
    float fan_in, fan_out;
    if (shape_size == 2) { // Linear: [in, out]
        fan_in = (float)shape[0];
        fan_out = (float)shape[1];
    } 
    else if (shape_size == 4) { // Conv2D: [out, in, kh, kw]
        float receptive_field = (float)(shape[2] * shape[3]); // kh * kw
        fan_in = (float)shape[1] * receptive_field;          // in_channels * kh * kw
        fan_out = (float)shape[0] * receptive_field;         // out_channels * kh * kw
    } 
    else { // Fallback for other dimensions
        fan_in = (float)shape[0];
        fan_out = (shape_size > 1) ? (float)shape[1] : 1.0f;
    }

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