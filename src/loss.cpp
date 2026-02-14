#include "loss.hpp"
#include <stdexcept>

std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target) {
    if (pred->data.size() != target->data.size()) 
        throw std::runtime_error("MSE shape mismatch");
    
    int n = (int)pred->data.size();
    float diff_sum = 0;
    std::vector<float> diffs(n);

    for (int i = 0; i < n; i++) {
        float d = pred->data[i] - target->data[i];
        diffs[i] = d;
        diff_sum += d * d;
    }

    auto res = std::make_shared<Tensor>(std::vector<float>{diff_sum / n}, std::vector<int>{1, 1});
    res->parents = {pred};

    res->backward_operation = [res, pred, n, diffs]() {
        for (int i = 0; i < n; i++) {
            pred->grad[i] += (2.0f / n) * diffs[i] * res->grad[0];
        }
    };
    return res;
}