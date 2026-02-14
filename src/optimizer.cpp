#include "optimizer.hpp"
#include <algorithm>

SGD::SGD(std::vector<std::shared_ptr<Tensor>> p, float learning_rate) 
    : params(p), lr(learning_rate) {}

void SGD::step() {
    for (auto& p : params) {
        for (size_t i = 0; i < p->data.size(); i++) {
            p->data[i] -= lr * p->grad[i];
        }
    }
}

void SGD::zero_grad() {
    for (auto& p : params) {
        std::fill(p->grad.begin(), p->grad.end(), 0.0f);
    }
}