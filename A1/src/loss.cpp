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

    float* res_grad_ptr = res->grad.data();

    res->backward_operation = [res_grad_ptr, pred, n, diffs]() {
        float upstream_grad = res_grad_ptr[0];
        
        for (int i = 0; i < n; i++) {
            pred->grad[i] += (2.0f / n) * diffs[i] * upstream_grad;
        }
    };
    return res;
}

std::shared_ptr<Tensor> cross_entropy_loss(std::shared_ptr<Tensor> logits, const std::vector<int>& targets) {
    int batch_size = logits->shape[0];
    int features = logits->shape[1];
    
    std::vector<float> probs(logits->data.size());
    float total_loss = 0.0f;

    // compute softmax probabilities and cross entropy simultaneously
    for (int b = 0; b < batch_size; b++) {
        float max_val = -1e9;
        for (int f = 0; f < features; f++) 
            max_val = std::max(max_val, logits->data[b * features + f]);

        float sum = 0.0f;
        for (int f = 0; f < features; f++) {
            probs[b * features + f] = std::exp(logits->data[b * features + f] - max_val);
            sum += probs[b * features + f];
        }
        
        for (int f = 0; f < features; f++) {
            probs[b * features + f] /= (sum + 1e-9f);
        }

        total_loss -= std::log(probs[b * features + targets[b]] + 1e-9f);
    }

    auto res = std::make_shared<Tensor>(std::vector<float>{total_loss / batch_size}, std::vector<int>{1, 1});
    res->parents = {logits};

    float* res_grad_ptr = res->grad.data();
    
    res->backward_operation = [res_grad_ptr, logits, probs, targets, batch_size, features]() {
        float upstream_grad = res_grad_ptr[0]; 
        
        for (int b = 0; b < batch_size; b++) {
            for (int f = 0; f < features; f++) {
                float p = probs[b * features + f];
                float y = (f == targets[b]) ? 1.0f : 0.0f;
                
                // Gradient is (Prob - Target) / BatchSize
                logits->grad[b * features + f] += (p - y) * upstream_grad / batch_size;
            }
        }
    };
    
    return res;
}