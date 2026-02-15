#include "ops.hpp"
#include <cmath>

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    // Check if shapes match exactly
    if (a->shape == b->shape)
    {
        std::vector<float> result_data(a->data.size());
        for (size_t i = 0; i < a->data.size(); i++)
        {
            result_data[i] = a->data[i] + b->data[i];
        }

        auto result = std::make_shared<Tensor>(std::move(result_data), a->shape);
        result->parents = {a, b};

        // Capture a raw pointer to result's grad vector to avoid a circular shared_ptr reference
        float* res_grad_ptr = result->grad.data();
        size_t data_size = result->data.size();

        result->backward_operation = [res_grad_ptr, a, b, data_size]()
        {
            for (size_t i = 0; i < data_size; i++)
            {
                a->grad[i] += res_grad_ptr[i];
                b->grad[i] += res_grad_ptr[i];
            }
        };

        return result;
    }

    // Broadcasting: b is [1, features] and a is [batch, features]
    if (b->shape.size() == 2 && b->shape[0] == 1 && a->shape.size() == 2 && b->shape[1] == a->shape[1])
    {
        int batch_size = a->shape[0];
        int features = a->shape[1];

        std::vector<float> result_data(a->data.size());
        for (int i = 0; i < batch_size; i++)
        {
            for (int j = 0; j < features; j++)
            {
                result_data[i * features + j] = a->data[i * features + j] + b->data[j];
            }
        }

        auto result = std::make_shared<Tensor>(std::move(result_data), a->shape);
        result->parents = {a, b};

        // Capture the raw pointer to the gradient data and the dimensions
        float* res_grad_ptr = result->grad.data();

        result->backward_operation = [res_grad_ptr, a, b, batch_size, features]()
        {
            for (int i = 0; i < batch_size; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    // Access using the raw pointer
                    float grad_val = res_grad_ptr[i * features + j];
                    a->grad[i * features + j] += grad_val;
                    b->grad[j] += grad_val;
                }
            }
        };

        return result;
    }

    throw std::runtime_error("shape mismatch in add");
}

std::shared_ptr<Tensor> sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b)
{
    // Check if shapes match exactly
    if (a->shape == b->shape)
    {
        std::vector<float> result_data(a->data.size());
        for (size_t i = 0; i < a->data.size(); i++)
        {
            result_data[i] = a->data[i] - b->data[i];
        }

        auto result = std::make_shared<Tensor>(std::move(result_data), a->shape);
        result->parents = {a, b};

        // Capture a raw pointer to result's grad vector to avoid a circular shared_ptr reference
        float* res_grad_ptr = result->grad.data();
        size_t data_size = result->data.size();

        result->backward_operation = [res_grad_ptr, a, b, data_size]()
        {
            for (size_t i = 0; i < data_size; i++)
            {
                a->grad[i] += res_grad_ptr[i];
                b->grad[i] -= res_grad_ptr[i];
            }
        };

        return result;
    }

    // Broadcasting: b is [1, features] and a is [batch, features]
    if (b->shape.size() == 2 && b->shape[0] == 1 && a->shape.size() == 2 && b->shape[1] == a->shape[1])
    {
        int batch_size = a->shape[0];
        int features = a->shape[1];

        std::vector<float> result_data(a->data.size());
        for (int i = 0; i < batch_size; i++)
        {
            for (int j = 0; j < features; j++)
            {
                result_data[i * features + j] = a->data[i * features + j] - b->data[j];
            }
        }

        auto result = std::make_shared<Tensor>(std::move(result_data), a->shape);
        result->parents = {a, b};

        // Capture the raw pointer to the gradient data and the dimensions
        float* res_grad_ptr = result->grad.data();

        result->backward_operation = [res_grad_ptr, a, b, batch_size, features]()
        {
            for (int i = 0; i < batch_size; i++) {
                for (int j = 0; j < features; j++) {
                    int idx = i * features + j;
                    float grad_val = res_grad_ptr[idx];
                    a->grad[idx] += grad_val;
                    b->grad[j] -= grad_val; 
                }
            }
        };

        return result;
    }

    throw std::runtime_error("shape mismatch in sub");
}

std::shared_ptr<Tensor> mult(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b)
{
    // Check if shapes match exactly
    if (a->shape == b->shape)
    {
        std::vector<float> result_data(a->data.size());
        for (size_t i = 0; i < a->data.size(); i++)
        {
            result_data[i] = a->data[i] * b->data[i];
        }

        auto result = std::make_shared<Tensor>(std::move(result_data), a->shape);
        result->parents = {a, b};

        // Capture a raw pointer to result's grad vector to avoid a circular shared_ptr reference
        float* res_grad_ptr = result->grad.data();
        size_t data_size = result->data.size();

        result->backward_operation = [res_grad_ptr, a, b, data_size]()
        {
            for (size_t i = 0; i < data_size; i++)
            {
                a->grad[i] += res_grad_ptr[i] * b->data[i];
                b->grad[i] += res_grad_ptr[i] * a->data[i];
            }
        };

        return result;
    }

    // Broadcasting: b is [1, features] and a is [batch, features]
    if (b->shape.size() == 2 && b->shape[0] == 1 && a->shape.size() == 2 && b->shape[1] == a->shape[1])
    {
        int batch_size = a->shape[0];
        int features = a->shape[1];

        std::vector<float> result_data(a->data.size());
        for (int i = 0; i < batch_size; i++)
        {
            for (int j = 0; j < features; j++)
            {
                result_data[i * features + j] = a->data[i * features + j] * b->data[j];
            }
        }

        auto result = std::make_shared<Tensor>(std::move(result_data), a->shape);
        result->parents = {a, b};

        // Capture the raw pointer to the gradient data and the dimensions
        float* res_grad_ptr = result->grad.data();

        result->backward_operation = [res_grad_ptr, a, b, batch_size, features]()
        {
            for (int i = 0; i < batch_size; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    // Access using the raw pointer
                    int idx = i * features + j;
                    float grad_val = res_grad_ptr[idx];
                    a->grad[idx] += grad_val * b->data[j];
                    b->grad[j] += grad_val * a->data[idx];
                }
            }
        };

        return result;
    }

    throw std::runtime_error("shape mismatch in mult");
}

std::shared_ptr<Tensor> div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    // Check if shapes match exactly
    if (a->shape == b->shape)
    {
        std::vector<float> result_data(a->data.size());
        for (size_t i = 0; i < a->data.size(); i++)
        {
            if (std::abs(b->data[i]) < 1e-8f)
            {
                throw std::runtime_error("division by zero");
            }
            result_data[i] = a->data[i] / b->data[i];
        }

        auto result = std::make_shared<Tensor>(std::move(result_data), a->shape);
        result->parents = {a, b};

        // Capture a raw pointer to result's grad vector to avoid a circular shared_ptr reference
        float* res_grad_ptr = result->grad.data();
        size_t data_size = result->data.size();

        result->backward_operation = [res_grad_ptr, a, b, data_size]()
        {
            for (size_t i = 0; i < data_size; i++)
            {
                a->grad[i] += res_grad_ptr[i] / b->data[i];
                b->grad[i] -= res_grad_ptr[i] * a->data[i] / (b->data[i] * b->data[i]);
            }
        };

        return result;
    }

    // Broadcasting: b is [1, features] and a is [batch, features]
    if (b->shape.size() == 2 && b->shape[0] == 1 && a->shape.size() == 2 && b->shape[1] == a->shape[1])
    {
        int batch_size = a->shape[0];
        int features = a->shape[1];

        std::vector<float> result_data(a->data.size());
        for (int i = 0; i < batch_size; i++)
        {
            for (int j = 0; j < features; j++)
            {
                if (std::abs(b->data[j]) < 1e-8f)
                {
                    throw std::runtime_error("division by zero");
                }
                result_data[i * features + j] = a->data[i * features + j] / b->data[j];
            }
        }

        auto result = std::make_shared<Tensor>(std::move(result_data), a->shape);
        result->parents = {a, b};

        // Capture the raw pointer to the gradient data and the dimensions
        float* res_grad_ptr = result->grad.data();

        result->backward_operation = [res_grad_ptr, a, b, batch_size, features]()
        {
            for (int i = 0; i < batch_size; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    // Access using the raw pointer
                    int idx = i * features + j;
                    float grad_val = res_grad_ptr[idx];
                    a->grad[idx] += grad_val / b->data[j];
                    b->grad[j] -= grad_val * a->data[idx] / (b->data[j] * b->data[j]);
                }
            }
        };

        return result;
    }

    throw std::runtime_error("shape mismatch in div");
}

std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b)
{
    if (a->shape.size() != 2 || b->shape.size() != 2)
    {
        throw std::runtime_error("matmul requires 2D tensors");
    }

    int m = a->shape[0];
    int k = a->shape[1];
    int n = b->shape[1];

    if (k != b->shape[0])
    {
        throw std::runtime_error("matmul dimension mismatch");
    }

    std::vector<float> result_data(m * n, 0.0f);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int p = 0; p < k; p++)
            {
                result_data[i * n + j] += a->data[i * k + p] * b->data[p * n + j];
            }
        }
    }

    auto result = std::make_shared<Tensor>(std::move(result_data), std::vector<int>{m, n});
    result->parents = {a, b}; // CRITICAL: Must reference the actual input tensors

    float* res_grad_ptr = result->grad.data();

    result->backward_operation = [res_grad_ptr, a, b, m, k, n]()
    {
        for (int i = 0; i < m; i++) {
            for (int p = 0; p < k; p++) {
                float sum = 0.0f;
                for (int j = 0; j < n; j++) {
                    sum += res_grad_ptr[i * n + j] * b->data[p * n + j];
                }
                a->grad[i * k + p] += sum;
            }
        }

        for (int p = 0; p < k; p++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int i = 0; i < m; i++) {
                    sum += a->data[i * k + p] * res_grad_ptr[i * n + j];
                }
                b->grad[p * n + j] += sum;
            }
        }
    };

    return result;
}

std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a) {
    std::vector<float> res_data(a->data.size());
    for (size_t i = 0; i < a->data.size(); ++i) {
        res_data[i] = std::max(0.0f, a->data[i]);
    }

    auto res = std::make_shared<Tensor>(std::move(res_data), a->shape);
    res->parents = {a};

    auto res_grad_ptr = res->grad.data();
    res->backward_operation = [res_grad_ptr, a]() {
        for (size_t i = 0; i < a->data.size(); ++i) {
            if (a->data[i] > 0) {
                a->grad[i] += res_grad_ptr[i];
            }
            // else grad is 0, so we do nothing
        }
    };
    return res;
}

std::shared_ptr<Tensor> sigmoid(std::shared_ptr<Tensor> a)
{
    std::vector<float> res_data(a->data.size());
    for (size_t i = 0; i < a->data.size(); ++i)
    {
        res_data[i] = 1.0f / (1.0f + exp(-a->data[i]));
    }

    auto res = std::make_shared<Tensor>(std::move(res_data), a->shape);
    res->parents = {a};

    std::vector<float> sigmoid_output = res->data; 
    float* res_grad_ptr = res->grad.data();

    res->backward_operation = [sigmoid_output, res_grad_ptr, a]()
    {
        for (size_t i = 0; i < a->data.size(); ++i)
        {
            float s = sigmoid_output[i];
            // Correct Chain Rule: upstream_grad * s * (1 - s)
            a->grad[i] += res_grad_ptr[i] * s * (1.0f - s); 
        }
    };
    return res;
}

std::shared_ptr<Tensor> conv2d(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> kernel, std::shared_ptr<Tensor> bias, int stride, int padding) {
    int B  = input->shape[0];
    int C  = input->shape[1]; // In_Channels
    int H  = input->shape[2];
    int W  = input->shape[3];
    
    int OC = kernel->shape[0]; // Out_Channels
    int KH = kernel->shape[2];
    int KW = kernel->shape[3];

    int OH = (H + 2 * padding - KH) / stride + 1;
    int OW = (W + 2 * padding - KW) / stride + 1;

    std::vector<float> res_data(B * OC * OH * OW, 0.0f);

    // --- FORWARD PASS ---
    for (int b = 0; b < B; ++b) {
        int input_batch_offset = b * C * H * W;
        int res_batch_offset = b * OC * OH * OW;

        for (int oc = 0; oc < OC; ++oc) {
            int res_channel_offset = res_batch_offset + (oc * OH * OW);
            int kernel_oc_offset = oc * C * KH * KW;
            float b_val = bias->data[oc];

            for (int oh = 0; oh < OH; ++oh) {
                int res_row_offset = res_channel_offset + (oh * OW);
                int ih_base = oh * stride - padding;

                for (int ow = 0; ow < OW; ++ow) {
                    int iw_base = ow * stride - padding;
                    float sum = 0.0f;

                    for (int ic = 0; ic < C; ++ic) {
                        int input_ic_offset = input_batch_offset + (ic * H * W);
                        int kernel_ic_offset = kernel_oc_offset + (ic * KH * KW);

                        for (int kh = 0; kh < KH; ++kh) {
                            int ih = ih_base + kh;
                            if (ih < 0 || ih >= H) continue;

                            int input_h_offset = input_ic_offset + (ih * W);
                            int kernel_h_offset = kernel_ic_offset + (kh * KW);

                            for (int kw = 0; kw < KW; ++kw) {
                                int iw = iw_base + kw;
                                if (iw >= 0 && iw < W) {
                                    sum += input->data[input_h_offset + iw] * kernel->data[kernel_h_offset + kw];
                                }
                            }
                        }
                    }
                    res_data[res_row_offset + ow] = sum + b_val;
                }
            }
        }
    }

    auto res = std::make_shared<Tensor>(std::move(res_data), std::vector<int>{B, OC, OH, OW});
    res->parents = {input, kernel, bias};

    float* res_grad_ptr = res->grad.data();

    float* input_data_ptr = input->data.data();
    float* input_grad_ptr = input->grad.data();
    float* kernel_data_ptr = kernel->data.data();
    float* kernel_grad_ptr = kernel->grad.data();
    float* bias_grad_ptr = bias->grad.data();

    res->backward_operation = [res_grad_ptr, input_data_ptr, input_grad_ptr, kernel_data_ptr, kernel_grad_ptr, bias_grad_ptr, 
                            input, kernel, bias, // Capture parents to keep them alive
                            stride, padding, B, C, H, W, OC, KH, KW, OH, OW]() {
        
        for (int b = 0; b < B; ++b) {
            int input_batch_offset = b * C * H * W;
            int res_batch_offset = b * OC * OH * OW;

            for (int oc = 0; oc < OC; ++oc) {
                int res_channel_offset = res_batch_offset + (oc * OH * OW);
                int kernel_oc_offset = oc * C * KH * KW;

                for (int oh = 0; oh < OH; ++oh) {
                    int res_row_offset = res_channel_offset + (oh * OW);
                    int ih_base = oh * stride - padding;

                    for (int ow = 0; ow < OW; ++ow) {
                        float upstream_grad = res_grad_ptr[res_row_offset + ow];
                        int iw_base = ow * stride - padding;

                        // 1. Gradient w.r.t Bias
                        bias_grad_ptr[oc] += upstream_grad;

                        for (int ic = 0; ic < C; ++ic) {
                            int input_ic_offset = input_batch_offset + (ic * H * W);
                            int kernel_ic_offset = kernel_oc_offset + (ic * KH * KW);

                            for (int kh = 0; kh < KH; ++kh) {
                                int ih = ih_base + kh;
                                if (ih < 0 || ih >= H) continue;

                                int input_h_offset = input_ic_offset + (ih * W);
                                int kernel_h_offset = kernel_ic_offset + (kh * KW);

                                for (int kw = 0; kw < KW; ++kw) {
                                    int iw = iw_base + kw;
                                    if (iw >= 0 && iw < W) {
                                        int in_idx = input_h_offset + iw;
                                        int kn_idx = kernel_h_offset + kw;

                                        // 2. Gradient w.r.t Kernel
                                        kernel_grad_ptr[kn_idx] += input_data_ptr[in_idx] * upstream_grad;
                                        
                                        // 3. Gradient w.r.t Input
                                        input_grad_ptr[in_idx] += kernel_data_ptr[kn_idx] * upstream_grad;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    return res;
}

std::shared_ptr<Tensor> maxpool2d(std::shared_ptr<Tensor> a, int pool_size, int stride) {
    int B = a->shape[0];
    int C = a->shape[1];
    int H = a->shape[2];
    int W = a->shape[3];

    int OH = (H - pool_size) / stride + 1;
    int OW = (W - pool_size) / stride + 1;

    std::vector<float> res_data(B * C * OH * OW);
    // need to store argmax for gradient back propagation
    std::vector<int> max_indices(B * C * OH * OW);

    for (int b = 0; b < B; ++b) {
        int batch_offset = b * C * H * W;
        int res_batch_offset = b * C * OH * OW;

        for (int c = 0; c < C; ++c) {
            int channel_offset = batch_offset + (c * H * W);
            int res_channel_offset = res_batch_offset + (c * OH * OW);

            for (int i = 0; i < OH; ++i) {
                int res_row_offset = res_channel_offset + (i * OW);
                int ih_base = i * stride;

                for (int j = 0; j < OW; ++j) {
                    int iw_base = j * stride;
                    
                    float max_val = -1e9f;
                    int max_idx = -1;

                    for (int pi = 0; pi < pool_size; ++pi) {
                        int ih = ih_base + pi;
                        int row_offset = channel_offset + (ih * W);

                        for (int pj = 0; pj < pool_size; ++pj) {
                            int iw = iw_base + pj;
                            int curr_idx = row_offset + iw;
                            
                            if (a->data[curr_idx] > max_val) {
                                max_val = a->data[curr_idx];
                                max_idx = curr_idx;
                            }
                        }
                    }
                    res_data[res_row_offset + j] = max_val;
                    max_indices[res_row_offset + j] = max_idx;
                }
            }
        }
    }

    auto res = std::make_shared<Tensor>(std::move(res_data), std::vector<int>{B, C, OH, OW});
    res->parents = {a};

    // Extract raw pointers
    float* res_grad_ptr = res->grad.data();
    float* a_grad_ptr = a->grad.data();
    size_t res_size = res->grad.size();

    // Capture a, max_indices (by value), and the raw pointers
    res->backward_operation = [res_grad_ptr, a_grad_ptr, a, max_indices, res_size]() {
        for (size_t i = 0; i < res_size; ++i) {
            // The gradient only flows back to the specific pixel that was the maximum
            int original_idx = max_indices[i];
            a_grad_ptr[original_idx] += res_grad_ptr[i];
        }
    };

    return res;
}