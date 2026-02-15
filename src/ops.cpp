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

        result->backward_operation = [result, a, b]()
        {
            for (size_t i = 0; i < result->data.size(); i++)
            {
                a->grad[i] += result->grad[i];
                b->grad[i] += result->grad[i];
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

        result->backward_operation = [result, a, b, batch_size, features]()
        {
            for (int i = 0; i < batch_size; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    a->grad[i * features + j] += result->grad[i * features + j];
                    b->grad[j] += result->grad[i * features + j];
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

        result->backward_operation = [result, a, b]()
        {
            for (size_t i = 0; i < result->data.size(); i++)
            {
                a->grad[i] += result->grad[i];
                b->grad[i] -= result->grad[i];
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

        result->backward_operation = [result, a, b, batch_size, features]()
        {
            for (int i = 0; i < batch_size; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    a->grad[i * features + j] += result->grad[i * features + j];
                    b->grad[j] -= result->grad[i * features + j];
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

        result->backward_operation = [result, a, b]()
        {
            for (size_t i = 0; i < result->data.size(); i++)
            {
                a->grad[i] += result->grad[i] * b->data[i];
                b->grad[i] += result->grad[i] * a->data[i];
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

        result->backward_operation = [result, a, b, batch_size, features]()
        {
            for (int i = 0; i < batch_size; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    int idx = i * features + j;
                    a->grad[idx] += result->grad[idx] * b->data[j];
                    b->grad[j] += result->grad[idx] * a->data[idx];
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

        result->backward_operation = [result, a, b]()
        {
            for (size_t i = 0; i < result->data.size(); i++)
            {
                a->grad[i] += result->grad[i] / b->data[i];
                b->grad[i] -= result->grad[i] * a->data[i] / (b->data[i] * b->data[i]);
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

        result->backward_operation = [result, a, b, batch_size, features]()
        {
            for (int i = 0; i < batch_size; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    int idx = i * features + j;
                    a->grad[idx] += result->grad[idx] / b->data[j];
                    b->grad[j] -= result->grad[idx] * a->data[idx] / (b->data[j] * b->data[j]);
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

    result->backward_operation = [result, a, b, m, k, n]()
    {
        for (int i = 0; i < m; i++)
        {
            for (int p = 0; p < k; p++)
            {
                for (int j = 0; j < n; j++)
                {
                    a->grad[i * k + p] += result->grad[i * n + j] * b->data[p * n + j];
                }
            }
        }

        for (int p = 0; p < k; p++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int i = 0; i < m; i++)
                {
                    b->grad[p * n + j] += a->data[i * k + p] * result->grad[i * n + j];
                }
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
    res->backward_operation = [res, a]() {
        for (size_t i = 0; i < a->data.size(); ++i) {
            if (a->data[i] > 0) {
                a->grad[i] += res->grad[i];
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
    res->backward_operation = [res, a]()
    {
        for (size_t i = 0; i < a->data.size(); ++i)
        {
            a->grad[i] += res->grad[i] * res->data[i] * (1.0f - res->data[i]);
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

    // --- BACKWARD PASS ---
    res->backward_operation = [res, input, kernel, bias, stride, padding, B, C, H, W, OC, KH, KW, OH, OW]() {
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
                        float upstream_grad = res->grad[res_row_offset + ow];
                        int iw_base = ow * stride - padding;

                        // Gradient w.r.t Bias
                        bias->grad[oc] += upstream_grad;

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

                                        kernel->grad[kn_idx] += input->data[in_idx] * upstream_grad;
                                        input->grad[in_idx] += kernel->data[kn_idx] * upstream_grad;
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