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