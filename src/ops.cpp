#include "ops.hpp"

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
	if (a->shape != b->shape) throw std::runtime_error("shape mismatch while adding");
	int size = a->data.size();
	
	std::vector<float> result_data(size);
	for (int i = 0; i < size; i++){
		result_data[i] = a->data[i] + b->data[i];
	}

	auto res = std::make_shared<Tensor>(std::move(result_data), std::vector<int>(a->shape));

	// for back propagation
	res->parents = {a,b};
	res->backward_operation = [res, a, b](){
		for (size_t i = 0; i < a->data.size(); i++){
			a->grad[i] += res->grad[i];
			b->grad[i] += res->grad[i];
		}
	};
	
	return res;
}

std::shared_ptr<Tensor> sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
	if (a->shape != b->shape) throw std::runtime_error("shape mismatch while subtracting");
	int size = a->data.size();
	
	std::vector<float> result_data(size);
	for (int i = 0; i < size; i++){
		result_data[i] = a->data[i] - b->data[i];
	}
	
	auto res = std::make_shared<Tensor>(std::move(result_data), std::vector<int>(a->shape));
	
	// for back propagation
	res->parents = {a,b};
	res->backward_operation = [res, a, b](){
				for (size_t i = 0; i < a->data.size(); i++){
					a->grad[i] += res->grad[i];
					b->grad[i] -= res->grad[i];
				}
		};

		return res;
	}

std::shared_ptr<Tensor> mult(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
	if (a->shape != b->shape) throw std::runtime_error("shape mismatch while multiplying");
	int size = a->data.size();

	std::vector<float> result_data(size);
	for (int i = 0; i < size; i++){
			result_data[i] = a->data[i] * b->data[i];
	}

	auto res = std::make_shared<Tensor>(std::move(result_data), std::vector<int>(a->shape));

	// for back propagation
	res->parents = {a,b};
	res->backward_operation = [res, a, b](){
	for (size_t i = 0; i < a->data.size(); i++){
			a->grad[i] += b->data[i] * res->grad[i];
			b->grad[i] += a->data[i] * res->grad[i];
		}
	};
	
	return res;
}
		
std::shared_ptr<Tensor> div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
	if (a->shape != b->shape) throw std::runtime_error("shape mismatch while dividing");
	int size = a->data.size();
	
	std::vector<float> result_data(size);
	for (int i = 0; i < size; i++){
		if (b->data[i] == 0) throw std::runtime_error("division by zero");
		result_data[i] = a->data[i] / b->data[i];
	}

	auto res = std::make_shared<Tensor>(std::move(result_data), std::vector<int>(a->shape));
	
	// for back propagation
	res->parents = {a,b};
	res->backward_operation = [res, a, b](){
	for (size_t i = 0; i < a->data.size(); i++){
		a->grad[i] += (1.0f / b->data[i]) * res->grad[i];
			b->grad[i] -= (a->data[i] / (b->data[i] * b->data[i])) * res->grad[i];
		}
	};
	
	return res;
}
			
std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
	if (a->shape.size() != 2 || b->shape.size() != 2) {
		throw std::runtime_error("matmul: Both tensors must be 2D matrices.");
	}
	if (a->shape[1] != b->shape[0]) {
		throw std::runtime_error("matmul: Dimension mismatch. A.cols (" + 
			std::to_string(a->shape[1]) + ") must match B.rows (" + 
								 std::to_string(b->shape[0]) + ").");
	}

	int M = a->shape[0];
	int K = a->shape[1];
	int N = b->shape[1];
	
	std::vector<float> res_data(M * N, 0.0f);
	for (int i = 0; i < M; ++i) {
		for (int k = 0; k < K; ++k) {
			float a_val = a->data[i * K + k];
			for (int j = 0; j < N; ++j) {
				res_data[i * N + j] += a_val * b->data[k * N + j];
			}
		}
	}
	
	auto res = std::make_shared<Tensor>(std::move(res_data), std::vector<int>{M, N});
	
	res->parents = {a, b};
	
	// Derivative of C = A @ B:
	// dL/dA = (dL/dC) @ B^T
	// dL/dB = A^T @ (dL/dC)
	res->backward_operation = [res, a, b, M, K, N]() {
		for (int i = 0; i < M; ++i) {
			for (int k = 0; k < K; ++k) {
				// Pre-calculate index and value for A to minimize lookups
				int a_idx = i * K + k;
				float a_val = a->data[a_idx];
				
				for (int j = 0; j < N; ++j) {
					float grad_out = res->grad[i * N + j];
					
					// Accumulate gradient for A: sum(grad_out * B)
					a->grad[a_idx] += grad_out * b->data[k * N + j];
					
					// Accumulate gradient for B: sum(A * grad_out)
					b->grad[k * N + j] += a_val * grad_out;
				}
			}
		}
	};

	return res;
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