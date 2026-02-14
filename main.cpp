#include <iostream>
#include <vector>
#include <memory>
#include <cassert>
#include "linear.h"

void print_tensor(const string& name, shared_ptr<Tensor> t) {
    cout << name << " (data): ";
    for (float val : t->data) cout << val << " ";
    cout << "\n" << name << " (grad): ";
    for (float val : t->grad) cout << val << " ";
    cout << "\n\n";
}

int main() {
    // 1. XOR Data: {x1, x2} -> {y}
    struct Sample { std::vector<float> x; std::vector<float> y; };
    std::vector<Sample> dataset = {
        {{0, 0}, {0}},
        {{0, 1}, {1}},
        {{1, 0}, {1}},
        {{1, 1}, {0}}
    };

    // 2. Define Two Layers
    Linear layer1(2, 2); // Hidden layer
    Linear layer2(2, 1); // Output layer

    // 3. Collect all parameters for the optimizer
    std::vector<std::shared_ptr<Tensor>> params = layer1.parameters();
    std::vector<std::shared_ptr<Tensor>> p2 = layer2.parameters();
    params.insert(params.end(), p2.begin(), p2.end());

    SGD optimizer(params, 0.1f); // Learning rate 0.1

    std::cout << "Training XOR Network..." << std::endl;

    for (int epoch = 0; epoch < 2000; ++epoch) {
        float total_loss = 0;
        for (auto& sample : dataset) {
            optimizer.zero_grad();

            // Forward Pass
            auto input = std::make_shared<Tensor>(sample.x, std::vector<int>{1, 2});
            auto target = std::make_shared<Tensor>(sample.y, std::vector<int>{1, 1});

            auto h1 = layer1.forward(input);
            auto a1 = relu(h1);           // Non-linearity!
            auto out = layer2.forward(a1);

            auto loss = mse_loss(out, target);
            total_loss += loss->data[0];

            // Backward & Step
            loss->backward();
            optimizer.step();
        }

        if (epoch % 200 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << total_loss / 4 << std::endl;
        }
    }

    // 4. Final Testing
    std::cout << "\nResults:" << std::endl;
    for (auto& sample : dataset) {
        auto input = std::make_shared<Tensor>(sample.x, std::vector<int>{1, 2});
        auto out = layer2.forward(relu(layer1.forward(input)));
        std::cout << sample.x[0] << " XOR " << sample.x[1] << " = " << out->data[0] << std::endl;
    }

    return 0;
}