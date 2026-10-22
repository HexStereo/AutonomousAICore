// include/nn/module.hpp
// Base class for all neural network modules in AutonomousAICore.
#pragma once

#include "tensor/tensor.hpp"
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace aic::nn {

class Module : public std::enable_shared_from_this<Module> {
public:
    Module() : training_(true) {}
    virtual ~Module() = default;

    virtual void train(bool mode = true) {
        training_ = mode;
        for (auto& [name, mod] : submodules_) {
            if (mod) mod->train(mode);
        }
    }

    virtual void eval() {
        train(false);
    }

    bool is_training() const noexcept { return training_; }

    // Register a learnable parameter
    std::shared_ptr<Tensor> register_parameter(const std::string& name, Tensor param) {
        param.set_requires_grad(true);
        auto ptr = std::make_shared<Tensor>(std::move(param));
        parameters_[name] = ptr;
        return ptr;
    }

    // Register a sub-module
    template <typename T>
    std::shared_ptr<T> register_module(const std::string& name, std::shared_ptr<T> module) {
        submodules_[name] = module;
        return module;
    }

    // Retrieve all parameters recursively
    std::vector<std::shared_ptr<Tensor>> parameters() const {
        std::vector<std::shared_ptr<Tensor>> params;
        for (const auto& [name, p] : parameters_) {
            params.push_back(p);
        }
        for (const auto& [name, mod] : submodules_) {
            if (mod) {
                auto sub_params = mod->parameters();
                params.insert(params.end(), sub_params.begin(), sub_params.end());
            }
        }
        return params;
    }

    // Retrieve named parameters recursively
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> named_parameters(const std::string& prefix = "") const {
        std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> result;
        for (const auto& [name, p] : parameters_) {
            std::string full_name = prefix.empty() ? name : (prefix + "." + name);
            result.emplace_back(full_name, p);
        }
        for (const auto& [name, mod] : submodules_) {
            if (mod) {
                std::string next_prefix = prefix.empty() ? name : (prefix + "." + name);
                auto sub_named = mod->named_parameters(next_prefix);
                result.insert(result.end(), sub_named.begin(), sub_named.end());
            }
        }
        return result;
    }

    // Zero out gradients on all parameters recursively
    void zero_grad() {
        for (auto& p : parameters()) {
            p->zero_grad();
        }
    }

    // Move all parameters and submodules to target device
    virtual void to(const Device& device) {
        for (auto& [name, p] : parameters_) {
            bool req = p->requires_grad();
            *p = p->to(device);
            p->set_requires_grad(req);
        }
        for (auto& [name, mod] : submodules_) {
            if (mod) mod->to(device);
        }
    }

    virtual Tensor forward(const Tensor& input) {
        return input;
    }

protected:
    bool training_{true};
    std::map<std::string, std::shared_ptr<Tensor>> parameters_;
    std::map<std::string, std::shared_ptr<Module>> submodules_;
};

}  // namespace aic::nn
