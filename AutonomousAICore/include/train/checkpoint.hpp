// include/train/checkpoint.hpp
// Custom binary checkpoint format (.aic) with tensor dictionary and checksum verification.
#pragma once

#include <cstdint>
#include <string>
#include "nn/module.hpp"
#include "optim/optimizer.hpp"

namespace aic::train {

constexpr uint32_t AIC_CHECKPOINT_MAGIC = 0x41494330; // 'AIC0'
constexpr uint32_t AIC_CHECKPOINT_VERSION = 1;

struct CheckpointMetadata {
    uint32_t magic = AIC_CHECKPOINT_MAGIC;
    uint32_t version = AIC_CHECKPOINT_VERSION;
    int64_t step = 0;
    int64_t epoch = 0;
    float best_val_loss = 0.0f;
    std::string description;
};

// Saves model parameters (and optional optimizer state) to custom binary format (.aic)
void save_checkpoint(
    const std::string& filepath,
    const nn::Module& model,
    const optim::Optimizer* opt = nullptr,
    const CheckpointMetadata& meta = {}
);

// Loads model parameters from custom binary format (.aic) with integrity verification
CheckpointMetadata load_checkpoint(
    const std::string& filepath,
    nn::Module& model,
    optim::Optimizer* opt = nullptr
);

}  // namespace aic::train
