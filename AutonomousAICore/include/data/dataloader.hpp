// include/data/dataloader.hpp
// Batched, shuffled data loader for sequence datasets with device transfer.
#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>
#include "data/dataset.hpp"
#include "core/device.hpp"
#include "tensor/tensor.hpp"

namespace aic::data {

class DataLoader {
public:
    DataLoader(
        std::shared_ptr<BinaryTokenDataset> dataset,
        int64_t batch_size,
        bool shuffle = false,
        bool drop_last = false,
        Device device = Device::cpu(),
        uint64_t seed = 42
    );

    int64_t batch_size() const noexcept { return batch_size_; }
    int64_t num_batches() const noexcept { return num_batches_; }
    Device device() const noexcept { return device_; }

    void reset();
    bool has_next() const noexcept { return current_batch_ < num_batches_; }

    // Returns a batch pair of (inputs, targets) of shape [BatchSize, SeqLen] on target device
    std::pair<Tensor, Tensor> next();

private:
    std::shared_ptr<BinaryTokenDataset> dataset_;
    int64_t batch_size_;
    bool shuffle_;
    bool drop_last_;
    Device device_;
    int64_t num_batches_ = 0;
    int64_t current_batch_ = 0;

    std::vector<int64_t> indices_;
    std::mt19937_64 rng_;
};

}  // namespace aic::data
