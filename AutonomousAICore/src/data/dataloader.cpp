// src/data/dataloader.cpp
// DataLoader implementation for batch assembly and device placement.
#include "data/dataloader.hpp"
#include "core/error.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>

namespace aic::data {

DataLoader::DataLoader(
    std::shared_ptr<BinaryTokenDataset> dataset,
    int64_t batch_size,
    bool shuffle,
    bool drop_last,
    Device device,
    uint64_t seed
) : dataset_(std::move(dataset)),
    batch_size_(batch_size),
    shuffle_(shuffle),
    drop_last_(drop_last),
    device_(device),
    rng_(seed) {
    AIC_CHECK(dataset_ != nullptr, InvalidArgument, "DataLoader requires a valid dataset");
    AIC_CHECK(batch_size_ > 0, InvalidArgument, "batch_size must be positive, got ", batch_size_);

    int64_t n = dataset_->size();
    indices_.resize(static_cast<size_t>(n));
    std::iota(indices_.begin(), indices_.end(), 0);

    if (drop_last_) {
        num_batches_ = n / batch_size_;
    } else {
        num_batches_ = (n + batch_size_ - 1) / batch_size_;
    }

    reset();
}

void DataLoader::reset() {
    current_batch_ = 0;
    if (shuffle_ && !indices_.empty()) {
        std::shuffle(indices_.begin(), indices_.end(), rng_);
    }
}

std::pair<Tensor, Tensor> DataLoader::next() {
    AIC_CHECK(has_next(), OutOfRange, "DataLoader reached end of epoch (current_batch = ",
              current_batch_, ", num_batches = ", num_batches_, ")");

    int64_t start_idx = current_batch_ * batch_size_;
    int64_t actual_batch = std::min(batch_size_, dataset_->size() - start_idx);
    int64_t seq_len = dataset_->seq_len();

    Tensor batch_inputs = Tensor::empty(Shape({actual_batch, seq_len}), DType::Int64, Device::cpu());
    Tensor batch_targets = Tensor::empty(Shape({actual_batch, seq_len}), DType::Int64, Device::cpu());

    int64_t* in_ptr = batch_inputs.data_ptr<int64_t>();
    int64_t* tgt_ptr = batch_targets.data_ptr<int64_t>();

    for (int64_t b = 0; b < actual_batch; ++b) {
        int64_t sample_idx = indices_[static_cast<size_t>(start_idx + b)];
        auto [sample_in, sample_tgt] = dataset_->get_item(sample_idx);

        std::memcpy(in_ptr + b * seq_len, sample_in.data_ptr<int64_t>(),
                    static_cast<size_t>(seq_len) * sizeof(int64_t));
        std::memcpy(tgt_ptr + b * seq_len, sample_tgt.data_ptr<int64_t>(),
                    static_cast<size_t>(seq_len) * sizeof(int64_t));
    }

    current_batch_++;

    if (device_.is_cuda()) {
        return {batch_inputs.to(device_), batch_targets.to(device_)};
    }
    return {batch_inputs, batch_targets};
}

}  // namespace aic::data
