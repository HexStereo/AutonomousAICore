// src/train/checkpoint.cpp
// Binary checkpoint format (.aic) implementation with Adler-32 integrity validation.
#include "train/checkpoint.hpp"
#include "core/error.hpp"

#if AIC_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <cstring>
#include <fstream>
#include <unordered_map>
#include <vector>

namespace aic::train {

namespace {

uint32_t compute_adler32(const uint8_t* data, size_t len) {
    const uint32_t MOD_ADLER = 65521;
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < len; ++i) {
        a = (a + data[i]) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }
    return (b << 16) | a;
}

}  // namespace

void save_checkpoint(
    const std::string& filepath,
    const nn::Module& model,
    const optim::Optimizer* /*opt*/,
    const CheckpointMetadata& meta
) {
    std::ofstream out(filepath, std::ios::binary);
    AIC_CHECK(out.is_open(), IOError, "Failed to open checkpoint file for writing: ", filepath);

    // 1. Header
    uint32_t magic = meta.magic;
    uint32_t version = meta.version;
    int64_t step = meta.step;
    int64_t epoch = meta.epoch;
    float best_val_loss = meta.best_val_loss;

    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&step), sizeof(step));
    out.write(reinterpret_cast<const char*>(&epoch), sizeof(epoch));
    out.write(reinterpret_cast<const char*>(&best_val_loss), sizeof(best_val_loss));

    uint32_t desc_len = static_cast<uint32_t>(meta.description.size());
    out.write(reinterpret_cast<const char*>(&desc_len), sizeof(desc_len));
    if (desc_len > 0) {
        out.write(meta.description.data(), desc_len);
    }

    // 2. Named Parameters
    auto named_params = model.named_parameters();
    uint64_t num_tensors = named_params.size();
    out.write(reinterpret_cast<const char*>(&num_tensors), sizeof(num_tensors));

    for (const auto& [name, tensor] : named_params) {
        AIC_CHECK(tensor && tensor->defined(), InvalidArgument,
                  "Cannot checkpoint undefined parameter: ", name);

        uint32_t name_len = static_cast<uint32_t>(name.size());
        out.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        out.write(name.data(), name_len);

        int32_t rank = tensor->rank();
        out.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
        for (int d = 0; d < rank; ++d) {
            int64_t dim_val = tensor->dim(d);
            out.write(reinterpret_cast<const char*>(&dim_val), sizeof(dim_val));
        }

        int32_t dt = static_cast<int32_t>(tensor->dtype());
        out.write(reinterpret_cast<const char*>(&dt), sizeof(dt));

        // Copy tensor data to host buffer if on CUDA
        Tensor host_tensor = tensor->cpu().contiguous();
        uint64_t nbytes = host_tensor.nbytes();
        out.write(reinterpret_cast<const char*>(&nbytes), sizeof(nbytes));

        const uint8_t* byte_ptr = reinterpret_cast<const uint8_t*>(host_tensor.raw_data());
        out.write(reinterpret_cast<const char*>(byte_ptr), static_cast<std::streamsize>(nbytes));

        uint32_t checksum = compute_adler32(byte_ptr, static_cast<size_t>(nbytes));
        out.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
    }
}

CheckpointMetadata load_checkpoint(
    const std::string& filepath,
    nn::Module& model,
    optim::Optimizer* /*opt*/
) {
    std::ifstream in(filepath, std::ios::binary);
    AIC_CHECK(in.is_open(), IOError, "Failed to open checkpoint file for reading: ", filepath);

    CheckpointMetadata meta;

    // 1. Header
    in.read(reinterpret_cast<char*>(&meta.magic), sizeof(meta.magic));
    AIC_CHECK(meta.magic == AIC_CHECKPOINT_MAGIC, FormatError,
              "Invalid checkpoint magic number in: ", filepath);

    in.read(reinterpret_cast<char*>(&meta.version), sizeof(meta.version));
    AIC_CHECK(meta.version == AIC_CHECKPOINT_VERSION, FormatError,
              "Unsupported checkpoint version: ", meta.version);

    in.read(reinterpret_cast<char*>(&meta.step), sizeof(meta.step));
    in.read(reinterpret_cast<char*>(&meta.epoch), sizeof(meta.epoch));
    in.read(reinterpret_cast<char*>(&meta.best_val_loss), sizeof(meta.best_val_loss));

    uint32_t desc_len = 0;
    in.read(reinterpret_cast<char*>(&desc_len), sizeof(desc_len));
    if (desc_len > 0) {
        meta.description.resize(desc_len);
        in.read(meta.description.data(), desc_len);
    }

    // Build model parameter lookup
    std::unordered_map<std::string, std::shared_ptr<Tensor>> param_map;
    for (const auto& [name, p] : model.named_parameters()) {
        param_map[name] = p;
    }

    // 2. Read Tensors
    uint64_t num_tensors = 0;
    in.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));

    for (uint64_t t = 0; t < num_tensors; ++t) {
        uint32_t name_len = 0;
        in.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        in.read(name.data(), name_len);

        int32_t rank = 0;
        in.read(reinterpret_cast<char*>(&rank), sizeof(rank));
        std::vector<int64_t> dims(static_cast<size_t>(rank));
        for (int d = 0; d < rank; ++d) {
            in.read(reinterpret_cast<char*>(&dims[static_cast<size_t>(d)]), sizeof(int64_t));
        }

        int32_t dt = 0;
        in.read(reinterpret_cast<char*>(&dt), sizeof(dt));

        uint64_t nbytes = 0;
        in.read(reinterpret_cast<char*>(&nbytes), sizeof(nbytes));

        std::vector<uint8_t> buffer(static_cast<size_t>(nbytes));
        in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(nbytes));

        uint32_t stored_checksum = 0;
        in.read(reinterpret_cast<char*>(&stored_checksum), sizeof(stored_checksum));

        uint32_t computed_checksum = compute_adler32(buffer.data(), static_cast<size_t>(nbytes));
        AIC_CHECK(stored_checksum == computed_checksum, FormatError,
                  "Adler-32 checksum mismatch for tensor '", name, "' in checkpoint: ", filepath);

        // Load into model parameter if present
        auto it = param_map.find(name);
        if (it != param_map.end()) {
            auto& p = it->second;
            AIC_CHECK(p->shape() == Shape(dims), ShapeMismatch,
                      "Shape mismatch loading tensor '", name, "'");
            AIC_CHECK(static_cast<int32_t>(p->dtype()) == dt, DTypeMismatch,
                      "DType mismatch loading tensor '", name, "'");

            if (p->is_cpu()) {
                std::memcpy(p->raw_data(), buffer.data(), static_cast<size_t>(nbytes));
            } else {
#if AIC_WITH_CUDA
                cudaMemcpy(p->raw_data(), buffer.data(), static_cast<size_t>(nbytes), cudaMemcpyHostToDevice);
#else
                AIC_THROW(BackendUnavailable, "CUDA not enabled");
#endif
            }
        }
    }

    return meta;
}

}  // namespace aic::train
