// tests/unit/test_data.cpp
// Verification of Tokenizer, MemoryMappedFile, Dataset, and DataLoader.
#include "data/data.hpp"
#include "runtime/cuda_info.hpp"
#include "tests/test_framework.hpp"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace aic;
using namespace aic::data;

AIC_TEST(TokenizerTest, ByteLevelRoundtrip) {
    auto tok = Tokenizer::create_default();

    std::string text = "Hello, World! 🚀 Autonomous AI Core v1.0! Deep Learning in Modern C++20 with CUDA sm_120. Türkiye 🇹🇷";
    auto tokens = tok.encode(text, /*add_bos=*/true, /*add_eos=*/true);

    ASSERT_FALSE(tokens.empty());
    ASSERT_EQ(tokens.front(), tok.bos_token_id());
    ASSERT_EQ(tokens.back(), tok.eos_token_id());

    // Decode back to string
    std::string decoded = tok.decode(tokens);
    ASSERT_EQ(decoded, text);
}

AIC_TEST(TokenizerTest, BpeMergesAndCompression) {
    auto tok = Tokenizer::create_default();

    std::string text = "the and for in to of";
    auto raw_bytes = text.size();
    auto tokens = tok.encode(text);

    // Because of merges, the number of tokens must be significantly smaller than raw bytes
    ASSERT_TRUE(tokens.size() < raw_bytes);

    std::string decoded = tok.decode(tokens);
    ASSERT_EQ(decoded, text);
}

AIC_TEST(TokenizerTest, SerializationRoundtrip) {
    auto tok = Tokenizer::create_default();
    std::string temp_path = "test_tokenizer_save.bin";

    tok.save(temp_path);
    auto loaded_tok = Tokenizer::load(temp_path);
    std::filesystem::remove(temp_path);

    ASSERT_EQ(loaded_tok.vocab_size(), tok.vocab_size());
    ASSERT_EQ(loaded_tok.bos_token_id(), tok.bos_token_id());
    ASSERT_EQ(loaded_tok.eos_token_id(), tok.eos_token_id());

    std::string text = "Autonomous Reasoning Core serialization test!";
    auto orig_encoded = tok.encode(text);
    auto loaded_encoded = loaded_tok.encode(text);
    ASSERT_TRUE(orig_encoded == loaded_encoded);
    ASSERT_EQ(loaded_tok.decode(loaded_encoded), text);
}

AIC_TEST(DatasetTest, InMemorySlidingWindow) {
    // 100 consecutive tokens: 0, 1, 2, ..., 99
    std::vector<int64_t> tokens(100);
    for (int64_t i = 0; i < 100; ++i) tokens[i] = i;

    int64_t seq_len = 8;
    int64_t stride = 4;
    BinaryTokenDataset dataset(tokens, seq_len, stride);

    // Expected samples: (100 - 1 - 8) / 4 + 1 = 91 / 4 + 1 = 22 + 1 = 23
    ASSERT_EQ(dataset.size(), 23);
    ASSERT_EQ(dataset.seq_len(), seq_len);

    // Verify first sample: x = [0..7], y = [1..8]
    auto [x0, y0] = dataset.get_item(0);
    ASSERT_EQ(x0.shape(), Shape({seq_len}));
    ASSERT_EQ(y0.shape(), Shape({seq_len}));

    const int64_t* px0 = x0.data_ptr<int64_t>();
    const int64_t* py0 = y0.data_ptr<int64_t>();
    for (int64_t i = 0; i < seq_len; ++i) {
        ASSERT_EQ(px0[i], i);
        ASSERT_EQ(py0[i], i + 1); // Exact next-token alignment
    }

    // Verify second sample: start = 4, x = [4..11], y = [5..12]
    auto [x1, y1] = dataset.get_item(1);
    const int64_t* px1 = x1.data_ptr<int64_t>();
    const int64_t* py1 = y1.data_ptr<int64_t>();
    for (int64_t i = 0; i < seq_len; ++i) {
        ASSERT_EQ(px1[i], 4 + i);
        ASSERT_EQ(py1[i], 5 + i);
    }
}

AIC_TEST(DatasetTest, MemoryMappedFileIo) {
    std::string test_file = "test_dataset_tokens.bin";
    std::vector<int64_t> tokens(500);
    for (int64_t i = 0; i < 500; ++i) tokens[i] = i * 2;

    BinaryTokenDataset::write_binary_file(test_file, tokens);

    {
        BinaryTokenDataset dataset(test_file, /*seq_len=*/16, /*stride=*/8);
        ASSERT_TRUE(dataset.size() > 0);
        ASSERT_EQ(dataset.total_tokens(), 500);

        auto [x, y] = dataset.get_item(0);
        const int64_t* px = x.data_ptr<int64_t>();
        const int64_t* py = y.data_ptr<int64_t>();
        for (int64_t i = 0; i < 16; ++i) {
            ASSERT_EQ(px[i], i * 2);
            ASSERT_EQ(py[i], (i + 1) * 2);
        }
    }

    std::filesystem::remove(test_file);
}

AIC_TEST(DataLoaderTest, BatchingAndShuffling) {
    std::vector<int64_t> tokens(120);
    for (int64_t i = 0; i < 120; ++i) tokens[i] = i;

    auto ds = std::make_shared<BinaryTokenDataset>(tokens, /*seq_len=*/8, /*stride=*/4);
    int64_t total_samples = ds->size(); // (120 - 9) / 4 + 1 = 111 / 4 + 1 = 28 samples

    // DataLoader with batch_size = 8, drop_last = false -> 28 / 8 = 4 batches (8, 8, 8, 4)
    DataLoader loader(ds, /*batch_size=*/8, /*shuffle=*/true, /*drop_last=*/false);
    ASSERT_EQ(loader.num_batches(), 4);

    int64_t total_seen = 0;
    while (loader.has_next()) {
        auto [bx, by] = loader.next();
        ASSERT_EQ(bx.shape()[1], 8);
        ASSERT_EQ(by.shape()[1], 8);
        total_seen += bx.shape()[0];
    }
    ASSERT_EQ(total_seen, total_samples);

    // With drop_last = true -> 3 batches of 8 = 24 samples
    DataLoader loader_drop(ds, /*batch_size=*/8, /*shuffle=*/false, /*drop_last=*/true);
    ASSERT_EQ(loader_drop.num_batches(), 3);
}

AIC_TEST(DataLoaderTest, CudaDevicePlacement) {
    if (!is_cuda_available()) return;

    std::vector<int64_t> tokens(64);
    for (int64_t i = 0; i < 64; ++i) tokens[i] = i;

    auto ds = std::make_shared<BinaryTokenDataset>(tokens, /*seq_len=*/4, /*stride=*/4);
    DataLoader loader(ds, /*batch_size=*/4, /*shuffle=*/false, /*drop_last=*/false, Device::cuda(0));

    ASSERT_TRUE(loader.has_next());
    auto [bx, by] = loader.next();

    ASSERT_TRUE(bx.is_cuda());
    ASSERT_TRUE(by.is_cuda());
    ASSERT_EQ(bx.shape(), Shape({4, 4}));
    ASSERT_EQ(by.shape(), Shape({4, 4}));
}
