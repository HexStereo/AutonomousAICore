// tests/unit/test_train.cpp
// Verification of Learning rate scheduler, Binary checkpointing, and Training Engine.
#include "train/train.hpp"
#include "models/models.hpp"
#include "runtime/cuda_info.hpp"
#include "tests/test_framework.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace aic;
using namespace aic::models;
using namespace aic::train;

AIC_TEST(SchedulerTest, CosineWarmupSchedule) {
    CosineWarmupSchedulerOptions opts;
    opts.initial_lr = 0.0f;
    opts.max_lr = 1e-3f;
    opts.min_lr = 1e-5f;
    opts.warmup_steps = 100;
    opts.total_steps = 1000;

    CosineWarmupScheduler sched(opts);

    // Step 0: initial_lr
    ASSERT_NEAR(sched.get_lr(0), 0.0f, 1e-6f);

    // Step 50: half warmup
    ASSERT_NEAR(sched.get_lr(50), 5e-4f, 1e-6f);

    // Step 100: peak max_lr
    ASSERT_NEAR(sched.get_lr(100), 1e-3f, 1e-6f);

    // Step 550: halfway through cosine decay (100 + (1000 - 100) / 2 = 550)
    // cos(pi / 2) = 0 -> coeff = 0.5 -> min_lr + 0.5 * (max_lr - min_lr) ~ 5.05e-4
    float expected_mid = 1e-5f + 0.5f * (1e-3f - 1e-5f);
    ASSERT_NEAR(sched.get_lr(550), expected_mid, 1e-5f);

    // Step 1000: min_lr
    ASSERT_NEAR(sched.get_lr(1000), 1e-5f, 1e-6f);

    // Step 1500 (beyond total): stays at min_lr
    ASSERT_NEAR(sched.get_lr(1500), 1e-5f, 1e-6f);
}

AIC_TEST(CheckpointTest, SaveLoadVerification) {
    TransformerLMConfig cfg;
    cfg.vocab_size = 32;
    cfg.hidden_dim = 16;
    cfg.num_layers = 2;
    cfg.num_heads = 2;
    cfg.intermediate_dim = 32;

    TransformerLM model_orig(cfg, Device::cpu());

    std::string ckpt_path = "test_model_save.aic";
    CheckpointMetadata meta_orig;
    meta_orig.step = 42;
    meta_orig.epoch = 2;
    meta_orig.best_val_loss = 1.234f;
    meta_orig.description = "unit_test_checkpoint";

    save_checkpoint(ckpt_path, model_orig, nullptr, meta_orig);

    // Create a new model with different random weights
    TransformerLM model_loaded(cfg, Device::cpu());

    auto meta_loaded = load_checkpoint(ckpt_path, model_loaded, nullptr);
    std::filesystem::remove(ckpt_path);

    ASSERT_EQ(meta_loaded.step, 42);
    ASSERT_EQ(meta_loaded.epoch, 2);
    ASSERT_NEAR(meta_loaded.best_val_loss, 1.234f, 1e-5f);
    ASSERT_EQ(meta_loaded.description, "unit_test_checkpoint");

    // Verify all weights are bitwise identical
    auto orig_params = model_orig.named_parameters();
    auto loaded_params = model_loaded.named_parameters();
    ASSERT_EQ(orig_params.size(), loaded_params.size());

    for (size_t i = 0; i < orig_params.size(); ++i) {
        ASSERT_EQ(orig_params[i].first, loaded_params[i].first);
        const float* p_orig = orig_params[i].second->data_ptr<float>();
        const float* p_load = loaded_params[i].second->data_ptr<float>();
        int64_t n = orig_params[i].second->numel();
        for (int64_t j = 0; j < n; ++j) {
            ASSERT_EQ(p_orig[j], p_load[j]);
        }
    }
}

AIC_TEST(CheckpointTest, Adler32CorruptionDetection) {
    TransformerLMConfig cfg;
    cfg.vocab_size = 16;
    cfg.hidden_dim = 8;
    cfg.num_layers = 1;
    cfg.num_heads = 1;
    cfg.intermediate_dim = 16;

    TransformerLM model(cfg, Device::cpu());
    std::string ckpt_path = "test_corrupt.aic";
    save_checkpoint(ckpt_path, model);

    // Intentionally corrupt bytes in the tensor payload near the end of file
    {
        std::fstream f(ckpt_path, std::ios::in | std::ios::out | std::ios::binary);
        ASSERT_TRUE(f.is_open());
        f.seekg(-20, std::ios::end);
        char byte = 0;
        f.read(&byte, 1);
        byte ^= 0xFF;
        f.seekp(-20, std::ios::end);
        f.write(&byte, 1);
        f.flush();
    }

    TransformerLM model_target(cfg, Device::cpu());
    ASSERT_THROW(load_checkpoint(ckpt_path, model_target), Error);

    std::filesystem::remove(ckpt_path);
}

AIC_TEST(TrainerTest, GradientAccumulationAndTraining) {
    auto run_trainer_test = [](Device dev) {
        TransformerLMConfig cfg;
        cfg.vocab_size = 32;
        cfg.hidden_dim = 16;
        cfg.num_layers = 2;
        cfg.num_heads = 2;
        cfg.intermediate_dim = 32;

        auto model = std::make_shared<TransformerLM>(cfg, dev);

        // Synthetic tokens: repeated sequence
        std::vector<int64_t> tokens(256);
        for (int64_t i = 0; i < 256; ++i) tokens[i] = i % cfg.vocab_size;

        auto dataset = std::make_shared<data::BinaryTokenDataset>(tokens, /*seq_len=*/8, /*stride=*/4);
        auto dataloader = std::make_shared<data::DataLoader>(dataset, /*batch_size=*/4, /*shuffle=*/false, false, dev);

        optim::AdamWOptions opt_opts;
        opt_opts.lr = 0.01f;
        auto optimizer = std::make_shared<optim::AdamW>(model->parameters(), opt_opts);

        CosineWarmupSchedulerOptions sched_opts;
        sched_opts.max_lr = 0.01f;
        sched_opts.min_lr = 1e-4f;
        sched_opts.warmup_steps = 2;
        sched_opts.total_steps = 15;
        CosineWarmupScheduler scheduler(sched_opts);

        TrainingConfig train_cfg;
        train_cfg.max_steps = 10;
        train_cfg.grad_accum_steps = 2;
        train_cfg.checkpoint_dir = "test_trainer_ckpts";

        Trainer trainer(model, optimizer, scheduler, dataloader, nullptr, train_cfg);

        float initial_loss = 0.0f;
        float final_loss = 0.0f;

        trainer.on_step_end = [&](const StepMetrics& m) {
            if (m.step == 1) initial_loss = m.loss;
            final_loss = m.loss;
        };

        trainer.train();

        ASSERT_EQ(trainer.current_step(), 10);
        ASSERT_TRUE(initial_loss > 0.0f);
        ASSERT_TRUE(final_loss > 0.0f);

        std::filesystem::remove_all(train_cfg.checkpoint_dir);
    };

    run_trainer_test(Device::cpu());
    if (is_cuda_available()) {
        run_trainer_test(Device::cuda(0));
    }
}
