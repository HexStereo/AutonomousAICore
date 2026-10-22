// tests/integration/test_end_to_end.cpp
// Full end-to-end integration test verifying training convergence, autonomous telemetry,
// and KV-cached autoregressive generation on CPU and CUDA Blackwell GPU.
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

#include "autonomous/controller.hpp"
#include "data/dataloader.hpp"
#include "data/dataset.hpp"
#include "data/tokenizer.hpp"
#include "models/transformer_lm.hpp"
#include "optim/adamw.hpp"
#include "runtime/cuda_info.hpp"
#include "tests/test_framework.hpp"
#include "train/checkpoint.hpp"
#include "train/trainer.hpp"

using namespace aic;
using namespace aic::models;
using namespace aic::autonomous;

AIC_TEST(EndToEndIntegration, FullTrainingConvergenceAndGeneration) {
    auto run_e2e_test = [](Device dev) {
        std::string test_dir = "test_e2e_ckpts_" + dev.str();
        for (char& c : test_dir) if (c == ':') c = '_';
        std::filesystem::remove_all(test_dir);

        // 1. Tokenizer
        data::Tokenizer tokenizer = data::Tokenizer::create_default();
        std::vector<std::string> corpus = {
            "Autonomous deep learning core on CUDA Blackwell architecture.",
            "High performance computing with Tensor Cores and AVX2 SIMD.",
            "Closed loop self healing telemetry ensures training stability."
        };

        std::vector<int64_t> all_tokens;
        for (const auto& line : corpus) {
            auto tok = tokenizer.encode(line);
            all_tokens.insert(all_tokens.end(), tok.begin(), tok.end());
        }

        // 2. Dataset & DataLoader
        int64_t seq_len = 8;
        int64_t batch_size = 2;
        auto dataset = std::make_shared<data::BinaryTokenDataset>(all_tokens, seq_len, /*stride=*/4);
        auto loader = std::make_shared<data::DataLoader>(dataset, batch_size, /*shuffle=*/false, false, dev);

        // 3. Model
        TransformerLMConfig cfg;
        cfg.vocab_size = static_cast<int64_t>(tokenizer.vocab_size());
        cfg.hidden_dim = 32;
        cfg.num_layers = 2;
        cfg.num_heads = 2;
        cfg.intermediate_dim = 64;
        cfg.max_seq_len = 32;

        auto model = std::make_shared<TransformerLM>(cfg, dev);

        // 4. Optimizer & Scheduler
        optim::AdamWOptions opt_opts;
        opt_opts.lr = 0.005f;
        auto optimizer = std::make_shared<optim::AdamW>(model->parameters(), opt_opts);

        train::CosineWarmupSchedulerOptions sched_opts;
        sched_opts.max_lr = 0.005f;
        sched_opts.min_lr = 1e-4f;
        sched_opts.warmup_steps = 2;
        sched_opts.total_steps = 15;
        train::CosineWarmupScheduler scheduler(sched_opts);

        // 5. Trainer & Autonomous Controller
        train::TrainingConfig train_cfg;
        train_cfg.max_steps = 15;
        train_cfg.grad_accum_steps = 1;
        train_cfg.checkpoint_dir = test_dir;

        train::Trainer trainer(model, optimizer, scheduler, loader, nullptr, train_cfg);

        AutonomousConfig auto_cfg;
        auto_cfg.checkpoint_dir = test_dir;
        auto_cfg.checkpoint_interval = 5;
        AutonomousController controller(auto_cfg);

        float initial_loss = 0.0f;
        float final_loss = 0.0f;

        trainer.on_step_end = [&](const train::StepMetrics& m) {
            if (m.step == 1) initial_loss = m.loss;
            final_loss = m.loss;
            controller.step(trainer, m, 50000, dev.is_cuda() ? dev.index : 0);
        };

        trainer.train();

        // Verifications
        ASSERT_TRUE(initial_loss > 0.0f);
        ASSERT_TRUE(final_loss > 0.0f);
        ASSERT_TRUE(final_loss < initial_loss); // Verified loss descent!
        ASSERT_TRUE(controller.rollback_count() == 0); // No anomalies during normal training
        ASSERT_EQ(controller.health_state(), HealthState::Optimal);

        // Verify checkpoint saving
        ASSERT_FALSE(controller.last_stable_checkpoint().empty());
        ASSERT_TRUE(std::filesystem::exists(controller.last_stable_checkpoint()));

        // 6. Test Autoregressive Generation
        std::vector<int64_t> prompt = {65, 117, 116, 111}; // "Auto"
        auto gen = model->generate(prompt, /*max_new_tokens=*/8, /*temperature=*/0.0f); // greedy
        ASSERT_EQ(gen.size(), prompt.size() + 8);
        for (size_t i = 0; i < prompt.size(); ++i) {
            ASSERT_EQ(gen[i], prompt[i]);
        }

        std::filesystem::remove_all(test_dir);
    };

    run_e2e_test(Device::cpu());
    if (is_cuda_available()) {
        run_e2e_test(Device::cuda(0));
    }
}
