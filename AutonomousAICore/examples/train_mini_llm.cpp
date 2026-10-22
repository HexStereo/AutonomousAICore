// examples/train_mini_llm.cpp
// Standalone end-to-end example: Trains a mini-LLM with real-time autonomous telemetry
// on NVIDIA GeForce RTX 5090 Blackwell GPU (or CPU) and generates text.
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "autonomous/controller.hpp"
#include "data/dataloader.hpp"
#include "data/dataset.hpp"
#include "data/tokenizer.hpp"
#include "models/transformer_lm.hpp"
#include "optim/adamw.hpp"
#include "runtime/cuda_info.hpp"
#include "train/checkpoint.hpp"
#include "train/trainer.hpp"

using namespace aic;
using namespace aic::models;
using namespace aic::autonomous;

int main() {
    std::cout << "=======================================================\n"
              << "   AutonomousAICore: End-to-End Mini-LLM Training Demo\n"
              << "=======================================================\n\n";

    Device device = is_cuda_available() ? Device::cuda(0) : Device::cpu();
    std::cout << "Using device: " << device.str() << "\n";
    if (device.is_cuda()) {
        auto prop = get_gpu_properties(device.index);
        std::cout << "  Device: " << prop.name
                  << " (Blackwell sm_" << prop.compute_major << prop.compute_minor << ")\n"
                  << "  VRAM:   " << (prop.total_vram_bytes / (1024 * 1024 * 1024)) << " GB\n\n";
    }

    // 1. Prepare Training Corpus & Initialize Byte-Level BPE Tokenizer
    std::cout << "1. Initializing Byte-Level BPE Tokenizer...\n";
    std::vector<std::string> corpus = {
        "Deep learning systems require robust tensor abstractions and automatic differentiation engines.",
        "High performance computing relies on hardware-accelerated matrix multiplication on Tensor Cores.",
        "Autonomous training controllers dynamically adjust learning rates and mitigate divergence via rollbacks.",
        "Modern C++20 and CUDA C++ sm_120 enable zero-overhead systems programming for artificial intelligence.",
        "The NVIDIA Blackwell architecture features fifth-generation Tensor Cores for deep learning workloads."
    };

    data::Tokenizer tokenizer = data::Tokenizer::create_default();
    std::cout << "  Vocabulary size: " << tokenizer.vocab_size() << " tokens\n";

    // Flatten corpus into a continuous token stream
    std::vector<int64_t> all_tokens;
    for (const auto& line : corpus) {
        auto tok = tokenizer.encode(line);
        all_tokens.insert(all_tokens.end(), tok.begin(), tok.end());
    }
    std::cout << "  Corpus token count: " << all_tokens.size() << " tokens\n\n";

    // 2. Setup Memory-Mapped Sliding-Window Dataset & DataLoader
    std::cout << "2. Creating Dataset & DataLoader...\n";
    int64_t seq_len = 16;
    int64_t batch_size = 4;
    auto dataset = std::make_shared<data::BinaryTokenDataset>(all_tokens, seq_len, /*stride=*/4);
    auto loader = std::make_shared<data::DataLoader>(dataset, batch_size, /*shuffle=*/true, false, device);
    std::cout << "  Dataset sequences: " << dataset->size() << " (batch size " << batch_size << ")\n\n";

    // 3. Initialize TransformerLM Model Architecture
    std::cout << "3. Initializing TransformerLM Model Architecture...\n";
    TransformerLMConfig cfg;
    cfg.vocab_size = static_cast<int64_t>(tokenizer.vocab_size());
    cfg.hidden_dim = 64;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.intermediate_dim = 128;
    cfg.max_seq_len = seq_len * 2;

    auto model = std::make_shared<TransformerLM>(cfg, device);
    int64_t param_count = 0;
    for (const auto& p : model->parameters()) {
        if (p && p->defined()) param_count += p->numel();
    }
    std::cout << "  Model Parameters: " << param_count << " ("
              << (param_count * 4 / 1024) << " KB FP32 weights)\n\n";

    // 4. Configure AdamW Optimizer & Cosine Warmup Scheduler
    std::cout << "4. Configuring AdamW Optimizer & Warmup Scheduler...\n";
    optim::AdamWOptions opt_opts;
    opt_opts.lr = 0.003f;
    opt_opts.weight_decay = 0.01f;
    auto optimizer = std::make_shared<optim::AdamW>(model->parameters(), opt_opts);

    int64_t total_steps = 30;
    train::CosineWarmupSchedulerOptions sched_opts;
    sched_opts.max_lr = 0.003f;
    sched_opts.min_lr = 1e-4f;
    sched_opts.warmup_steps = 5;
    sched_opts.total_steps = total_steps;
    train::CosineWarmupScheduler scheduler(sched_opts);

    // 5. Initialize Trainer & Autonomous Controller
    std::cout << "5. Initializing Trainer & Autonomous Controller...\n";
    train::TrainingConfig train_cfg;
    train_cfg.max_steps = total_steps;
    train_cfg.grad_accum_steps = 2;
    train_cfg.max_grad_norm = 1.0f;
    train_cfg.checkpoint_dir = "example_ckpts";

    train::Trainer trainer(model, optimizer, scheduler, loader, nullptr, train_cfg);

    AutonomousConfig auto_cfg;
    auto_cfg.checkpoint_dir = "example_ckpts";
    auto_cfg.checkpoint_interval = 10;
    AutonomousController controller(auto_cfg);

    trainer.on_step_end = [&](const train::StepMetrics& m) {
        auto action = controller.step(trainer, m, param_count, device.is_cuda() ? device.index : 0);
        std::cout << "  " << controller.telemetry().latest().summary();
        if (action != ControllerAction::None) {
            std::cout << " -> [AUTONOMOUS ACTION: " << action_name(action) << "]";
        }
        std::cout << "\n";
    };

    // 6. Execute Autonomous Training Run
    std::cout << "\n6. Executing Training Run with Closed-Loop Autonomous Control:\n";
    trainer.train();

    // 7. Autoregressive Generation Demonstration
    std::cout << "\n7. Testing Autoregressive Generation with KV Cache:\n";
    std::string prompt_text = "Autonomous";
    auto prompt_tokens = tokenizer.encode(prompt_text);
    std::cout << "  Prompt: \"" << prompt_text << "\" (tokens: ";
    for (auto id : prompt_tokens) std::cout << id << " ";
    std::cout << ")\n";

    auto gen_tokens = model->generate(prompt_tokens, /*max_new_tokens=*/16, /*temperature=*/0.8f, /*top_k=*/10, /*top_p=*/0.9f);
    std::string generated_text = tokenizer.decode(gen_tokens);

    std::cout << "  Output: \"" << generated_text << "\"\n\n";
    std::cout << "Autonomous AICore End-to-End Demo Finished Successfully!\n";
    return 0;
}
