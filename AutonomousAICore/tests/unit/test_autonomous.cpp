// tests/unit/test_autonomous.cpp
// Unit tests for TelemetryTracker, Anomaly Detection, and AutonomousController.
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include "autonomous/controller.hpp"
#include "autonomous/telemetry.hpp"
#include "models/transformer_lm.hpp"
#include "optim/adamw.hpp"
#include "runtime/cuda_info.hpp"
#include "tests/test_framework.hpp"
#include "train/trainer.hpp"

using namespace aic;
using namespace aic::autonomous;
using namespace aic::models;

AIC_TEST(TelemetryTest, TrackingAndEMACalculation) {
    TelemetryTracker tracker(/*window_size=*/10, /*alpha_short=*/0.8f, /*alpha_long=*/0.95f);

    // Feed normal training steps
    for (int64_t step = 1; step <= 5; ++step) {
        float loss = 2.0f - static_cast<float>(step) * 0.1f;
        float grad_norm = 1.0f;
        float lr = 1e-3f;
        float tok_per_sec = 10000.0f;

        auto snap = tracker.record_step(
            step, loss, grad_norm, lr, tok_per_sec,
            /*step_duration_sec=*/0.01, /*total_tokens=*/100, /*model_params=*/1000000
        );

        ASSERT_EQ(snap.step, step);
        ASSERT_TRUE(snap.loss > 0.0f);
        ASSERT_TRUE(snap.loss_ema_short > 0.0f);
        ASSERT_TRUE(snap.loss_ema_long > 0.0f);
        ASSERT_TRUE(snap.estimated_tflops > 0.0f);
    }

    ASSERT_EQ(tracker.count(), 5);
    ASSERT_FALSE(tracker.is_nan_or_inf());
    ASSERT_FALSE(tracker.is_diverging());
    ASSERT_FALSE(tracker.is_gradient_exploding());

    auto snap = tracker.latest();
    std::string summary = snap.summary();
    ASSERT_FALSE(summary.empty());
    std::string json = snap.to_json();
    ASSERT_TRUE(json.find("\"step\":5") != std::string::npos);
}

AIC_TEST(TelemetryTest, HardwareCounters) {
    auto stats = TelemetryTracker::query_hardware_memory(0);
    ASSERT_TRUE(stats.host_total_bytes > 0);
    ASSERT_TRUE(stats.host_free_bytes > 0);
    ASSERT_TRUE(stats.host_free_bytes <= stats.host_total_bytes);
    ASSERT_TRUE(stats.host_used_ratio() >= 0.0f && stats.host_used_ratio() <= 1.0f);

    if (is_cuda_available()) {
        ASSERT_TRUE(stats.gpu_total_bytes > 0);
        ASSERT_TRUE(stats.gpu_free_bytes > 0);
        ASSERT_TRUE(stats.gpu_used_ratio() >= 0.0f && stats.gpu_used_ratio() <= 1.0f);
    }
}

AIC_TEST(AnomalyTest, NanAndInfDetection) {
    TelemetryTracker tracker;
    tracker.record_step(1, 1.5f, 0.5f, 1e-3f, 5000.0f, 0.01, 50);
    ASSERT_FALSE(tracker.is_nan_or_inf());

    // Inject NaN loss
    float nan_val = std::numeric_limits<float>::quiet_NaN();
    auto snap = tracker.record_step(2, nan_val, 0.5f, 1e-3f, 5000.0f, 0.01, 50);

    ASSERT_TRUE(tracker.is_nan_or_inf());
    ASSERT_EQ(snap.detected_anomaly, AnomalyType::NanOrInfLoss);

    // Inject Inf gradient
    float inf_val = std::numeric_limits<float>::infinity();
    auto snap_inf = tracker.record_step(3, 1.2f, inf_val, 1e-3f, 5000.0f, 0.01, 50);
    ASSERT_TRUE(tracker.is_nan_or_inf());
    ASSERT_EQ(snap_inf.detected_anomaly, AnomalyType::NanOrInfGradient);
}

AIC_TEST(AnomalyTest, DivergenceAndSpikeDetection) {
    TelemetryTracker tracker(/*window_size=*/20, /*alpha_short=*/0.7f, /*alpha_long=*/0.99f);

    // Prime with stable loss around 1.0
    for (int i = 1; i <= 10; ++i) {
        tracker.record_step(i, 1.0f, 0.5f, 1e-3f, 5000.0f, 0.01, 50);
    }
    ASSERT_FALSE(tracker.is_diverging());

    // Inject sudden 5x loss divergence for 3 steps
    tracker.record_step(11, 5.0f, 0.5f, 1e-3f, 5000.0f, 0.01, 50);
    tracker.record_step(12, 6.0f, 0.5f, 1e-3f, 5000.0f, 0.01, 50);
    auto snap = tracker.record_step(13, 7.0f, 0.5f, 1e-3f, 5000.0f, 0.01, 50);

    ASSERT_TRUE(tracker.is_diverging());
    ASSERT_EQ(snap.detected_anomaly, AnomalyType::LossDivergence);

    // Test gradient norm explosion
    TelemetryTracker tracker_grad(/*window_size=*/20);
    for (int i = 1; i <= 6; ++i) {
        tracker_grad.record_step(i, 1.0f, 0.2f, 1e-3f, 5000.0f, 0.01, 50);
    }
    auto grad_snap = tracker_grad.record_step(7, 1.0f, 5.0f, 1e-3f, 5000.0f, 0.01, 50); // 25x baseline
    ASSERT_TRUE(tracker_grad.is_gradient_exploding());
    ASSERT_EQ(grad_snap.detected_anomaly, AnomalyType::GradientExplosion);
}

AIC_TEST(ControllerTest, AutonomousRollbackAndSelfHealing) {
    std::string test_dir = "test_auto_ckpts";
    std::filesystem::remove_all(test_dir);

    AutonomousConfig auto_cfg;
    auto_cfg.checkpoint_interval = 2;
    auto_cfg.checkpoint_dir = test_dir;
    auto_cfg.lr_decay_on_rollback = 0.5f;

    AutonomousController controller(auto_cfg);

    // Setup miniature model and trainer
    TransformerLMConfig model_cfg;
    model_cfg.vocab_size = 16;
    model_cfg.hidden_dim = 8;
    model_cfg.num_layers = 1;
    model_cfg.num_heads = 1;
    model_cfg.intermediate_dim = 16;

    auto model = std::make_shared<TransformerLM>(model_cfg, Device::cpu());

    optim::AdamWOptions opt_opts;
    opt_opts.lr = 0.01f;
    auto optimizer = std::make_shared<optim::AdamW>(model->parameters(), opt_opts);

    train::CosineWarmupSchedulerOptions sched_opts;
    sched_opts.max_lr = 0.01f;
    sched_opts.min_lr = 1e-4f;
    sched_opts.warmup_steps = 1;
    sched_opts.total_steps = 10;
    train::CosineWarmupScheduler scheduler(sched_opts);

    std::vector<int64_t> tokens(64, 1);
    auto dataset = std::make_shared<data::BinaryTokenDataset>(tokens, 4, 2);
    auto loader = std::make_shared<data::DataLoader>(dataset, 2, false, false, Device::cpu());

    train::TrainingConfig train_cfg;
    train_cfg.max_steps = 10;
    train_cfg.checkpoint_dir = test_dir;
    train::Trainer trainer(model, optimizer, scheduler, loader, nullptr, train_cfg);

    // Step 1: Normal step
    train::StepMetrics m1{1, 2.5f, 0.8f, 0.01f, 1000.0f};
    auto act1 = controller.step(trainer, m1, 10000);
    ASSERT_EQ(act1, ControllerAction::None);
    ASSERT_EQ(controller.health_state(), HealthState::Optimal);

    // Step 2: Normal step -> triggers auto-checkpoint save
    train::StepMetrics m2{2, 2.3f, 0.7f, 0.01f, 1000.0f};
    auto act2 = controller.step(trainer, m2, 10000);
    ASSERT_EQ(act2, ControllerAction::None);
    ASSERT_FALSE(controller.last_stable_checkpoint().empty());
    ASSERT_TRUE(std::filesystem::exists(controller.last_stable_checkpoint()));

    // Step 3: Inject NaN loss anomaly!
    train::StepMetrics m3{3, std::numeric_limits<float>::quiet_NaN(), 0.7f, 0.01f, 1000.0f};
    auto act3 = controller.step(trainer, m3, 10000);

    // Controller should automatically trigger and execute RollbackCheckpoint
    ASSERT_EQ(act3, ControllerAction::RollbackCheckpoint);
    ASSERT_EQ(controller.health_state(), HealthState::Recovering);
    ASSERT_EQ(controller.rollback_count(), 1);

    // Learning rate should have decayed by 50%
    ASSERT_NEAR(optimizer->options().lr, 0.005f, 1e-6f);

    std::filesystem::remove_all(test_dir);
}

AIC_TEST(ControllerTest, DynamicGradientClippingAndGradAccum) {
    AutonomousConfig auto_cfg;
    auto_cfg.enable_grad_clip_tuning = true;
    auto_cfg.tighten_clip_factor = 0.5f;

    AutonomousController controller(auto_cfg);

    TransformerLMConfig model_cfg;
    model_cfg.vocab_size = 16;
    model_cfg.hidden_dim = 8;
    model_cfg.num_layers = 1;
    model_cfg.num_heads = 1;
    model_cfg.intermediate_dim = 16;

    auto model = std::make_shared<TransformerLM>(model_cfg, Device::cpu());
    auto optimizer = std::make_shared<optim::AdamW>(model->parameters(), optim::AdamWOptions{});
    train::CosineWarmupScheduler scheduler(train::CosineWarmupSchedulerOptions{});
    std::vector<int64_t> tokens(32, 1);
    auto dataset = std::make_shared<data::BinaryTokenDataset>(tokens, 4, 2);
    auto loader = std::make_shared<data::DataLoader>(dataset, 2, false, false, Device::cpu());

    train::TrainingConfig train_cfg;
    train_cfg.max_grad_norm = 2.0f;
    train::Trainer trainer(model, optimizer, scheduler, loader, nullptr, train_cfg);

    // Feed baseline steps
    for (int64_t i = 1; i <= 6; ++i) {
        train::StepMetrics m{i, 1.0f, 0.3f, 0.001f, 1000.0f};
        controller.step(trainer, m);
    }
    ASSERT_NEAR(trainer.config().max_grad_norm, 2.0f, 1e-5f);

    // Inject gradient norm explosion
    train::StepMetrics m_spike{7, 1.0f, 8.0f, 0.001f, 1000.0f};
    auto act = controller.step(trainer, m_spike);

    ASSERT_EQ(act, ControllerAction::TightenGradientClipping);
    // Max grad norm should be tightened by factor 0.5 -> 1.0
    ASSERT_NEAR(trainer.config().max_grad_norm, 1.0f, 1e-5f);
}

AIC_TEST(ControllerTest, MaxRollbackExceededHalt) {
    AutonomousConfig auto_cfg;
    auto_cfg.max_rollback_retries = 2;
    auto_cfg.checkpoint_interval = 1;
    auto_cfg.checkpoint_dir = "test_halt_ckpts";
    std::filesystem::remove_all(auto_cfg.checkpoint_dir);

    AutonomousController controller(auto_cfg);

    TransformerLMConfig model_cfg;
    model_cfg.vocab_size = 16;
    model_cfg.hidden_dim = 8;
    model_cfg.num_layers = 1;
    model_cfg.num_heads = 1;
    model_cfg.intermediate_dim = 16;

    auto model = std::make_shared<TransformerLM>(model_cfg, Device::cpu());
    auto optimizer = std::make_shared<optim::AdamW>(model->parameters(), optim::AdamWOptions{});
    train::CosineWarmupScheduler scheduler(train::CosineWarmupSchedulerOptions{});
    std::vector<int64_t> tokens(32, 1);
    auto dataset = std::make_shared<data::BinaryTokenDataset>(tokens, 4, 2);
    auto loader = std::make_shared<data::DataLoader>(dataset, 2, false, false, Device::cpu());

    train::TrainingConfig train_cfg;
    train::Trainer trainer(model, optimizer, scheduler, loader, nullptr, train_cfg);

    // Save stable checkpoint at step 1
    train::StepMetrics m1{1, 1.5f, 0.4f, 0.001f, 1000.0f};
    controller.step(trainer, m1);

    // Rollback 1
    train::StepMetrics m_nan1{2, std::numeric_limits<float>::quiet_NaN(), 0.4f, 0.001f, 1000.0f};
    auto act1 = controller.step(trainer, m_nan1);
    ASSERT_EQ(act1, ControllerAction::RollbackCheckpoint);

    // Rollback 2
    train::StepMetrics m_nan2{2, std::numeric_limits<float>::quiet_NaN(), 0.4f, 0.001f, 1000.0f};
    auto act2 = controller.step(trainer, m_nan2);
    ASSERT_EQ(act2, ControllerAction::RollbackCheckpoint);

    // Rollback 3 -> Exceeds max_rollback_retries (2) -> Halt!
    train::StepMetrics m_nan3{2, std::numeric_limits<float>::quiet_NaN(), 0.4f, 0.001f, 1000.0f};
    auto act3 = controller.step(trainer, m_nan3);
    ASSERT_EQ(act3, ControllerAction::Halt);
    ASSERT_EQ(controller.health_state(), HealthState::Halted);

    std::filesystem::remove_all(auto_cfg.checkpoint_dir);
}
