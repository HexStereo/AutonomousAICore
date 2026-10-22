// tests/test_framework.hpp
// Lightweight, zero-dependency C++20 test framework with test suites, color reporting, and timing.
#pragma once

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace aic::test {

struct TestCase {
    std::string suite;
    std::string name;
    std::function<void()> fn;
};

class TestRegistry {
public:
    static TestRegistry& instance() {
        static TestRegistry reg;
        return reg;
    }

    void add(std::string suite, std::string name, std::function<void()> fn) {
        tests_.push_back({std::move(suite), std::move(name), std::move(fn)});
    }

    int run_all(const std::string& filter = "") {
        int passed = 0;
        int failed = 0;
        auto start_all = std::chrono::high_resolution_clock::now();

        std::cout << "\n=======================================================\n";
        std::cout << "  Autonomous AI Core - Test Suite Runner\n";
        std::cout << "=======================================================\n" << std::flush;

        for (const auto& t : tests_) {
            std::string full_name = t.suite + "." + t.name;
            if (!filter.empty() && full_name.find(filter) == std::string::npos) {
                continue;
            }
            std::cout << "[ RUN      ] " << full_name << "\n" << std::flush;
            auto start = std::chrono::high_resolution_clock::now();
            try {
                t.fn();
                auto end = std::chrono::high_resolution_clock::now();
                auto ms = std::chrono::duration<double, std::milli>(end - start).count();
                std::cout << "[       OK ] " << full_name << " (" << ms << " ms)\n" << std::flush;
                passed++;
            } catch (const std::exception& ex) {
                auto end = std::chrono::high_resolution_clock::now();
                auto ms = std::chrono::duration<double, std::milli>(end - start).count();
                std::cout << "[  FAILED  ] " << full_name << " (" << ms << " ms)\n" << std::flush;
                std::cout << "             Error: " << ex.what() << "\n" << std::flush;
                failed++;
            } catch (...) {
                std::cout << "[  FAILED  ] " << full_name << " (unknown exception)\n" << std::flush;
                failed++;
            }
        }

        auto end_all = std::chrono::high_resolution_clock::now();
        auto total_ms = std::chrono::duration<double, std::milli>(end_all - start_all).count();

        std::cout << "=======================================================\n";
        std::cout << "  Passed: " << passed << " | Failed: " << failed
                  << " | Total: " << (passed + failed)
                  << " (" << total_ms << " ms)\n";
        std::cout << "=======================================================\n\n";

        return failed == 0 ? 0 : 1;
    }

private:
    std::vector<TestCase> tests_;
};

struct TestRegistrar {
    TestRegistrar(std::string suite, std::string name, std::function<void()> fn) {
        TestRegistry::instance().add(std::move(suite), std::move(name), std::move(fn));
    }
};

}  // namespace aic::test

#define AIC_TEST(suite, name)                                                         \
    void aic_test_##suite##_##name();                                                \
    static ::aic::test::TestRegistrar aic_reg_##suite##_##name(                       \
        #suite, #name, aic_test_##suite##_##name);                                    \
    void aic_test_##suite##_##name()

#define ASSERT_TRUE(cond)                                                             \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::ostringstream os__;                                                  \
            os__ << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(os__.str());                                     \
        }                                                                             \
    } while (false)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))

#define ASSERT_EQ(a, b)                                                               \
    do {                                                                              \
        auto val_a__ = (a);                                                           \
        auto val_b__ = (b);                                                           \
        if (!(val_a__ == val_b__)) {                                                  \
            std::ostringstream os__;                                                  \
            os__ << "Assertion failed: (" #a " == " #b ") [" << val_a__ << " != "    \
                 << val_b__ << "] at " << __FILE__ << ":" << __LINE__;               \
            throw std::runtime_error(os__.str());                                     \
        }                                                                             \
    } while (false)

#define ASSERT_NEAR(a, b, tol)                                                        \
    do {                                                                              \
        double diff__ = std::abs(static_cast<double>(a) - static_cast<double>(b));    \
        if (diff__ > static_cast<double>(tol)) {                                      \
            std::ostringstream os__;                                                  \
            os__ << "Assertion failed: |" #a " - " #b "| <= " #tol " [diff="          \
                 << diff__ << "] at " << __FILE__ << ":" << __LINE__;                \
            throw std::runtime_error(os__.str());                                     \
        }                                                                             \
    } while (false)

#define ASSERT_THROW(expr, ExceptionType)                                             \
    do {                                                                              \
        bool threw__ = false;                                                         \
        try {                                                                         \
            expr;                                                                     \
        } catch (const ExceptionType&) {                                              \
            threw__ = true;                                                           \
        } catch (...) {                                                               \
            throw std::runtime_error("Expected " #ExceptionType " but caught different exception at " \
                                     __FILE__ ":" + std::to_string(__LINE__));        \
        }                                                                             \
        if (!threw__) {                                                               \
            throw std::runtime_error("Expected " #ExceptionType " was not thrown at " \
                                     __FILE__ ":" + std::to_string(__LINE__));        \
        }                                                                             \
    } while (false)
