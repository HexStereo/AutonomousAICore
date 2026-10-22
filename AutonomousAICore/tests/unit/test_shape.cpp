// tests/unit/test_shape.cpp
#include "core/shape.hpp"
#include "tests/test_framework.hpp"

using namespace aic;

AIC_TEST(ShapeTest, BasicProperties) {
    Shape s({2, 3, 4});
    ASSERT_EQ(s.rank(), 3);
    ASSERT_EQ(s.numel(), 24);
    ASSERT_EQ(s[0], 2);
    ASSERT_EQ(s[1], 3);
    ASSERT_EQ(s[2], 4);
    ASSERT_EQ(s[-1], 4);
    ASSERT_EQ(s[-2], 3);
    ASSERT_EQ(s[-3], 2);

    auto strides = s.default_strides();
    ASSERT_EQ(strides.size(), 3);
    ASSERT_EQ(strides[0], 12);
    ASSERT_EQ(strides[1], 4);
    ASSERT_EQ(strides[2], 1);
}

AIC_TEST(ShapeTest, ScalarShape) {
    Shape s({});
    ASSERT_EQ(s.rank(), 0);
    ASSERT_EQ(s.numel(), 1);
    ASSERT_TRUE(s.default_strides().empty());
}

AIC_TEST(ShapeTest, OverflowProtection) {
    ASSERT_THROW(Shape({1000000000LL, 1000000000LL, 1000000000LL}), Error);
    ASSERT_THROW(Shape({-5, 10}), Error);
}

AIC_TEST(ShapeTest, SqueezeAndUnsqueeze) {
    Shape s({1, 3, 1, 4});
    ASSERT_EQ(s.squeeze(0), Shape({3, 1, 4}));
    ASSERT_EQ(s.squeeze(2), Shape({1, 3, 4}));
    ASSERT_EQ(s.squeeze_all(), Shape({3, 4}));

    Shape s2({3, 4});
    ASSERT_EQ(s2.unsqueeze(0), Shape({1, 3, 4}));
    ASSERT_EQ(s2.unsqueeze(1), Shape({3, 1, 4}));
    ASSERT_EQ(s2.unsqueeze(2), Shape({3, 4, 1}));
}

AIC_TEST(ShapeTest, BroadcastingRules) {
    Shape a({2, 1, 4});
    Shape b({3, 4});
    ASSERT_TRUE(Shape::can_broadcast(a, b));
    ASSERT_EQ(Shape::broadcast(a, b), Shape({2, 3, 4}));

    Shape c({2, 3, 4});
    Shape d({5, 4});
    ASSERT_FALSE(Shape::can_broadcast(c, d));
    ASSERT_THROW(Shape::broadcast(c, d), Error);
}

AIC_TEST(ShapeTest, BroadcastStrides) {
    Shape orig({1, 4});
    auto orig_strides = orig.default_strides();  // [4, 1]
    Shape target({2, 3, 4});
    auto b_strides = Shape::broadcast_strides(orig, orig_strides, target);
    ASSERT_EQ(b_strides.size(), 3);
    ASSERT_EQ(b_strides[0], 0);  // Broadcasted dim
    ASSERT_EQ(b_strides[1], 0);  // Broadcasted dim
    ASSERT_EQ(b_strides[2], 1);  // Contiguous trailing dim
}
