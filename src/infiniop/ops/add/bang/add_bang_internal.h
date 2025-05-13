#ifndef __ADD_BANG_INTERNAL_H__
#define __ADD_BANG_INTERNAL_H__

#include "../../../elementwise/bang/elementwise_bang.h"

namespace op::add::bang {

typedef struct AddOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __mlu_device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, cn_fp16_t>) {
            return cn_fp16_add(a, b);
        } else if constexpr (std::is_same_v<T, float>) {
            return cn_float_add(a, b);
        } else {
            return a + b;
        }
    }
} AddOp;

} // namespace op::add::cambricon

#endif // __ADD_BANG_INTERNAL_H__
