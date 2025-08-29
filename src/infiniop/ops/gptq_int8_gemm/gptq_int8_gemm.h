#ifndef __GPTQ_INT8_GEMM_H__
#define __GPTQ_INT8_GEMM_H__

#include "../../operator.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::gptq_int8_gemm::NAMESPACE {                    \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        infiniDtype_t _dtype;                                    \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            infiniDtype_t dtype,                                 \
            size_t workspace_size_,                              \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _dtype(dtype),                                     \
              _info(info),                                       \
              _workspace_size(workspace_size_) {}                \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t c_desc,                   \
            infiniopTensorDescriptor_t a_desc,                   \
            infiniopTensorDescriptor_t qweight_desc,             \
            infiniopTensorDescriptor_t scales_desc,              \
            infiniopTensorDescriptor_t qzeros_desc,              \
            infiniopTensorDescriptor_t g_idx_desc);              \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *c,                                             \
            float beta,                                          \
            const void *a,                                       \
            const void *qweight,                                 \
            const void *scales,                                  \
            const void *qzeros,                                  \
            const void *g_idx,                                   \
            float alpha,                                         \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __GEMM_H__
