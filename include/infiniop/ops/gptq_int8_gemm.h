#ifndef __INFINIOP_GPTQ_INT8_GEMM_API_H__
#define __INFINIOP_GPTQ_INT8_GEMM_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopGPTQInt8GemmDescriptor_t;

__C __export infiniStatus_t infiniopCreateGPTQInt8GemmDescriptor(infiniopHandle_t handle,
                                                                 infiniopGPTQInt8GemmDescriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t c,
                                                                 infiniopTensorDescriptor_t a,
                                                                 infiniopTensorDescriptor_t qweight,
                                                                 infiniopTensorDescriptor_t scales,
                                                                 infiniopTensorDescriptor_t qzeros,
                                                                 infiniopTensorDescriptor_t g_idx);

__C __export infiniStatus_t infiniopGetGPTQInt8GemmWorkspaceSize(infiniopGPTQInt8GemmDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopGPTQInt8Gemm(infiniopGPTQInt8GemmDescriptor_t desc,
                                                 void *workspace,
                                                 size_t workspace_size,
                                                 void *c,
                                                 const void *a,
                                                 const void *qweight,
                                                 const void *scales,
                                                 const void *qzeros,
                                                 const void *g_idx,
                                                 void *stream);

__C __export infiniStatus_t infiniopDestroyGPTQInt8GemmDescriptor(infiniopGPTQInt8GemmDescriptor_t desc);

#endif
