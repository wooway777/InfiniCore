#ifndef __INFINIOP_ELEMENTWISE_BANG_H__
#define __INFINIOP_ELEMENTWISE_BANG_H__

#include "../../../utils.h"
#include "../../devices/bang/common_bang.h"
#include "elementwise_bang_api.h"

namespace op::elementwise::bang {

struct DeviceImpl::Opaque {
    std::shared_ptr<device::bang::Handle::Internal> internal;

    Opaque(const std::shared_ptr<device::bang::Handle::Internal> &internal_)
        : internal(internal_) {}

    template <size_t N, typename Op, typename Tdata, typename... Args>
    infiniStatus_t calculateImpl(const op::elementwise::ElementwiseInfo &info,
                                 void *workspace,
                                 void *output,
                                 const std::vector<const void *> &inputs,
                                 cnrtQueue_t queue,
                                 Args &&...args) {

        auto output_size = info.getOutputSize();
        if (output_size == 0) {
            return INFINI_STATUS_SUCCESS;
        }

        // Device pointers
        const void **d_inputs_arr = nullptr;
        const bool *d_input_contiguous = nullptr;
        const bool *d_input_broadcasted = nullptr;
        const size_t *d_output_shape = nullptr;
        const ptrdiff_t *d_output_strides = nullptr;
        const size_t *d_input_shapes = nullptr;
        const ptrdiff_t *d_input_strides = nullptr;

        CHECK_STATUS(infoToDevice<N>(info, workspace, inputs.data(), d_inputs_arr,
                                     d_input_contiguous, d_input_broadcasted,
                                     d_output_shape, d_output_strides,
                                     d_input_shapes, d_input_strides));

        // return internal->useCnnl(queue, [&](cnnlHandle_t handle) {
        //     cnnlTensorDescriptor_t output_desc;
        //     CHECK_BANG(cnnlCreateTensorDescriptor(&output_desc));
        //     CHECK_BANG(device::bang::setCnnlTensorEx(output_desc, info));

        //     std::vector<cnnlTensorDescriptor_t> input_descs(N);
        //     for (size_t i = 0; i < N; ++i) {
        //         CHECK_BANG(cnnlCreateTensorDescriptor(&input_descs[i]));
        //         CHECK_BANG(device::bang::setCnnlTensorEx(input_descs[i], info));
        //     }

        //     // Launch the elementwise operation
        //     CHECK_BANG(Op::template launch<Tdata>(
        //         handle,
        //         output_desc, output,
        //         input_descs.data(), d_inputs_arr,
        //         args...));

        //     // Cleanup
        //     for (size_t i = 0; i < N; ++i) {
        //         CHECK_BANG(cnnlDestroyTensorDescriptor(input_descs[i]));
        //     }
        //     CHECK_BANG(cnnlDestroyTensorDescriptor(output_desc));

        //     return INFINI_STATUS_SUCCESS;
        // });
        return INFINI_STATUS_SUCCESS;
    }

private:
    template <size_t N>
    infiniStatus_t infoToDevice(
        const op::elementwise::ElementwiseInfo &info,
        void *workspace,
        const void *const *h_inputs_arr,
        const void **&d_inputs_arr,
        const bool *&d_input_contiguous,
        const bool *&d_input_broadcasted,
        const size_t *&d_output_shape,
        const ptrdiff_t *&d_output_strides,
        const size_t *&d_input_shapes,
        const ptrdiff_t *&d_input_strides) const {

        constexpr auto input_size = N;
        const auto ndim = info.getNdim();
        constexpr auto input_arr_size = N * sizeof(*h_inputs_arr);
        const int8_t *info_meta_start = info.getMetaStart();
        const int8_t *d_meta_start = reinterpret_cast<int8_t *>(workspace) + input_arr_size;

        // copy the input pointer array and meta to device
        CNRT_CHECK(cnrtMemcpy(workspace, (void*)h_inputs_arr, input_arr_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
        CNRT_CHECK(cnrtMemcpy((void *)d_meta_start, (void*)info_meta_start, info.getMetaMemSize(), CNRT_MEM_TRANS_DIR_HOST2DEV));

        // offset/assign the pointers
        d_inputs_arr = reinterpret_cast<const void **>(workspace);
        d_output_shape = reinterpret_cast<const size_t *>(d_meta_start);
        d_output_strides = reinterpret_cast<const ptrdiff_t *>(d_output_shape + ndim);
        d_input_shapes = reinterpret_cast<const size_t *>(d_output_strides + ndim);
        d_input_strides = reinterpret_cast<const ptrdiff_t *>(d_input_shapes + input_size * ndim);
        d_input_contiguous = reinterpret_cast<const bool *>(d_input_strides + input_size * ndim);
        d_input_broadcasted = reinterpret_cast<const bool *>(d_input_contiguous + input_size);

        return INFINI_STATUS_SUCCESS;
    }
};

template <typename... Args>
utils::Result<DeviceImpl *> DeviceImpl::create(Args &&...args) {
    auto opaque = std::make_shared<Opaque>(std::forward<Args>(args)...);
    return utils::Result<DeviceImpl *>(new DeviceImpl(opaque));
}

template <typename Op, typename Tdata, typename... Args>
infiniStatus_t DeviceImpl::calculate(const op::elementwise::ElementwiseInfo &info,
                                     void *workspace,
                                     void *output,
                                     const std::vector<const void *> &inputs,
                                     void *stream,
                                     Args &&...args) {
    constexpr size_t N = Op::num_inputs;
    return _opaque->calculateImpl<N, Op, Tdata>(
        info, workspace, output, inputs,
        reinterpret_cast<cnrtQueue_t>(stream),
        std::forward<Args>(args)...);
}
} // namespace op::elementwise::bang

#endif // __INFINIOP_ELEMENTWISE_BANG_H__
