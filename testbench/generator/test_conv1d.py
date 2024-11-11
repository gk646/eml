import random
import torch

import gen_utils

input_sizes = [4, 5, 6]
kernels = [1, 2, 3]
strides = [3, 2, 1]
channels_cases = [(1, 1), (3, 1), (1, 3)]
paddings = [0]
padding_modes = ["ZEROS", "REFLECT", "REPLICATE", "CIRCULAR"]


def generate_conv1d_layer(name, in_channels, out_channels, kernel_size, stride, padding, use_bias, padding_mode):
    """Generate C++ Conv2D layer initialization."""
    padding_mode_str = f"PaddingMode::{padding_mode}"
    bias_str = str(use_bias).lower()
    return f"Conv1D<qint32_t> {name}{{ {in_channels}, {out_channels}, {kernel_size}, {stride}, {padding}, {bias_str}, {padding_mode_str} }};"


def impl_gen_conv1d(input_size, in_channels, out_channels, kernel_size, stride, padding, use_bias, padding_mode):
    input_tensor = torch.arange(1, 1 + in_channels * input_size, dtype=torch.float32).reshape(in_channels, input_size)

    conv_layer = torch.nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=use_bias,
        padding_mode=padding_mode.lower()
    )

    torch.nn.init.constant_(conv_layer.weight, 1.0)
    if use_bias:
        torch.nn.init.constant_(conv_layer.bias, 1.0)

    output_tensor = conv_layer(input_tensor).detach().numpy()

    input_data = input_tensor.flatten().tolist()
    output_data = output_tensor.flatten().tolist()

    input_shape = (in_channels, 1, input_size)
    input_tensor_decl = gen_utils.generate_decl_tensor_range("A", input_shape)

    output_shape = (out_channels, 1, output_tensor.shape[-1])
    output_tensor_decl = gen_utils.generate_tensor_data("R", output_shape, output_data)

    conv2d_layer = generate_conv1d_layer("layer", in_channels, out_channels, kernel_size, stride, padding, use_bias,
                                         padding_mode)

    cpp_code = f"""
    {input_tensor_decl}
    {conv2d_layer}
    Fill(layer.weights, 1);
    {"Fill(layer.biases, 1);" if use_bias else ""}
    const auto out = layer.forward(A);

    {output_tensor_decl}
    {gen_utils.test_macro}(R, out);
"""
    return cpp_code


def gen_function(index):
    direction = 1 if index % 2 == 0 else -1
    random.shuffle(input_sizes)
    random.shuffle(kernels)
    random.shuffle(strides)
    random.shuffle(channels_cases)
    random.shuffle(paddings)
    random.shuffle(padding_modes)

    input_size = input_sizes[(index * direction) % len(input_sizes)]
    in_channels, out_channels = channels_cases[(index * direction) % len(channels_cases)]
    kernel_size = kernels[(index * direction) % len(kernels)]
    stride = strides[(index * direction) % len(strides)]
    padding = paddings[(index * direction) % len(paddings)]
    use_bias = (index % 2 == 0)
    padding_mode = padding_modes[(index * direction) % len(padding_modes)]
    return impl_gen_conv1d(input_size, in_channels, out_channels, kernel_size, stride, padding, use_bias, padding_mode)
