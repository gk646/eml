import random
import torch
import gen_utils


def impl_gen_relu(input_shape):
    input_tensor_decl = gen_utils.generate_decl_tensor_range("A", input_shape,"float", -10)

    pad = torch.nn.ReLU()
    size = 1
    for i in input_shape:
        size *= i
    input_tensor = torch.arange(-10, -10 +  size ).reshape(input_shape)
    torch_output = pad(input_tensor).detach()
    output_data = torch_output.numpy()
    output_tensor_decl = gen_utils.generate_tensor_data("R", output_data.shape, output_data.flatten().tolist(),"float")

    # Generate the C++ function call
    cpp_code = f"""
    {input_tensor_decl}
    auto out = ReLU::forward(A);
    {output_tensor_decl}
    {gen_utils.test_macro}(R, out);
"""
    return cpp_code


# Define a list of input shapes and paddings for test cases
input_shapes = [
    (2, 3, 5, 4),
    (1, 1, 4, 5),
    (2, 3, 6, 6),
    (1, 1, 9, 8),
]


def gen_function(index):
    if index >= len(input_shapes):
        return None

    input_shape = input_shapes[index % len(input_shapes)]
    return impl_gen_relu(input_shape)
