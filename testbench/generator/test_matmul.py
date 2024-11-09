import torch
import gen_utils


def impl_gen_matmul(A_shape, B_shape, transpose_A=False, transpose_B=False):
    # Generate A and B tensors with reproducible values
    A_tensor = torch.arange(1, 1 + A_shape[0] * A_shape[1], dtype=torch.float32).reshape(A_shape)
    B_tensor = torch.arange(1, 1 + B_shape[0] * B_shape[1], dtype=torch.float32).reshape(B_shape)

    # Transpose if required
    if transpose_A:
        A_tensor = A_tensor.T
    if transpose_B:
        B_tensor = B_tensor.T

    # Perform matrix multiplication
    output_tensor = torch.matmul(A_tensor, B_tensor).detach().numpy()

    # Flatten input and output for C++ representation
    A_data = A_tensor.flatten().tolist()
    B_data = B_tensor.flatten().tolist()
    output_data = output_tensor.flatten().tolist()

    # Generate C++ code for the test function
    A_array = gen_utils.generate_stack_array_data("A_data", A_data)
    B_array = gen_utils.generate_stack_array_data("B_data", B_data)
    C_array = gen_utils.generate_stack_array("C_data", A_shape[0] * B_shape[1])
    output_array = gen_utils.generate_stack_array_data("expected", output_data)

    A_tensor_decl = gen_utils.generate_tensor("A", "A_data", A_shape, len(A_data))
    B_tensor_decl = gen_utils.generate_tensor("B", "B_data", B_shape, len(B_data))
    C_tensor_decl = gen_utils.generate_tensor("C", "C_data", (A_shape[0], B_shape[1]), A_shape[0] * B_shape[1])

    matmul_decl = f"Matmul{'ATrans' if transpose_A else ''}{'BTrans' if transpose_B else ''}"

    cpp_code = f"""
    {A_array}
    {B_array}
    {C_array}
    {A_tensor_decl}
    {B_tensor_decl}
    {C_tensor_decl}
    {matmul_decl}(A, B,C);
    {output_array}
    Tensor<qint32_t> R{{C.shape()}};
    R.allocateCustom(expected, {len(output_data)});
    return Equals(C, R);
"""
    return cpp_code


# Test configurations
A_shapes = [(4, 4), (3, 5), (5, 3), (5, 5)]  # Square and non-square matrices
B_shapes = [(4, 4), (5, 3), (3, 5), (5, 5)]  # Compatible shapes for each A
transpose_options = [False, True]  # Whether to transpose A or B


def matmul_gen_function(index):
    if index >= 10:
        return None  # Stop after 10 tests

    # Select shapes for A and B
    A_shape = A_shapes[index % len(A_shapes)]
    B_shape = B_shapes[index % len(B_shapes)]

    # Determine transpose flags
    transpose_A = (index == 0)  # Set A transposed on every third case
    transpose_B = (index == 3)  # Set B transposed on every third case

    return impl_gen_matmul(A_shape, B_shape, transpose_A, transpose_B)
