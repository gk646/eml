def generate_stack_array_data(name, data, dtype="qint32_t"):
    """Generate a C++ stack array from a Python list of data."""
    data_str = ', '.join([f"{int(x)}" for x in data])
    return f"{dtype} {name}[] = {{ {data_str} }};"


def generate_stack_array(name, len, dtype="qint32_t"):
    """Generate a C++ stack array from a Python list of data."""
    return f"{dtype} {name}[{len}] ;\n"


def generate_tensor(name, array, shape, data_size, dtype="qint32_t"):
    """Generate C++ tensor initialization and allocation."""
    shape_str = ', '.join(map(str, shape))
    return f"Tensor<{dtype}> {name}{{ {shape_str} }};\n{name}.allocateCustom({array}, {data_size});"


def generate_tensor_range(name, array, shape, data_size, dtype="qint32_t"):
    """Generate C++ tensor initialization and allocation."""
    shape_str = ', '.join(map(str, shape))
    return f"Tensor<{dtype}> {name}{{ {shape_str} }};\n{name}.allocateCustom({array}, {data_size});\nArange({name},0);"
