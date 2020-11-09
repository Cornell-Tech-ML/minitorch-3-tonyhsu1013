from numba import cuda
import numba
from .tensor_data import (
    count,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)
import numpy
import numpy as np

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

count = cuda.jit(device=True)(count)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)


def tensor_map(fn):
    """
    CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.
        out (array): storage for out tensor.
        out_shape (array): shape for out tensor.
        out_strides (array): strides for out tensor.
        out_size (array): size for out tensor.
        in_storage (array): storage for in tensor.
        in_shape (array): shape for in tensor.
        in_strides (array): strides for in tensor.

    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, out_size, in_storage, in_shape, in_strides):
        # TODO: Implement for Task 3.3.
        # Find my position.
        x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Create the indices
        out_index = cuda.local.array(MAX_DIMS, dtype=numba.int32)
        in_index = cuda.local.array(MAX_DIMS, dtype=numba.int32)

        if x < out_size:
            # Get the current index
            count(int(x), out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # Find position in storage
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            # Perform the map function
            out[o] = fn(in_storage[j])

    return cuda.jit()(_map)


def map(fn):
    # CUDA compile your kernel
    f = tensor_map(cuda.jit(device=True)(fn))

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)

        # Instantiate and run the cuda kernel.
        threadsperblock = 32
        blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
        f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function mappings two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        b_storage (array): storage for `b` tensor.
        b_shape (array): shape for `b` tensor.
        b_strides (array): strides for `b` tensor.

    Returns:
        None : Fills in `out`
    """

    def _zip(
        out,
        out_shape,
        out_strides,
        out_size,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        # TODO: Implement for Task 3.3.
        # Find thread position
        x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Create the indices
        out_index = cuda.local.array(MAX_DIMS, dtype=numba.int32)
        a_index = cuda.local.array(MAX_DIMS, dtype=numba.int32)
        b_index = cuda.local.array(MAX_DIMS, dtype=numba.int32)

        if x < out_size:
            # Get the current index
            count(int(x), out_shape, out_index)

            # Get position in the storage
            o = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            j = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            k = index_to_position(b_index, b_strides)

            # Perform the zip function
            out[o] = fn(a_storage[j], b_storage[k])

    return cuda.jit()(_zip)


def zip(fn):
    f = tensor_zip(cuda.jit(device=True)(fn))

    def ret(a, b):
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        threadsperblock = 32
        blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
        f[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )
        return out

    return ret


def tensor_reduce(fn):
    """
    CUDA higher-order tensor reduce function.

    Args:
        fn: reduction function maps two floats to float.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        reduce_shape (array): shape of reduction (1 for dimension kept, shape value for dimensions summed out)
        reduce_size (int): size of reduce shape

    Returns:
        None : Fills in `out`
    """

    def _reduce(
        out,
        out_shape,
        out_strides,
        out_size,
        a_storage,
        a_shape,
        a_strides,
        reduce_shape,
        reduce_size,
    ):
        # TODO: Implement for Task 3.3.
        # Find thread position
        x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Create indices array
        out_index = cuda.local.array(MAX_DIMS, dtype=numba.int32)
        a_index = cuda.local.array(MAX_DIMS, dtype=numba.int32)

        if x < out_size:
            # Get the current indices
            count(x, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            # Iterating through the dimension we're reducing
            for s in range(reduce_size):
                # Figure out the position we're reducing in this iteration
                count(s, reduce_shape, a_index)
                # Reducing by going over the dimension we're not reducing
                for k in range(len(reduce_shape)):
                    if reduce_shape[k] != 1:
                        out_index[k] = a_index[k]
                # Map to corresponding position in the storage
                j = index_to_position(out_index, a_strides)
                # Reduce at the position by aggregating the function
                out[o] = fn(out[o], a_storage[j])

    return cuda.jit()(_reduce)


def reduce(fn, start=0.0):
    f = tensor_reduce(cuda.jit(device=True)(fn))

    def ret(a, dims=None, out=None):
        old_shape = None
        if out is None:
            out_shape = list(a.shape)
            for d in dims:
                out_shape[d] = 1
            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start
        else:
            old_shape = out.shape
            diff = len(a.shape) - len(out.shape)
            out = out.view(*([1] * diff + list(old_shape)))

        # Assume they are the same dim
        assert len(out.shape) == len(a.shape)

        # Create a reduce shape / reduce size
        reduce_shape = []
        reduce_size = 1
        for i, s in enumerate(a.shape):
            if out.shape[i] == 1:
                reduce_shape.append(s)
                reduce_size *= s
            else:
                reduce_shape.append(1)

        threadsperblock = 32
        blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock

        f[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), np.array(reduce_shape), reduce_size
        )
        # START CODE CHANGE
        if old_shape is not None:
            out = out.view(*old_shape)
        # END CODE CHANGE
        return out

    return ret


@cuda.jit()
def tensor_matrix_multiply(
    out,
    out_shape,
    out_strides,
    out_size,
    a_storage,
    a_shape,
    a_strides,
    b_storage,
    b_shape,
    b_strides,
):
    """
    CUDA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as ::

        assert a_shape[-1] == b_shape[-2]

    Args:
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """

    # TODO: Implement for Task 3.4.
    # Find thread position
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if x < out_size:
        # Figure out how many dimensions there are in each tensors
        a_num_positions = len(a_shape)
        b_num_positions = len(b_shape)
        out_num_positions = len(out_shape)

        # Create the indices
        out_index = cuda.local.array(MAX_DIMS, dtype=numba.int32)
        a_index = cuda.local.array(MAX_DIMS, dtype=numba.int32)
        b_index = cuda.local.array(MAX_DIMS, dtype=numba.int32)

        # Find the current index
        count(x, out_shape, out_index)
        o = index_to_position(out_index, out_strides)

        # Figure out input indices from the corresponding output
        broadcast_index(out_index, out_shape, a_shape, a_index)
        broadcast_index(out_index, out_shape, b_shape, b_index)
        # Fix a position from output that we're multiplying by
        a_index[a_num_positions - 2] = out_index[out_num_positions - 2]
        b_index[b_num_positions - 1] = out_index[out_num_positions - 1]

        temp = 0
        # Iterating through dimension from input that we're multiplying by
        for s in range(a_shape[-1]):
            # Get the current position from iterating through the dimension
            a_index[a_num_positions - 1] = s
            b_index[b_num_positions - 2] = s
            # Map to the input storage
            j = index_to_position(a_index, a_strides)
            k = index_to_position(b_index, b_strides)
            # Reduce part as we're summing
            temp += a_storage[j] * b_storage[k]
        out[o] += temp


def matrix_multiply(a, b):
    """
    Tensor matrix multiply

    Should work for any tensor shapes that broadcast in the first n-2 dims and
    have ::

        assert a.shape[-1] == b.shape[-2]

    Args:
        a (:class:`Tensor`): tensor a
        b (:class:`Tensor`): tensor b

    Returns:
        :class:`Tensor` : new tensor
    """

    ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
    ls.append(a.shape[-2])
    ls.append(b.shape[-1])
    assert a.shape[-1] == b.shape[-2]
    out = a.zeros(tuple(ls))
    threadsperblock = 32
    blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
    tensor_matrix_multiply[blockspergrid, threadsperblock](
        *out.tuple(), out.size, *a.tuple(), *b.tuple()
    )

    return out


class CudaOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply
