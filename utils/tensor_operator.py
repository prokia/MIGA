from collections import abc as container_abc
from inspect import getfullargspec

import functools
import numpy as np
import torch
from torch.nn import functional as F


class GraphTensorPacking:
    def __init__(self, X, E, y=None):
        self.X = X
        self.E = E
        self.y = y

    def to(self, device=None, to_type=None):
        self.X = to_device(self.X, device, to_type)
        self.E = to_device(self.E, device, to_type)
        if self.y is not None:
            self.y = to_device(self.y, device, to_type)
        return self

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        if self.y is not None:
            self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


def clamp_to_prob(x, inplace=True):
    if not inplace:
        x = x.clone()
    x[x >= 1] = 1
    x[x < 0] = 0
    return x


def sigmoid_with_clamp(x, inf=1e-4, sup=1 - 1e-4):
    x = x.clone()
    x = torch.clamp(x.sigmoid_(), min=inf, max=sup)
    return x


def tensor2prob_map(in_tensor, trivial=False):
    cls_num = in_tensor.size()[1]
    prob_map = sigmoid_with_clamp(in_tensor) if cls_num == 1 else F.softmax(in_tensor, dim=1)
    if cls_num == 1 and trivial:
        prob_map = torch.cat([prob_map, 1 - prob_map], dim=1)
    return prob_map


def feature_transform(feature, t_matrix, device=None, dtype='float'):
    if device is not None:
        feature, t_matrix = tuple(map(lambda x: to_device(x, device, dtype), [feature, t_matrix]))
    grid = F.affine_grid(t_matrix, feature.size(), align_corners=True)
    re = F.grid_sample(feature, grid, align_corners=True)
    return re


def make_transform_matrix(mag):
    basic_matrix = np.array([[1., 0., 0.], [0., 1., 0.]])
    shift, scale_list, shift_index_list, scale_index_list = \
        [-mag, mag], [1 - mag, 1 / (1 - mag)], [(0, 2), (1, 2)], [(0, 0), (1, 1)]
    transform_matrix_list = [basic_matrix]
    for op, scale in zip(shift, scale_list):
        for shift_index, scale_index in zip(shift_index_list, scale_index_list):
            plus = np.zeros_like(basic_matrix)
            plus[shift_index] = op * 1
            transform_matrix_list.append(basic_matrix + plus)
            scale_matrix = basic_matrix.copy()
            scale_matrix[scale_index] *= scale
            transform_matrix_list.append(scale_matrix)

    return transform_matrix_list


def tensor2array(inp):
    if isinstance(inp, container_abc.Mapping):
        return {key: tensor2array(inp[key]) for key in inp}
    if isinstance(inp, list):
        return [tensor2array(item) for item in inp]
    if not isinstance(inp, torch.Tensor):
        return inp
    inp = inp.detach()
    if inp.device.type == 'cuda':
        inp = inp.cpu()
    return inp.numpy()


def to_device(inp, device=None, to_type=None):
    if inp is None:
        return inp
    if isinstance(inp, container_abc.Mapping):
        return {key: to_device(inp[key], device, to_type) for key in inp}
    if isinstance(inp, container_abc.Sequence):
        return [to_device(item, device, to_type) for item in inp]
    if isinstance(inp, np.ndarray):
        inp = torch.as_tensor(inp)
    if to_type is None or to_type == 'identity':
        if device == 'identity' or device is None:
            return inp
        return inp.to(device)
    if device is None or device == 'identity':
        device = inp.device
    return inp.to(device).__getattribute__(to_type)()


def cast_tensor_type(inputs, src_type, dst_type):
    """Recursively convert Tensor in inputs from src_type to dst_type.

    Args:
        inputs: Inputs that to be casted.
        src_type (torch.dtype): Source type..
        dst_type (torch.dtype): Destination type.

    Returns:
        The same type with inputs, but all contained Tensors have been cast.
    """
    if isinstance(inputs, torch.Tensor):
        return inputs.to(dst_type)
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, container_abc.Mapping):
        return type(inputs)({
            k: cast_tensor_type(v, src_type, dst_type)
            for k, v in inputs.items()
        })
    elif isinstance(inputs, container_abc.Iterable):
        return type(inputs)(
            cast_tensor_type(item, src_type, dst_type) for item in inputs)
    else:
        return inputs


def force_fp32(apply_to=None, out_fp16=False):
    """Decorator to convert input arguments to fp32 in force.

    This decorator is useful when you write custom modules and want to support
    mixed precision training. If there are some inputs that must be processed
    in fp32 mode, then this decorator can handle it. If inputs arguments are
    fp16 tensors, they will be converted to fp32 automatically. Arguments other
    than fp16 tensors are ignored.

    Args:
        apply_to (Iterable, optional): The argument names to be converted.
            `None` indicates all arguments.
        out_fp16 (bool): Whether to convert the output back to fp16.

    Example:

        >>> import torch.nn as nn
        >>> class MyModule1(nn.Module):
        >>>
        >>>     # Convert x and y to fp32
        >>>     @force_fp32()
        >>>     def loss(self, x, y):
        >>>         pass

        >>> import torch.nn as nn
        >>> class MyModule2(nn.Module):
        >>>
        >>>     # convert pred to fp32
        >>>     @force_fp32(apply_to=('pred', ))
        >>>     def post_process(self, pred, others):
        >>>         pass
    """

    def force_fp32_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # check if the module has set the attribute `fp16_enabled`, if not,
            # just fallback to the original method.
            # if not isinstance(args[0], torch.nn.Module):
            #     raise TypeError('@force_fp32 can only be used to decorate the '
            #                     'method of nn.Module')
            # if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
            #     return old_func(*args, **kwargs)
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get the argument names to be casted
            args_to_cast = args_info.args if apply_to is None else apply_to
            # convert the args that need to be processed
            new_args = []
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(
                            cast_tensor_type(args[i], torch.half, torch.float))
                    else:
                        new_args.append(args[i])
            # convert the kwargs that need to be processed
            new_kwargs = dict()
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(
                            arg_value, torch.half, torch.float)
                    else:
                        new_kwargs[arg_name] = arg_value
            # apply converted arguments to the decorated method
            output = old_func(*new_args, **new_kwargs)
            # cast the results back to fp32 if necessary
            if out_fp16:
                output = cast_tensor_type(output, torch.float, torch.half)
            return output

        return new_func

    return force_fp32_wrapper


class TemporaryGrad(object):
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        torch.set_grad_enabled(self.prev)


def make_one_hot(in_tensor, cls):
    if len(in_tensor) == 0:
        return in_tensor
    in_shape = tuple(in_tensor.size())
    in_tensor = in_tensor.view(-1)
    res_one_hot = torch.zeros(in_tensor.size() + (cls,), device=in_tensor.device)
    select_index = (torch.arange(0, in_tensor.size()[-1]).long(), in_tensor.long())
    res_one_hot[select_index] = 1
    res = res_one_hot.reshape(in_shape[:-1] + (cls,))
    return res


def repeat_on_dim(in_tensor, repeat_num, dim=0):
    dims_num = list(in_tensor.size())
    sq = in_tensor.unsqueeze(dim)
    repeat_v = np.ones(len(dims_num) + 1, dtype=np.int)
    repeat_v[dim + 1] = repeat_num
    rep = sq.repeat(*list(repeat_v))
    dims_num[dim] *= repeat_num
    res = rep.reshape(dims_num)
    return res


# ############# graphs
def convert_node_matrix(x):
    atom_idx = torch.nonzero(x, as_tuple=True)
    atom = x[atom_idx]
    atom = torch.unsqueeze(atom, -1)
    return atom, atom_idx


def convert_edge_matrix(edge):
    pass
