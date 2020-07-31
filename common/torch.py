import torch
import numpy
import math
from . import numpy as cnumpy


def memory():
    """
    Get memory usage.

    :return: memory usage
    :rtype: str
    """

    index = torch.cuda.current_device()

    # results are in bytes
    # Decimal
    # Value 	Metric
    # 1000 	kB 	kilobyte
    # 1000^2 	MB 	megabyte
    # 1000^3 	GB 	gigabyte
    # 1000^4 	TB 	terabyte
    # 1000^5 	PB 	petabyte
    # 1000^6 	EB 	exabyte
    # 1000^7 	ZB 	zettabyte
    # 1000^8 	YB 	yottabyte
    # Binary
    # Value 	IEC 	JEDEC
    # 1024 	KiB 	kibibyte 	KB 	kilobyte
    # 1024^2 	MiB 	mebibyte 	MB 	megabyte
    # 1024^3 	GiB 	gibibyte 	GB 	gigabyte
    # 1024^4 	TiB 	tebibyte 	-
    # 1024^5 	PiB 	pebibyte 	-
    # 1024^6 	EiB 	exbibyte 	-
    # 1024^7 	ZiB 	zebibyte 	-
    # 1024^8 	YiB 	yobibyte 	-

    return '%g/%gMiB' % (
        BMiB(torch.cuda.memory_allocated(index) +
             torch.cuda.memory_cached(index)),
        BMiB(torch.cuda.max_memory_allocated(index) +
             torch.cuda.max_memory_cached(index)),
    )


def is_cuda(mixed):
    """
    Check if model/tensor is on CUDA.

    :param mixed: model or tensor
    :type mixed: torch.nn.Module or torch.autograd.Variable or torch.Tensor
    :return: on cuda
    :rtype: bool
    """

    assert isinstance(mixed, torch.nn.Module) or isinstance(mixed, torch.autograd.Variable) \
        or isinstance(mixed, torch.Tensor), 'mixed has to be torch.nn.Module, torch.autograd.Variable or torch.Tensor'

    is_cuda = False
    if isinstance(mixed, torch.nn.Module):
        is_cuda = True
        for parameters in list(mixed.parameters()):
            is_cuda = is_cuda and parameters.is_cuda
    if isinstance(mixed, torch.autograd.Variable):
        is_cuda = mixed.is_cuda
    if isinstance(mixed, torch.Tensor):
        is_cuda = mixed.is_cuda

    return is_cuda


def estimate_size(mixed):
    """
    Estimate tensor size.

    :param tensor: tensor or model
    :type tensor: numpy.ndarray, torch.tensor, torch.autograd.Variable or torch.nn.Module
    :return: size in bits
    :rtype: int
    """

    # PyTorch types:
    # Data type 	dtype 	CPU tensor 	GPU tensor
    # 32-bit floating point 	torch.float32 or torch.float 	torch.FloatTensor 	torch.cuda.FloatTensor
    # 64-bit floating point 	torch.float64 or torch.double 	torch.DoubleTensor 	torch.cuda.DoubleTensor
    # 16-bit floating point 	torch.float16 or torch.half 	torch.HalfTensor 	torch.cuda.HalfTensor
    # 8-bit integer (unsigned) 	torch.uint8 	torch.ByteTensor 	torch.cuda.ByteTensor
    # 8-bit integer (signed) 	torch.int8 	torch.CharTensor 	torch.cuda.CharTensor
    # 16-bit integer (signed) 	torch.int16 or torch.short 	torch.ShortTensor 	torch.cuda.ShortTensor
    # 32-bit integer (signed) 	torch.int32 or torch.int 	torch.IntTensor 	torch.cuda.IntTensor
    # 64-bit integer (signed) 	torch.int64 or torch.long 	torch.LongTensor 	torch.cuda.LongTensor

    # Numpy types:
    # Data type 	Description
    # bool_ 	Boolean (True or False) stored as a byte
    # int_ 	Default integer type (same as C long; normally either int64 or int32)
    # intc 	Identical to C int (normally int32 or int64)
    # intp 	Integer used for indexing (same as C ssize_t; normally either int32 or int64)
    # int8 	Byte (-128 to 127)
    # int16 	Integer (-32768 to 32767)
    # int32 	Integer (-2147483648 to 2147483647)
    # int64 	Integer (-9223372036854775808 to 9223372036854775807)
    # uint8 	Unsigned integer (0 to 255)
    # uint16 	Unsigned integer (0 to 65535)
    # uint32 	Unsigned integer (0 to 4294967295)
    # uint64 	Unsigned integer (0 to 18446744073709551615)
    # float_ 	Shorthand for float64.
    # float16 	Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
    # float32 	Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
    # float64 	Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
    # complex_ 	Shorthand for complex128.
    # complex64 	Complex number, represented by two 32-bit floats (real and imaginary components)
    # complex128 	Complex number, represented by two 64-bit floats (real and imaginary components)

    types8 = [
        torch.uint8, torch.int8,
        numpy.int8, numpy.uint8, numpy.bool_,
    ]

    types16 = [
        torch.float16, torch.half,
        torch.int16, torch.short,
        numpy.int16, numpy.uint16, numpy.float16,
    ]

    types32 = [
        torch.float32, torch.float,
        torch.int32, torch.int,
        numpy.int32, numpy.uint32, numpy.float32,
    ]

    types64 = [
        torch.float64, torch.double,
        torch.int64, torch.long,
        numpy.int64, numpy.uint64, numpy.float64, numpy.complex64,
        numpy.int_, numpy.float_
    ]

    types128 = [
        numpy.complex_, numpy.complex128
    ]

    if isinstance(mixed, torch.nn.Module):

        size = 0
        modules = mixed.modules()
        for module in modules:
            for parameters in list(module.parameters()):
                size += estimate_size(parameters)
        return size

    if isinstance(mixed, (torch.Tensor, numpy.ndarray)):

        if mixed.dtype in types128:
            bits = 128
        elif mixed.dtype in types64:
            bits = 64
        elif mixed.dtype in types32:
            bits = 32
        elif mixed.dtype in types16:
            bits = 16
        elif mixed.dtype in types8:
            bits = 8
        else:
            assert False, 'could not identify torch.Tensor or numpy.ndarray type %s' % mixed.type()

        size = numpy.prod(mixed.shape)
        return size * bits

    elif isinstance(mixed, torch.autograd.Variable):
        return estimate_size(mixed.data)
    else:
        assert False, 'unsupported tensor size for estimating size, either numpy.ndarray, torch.tensor or torch.autograd.Variable'


def bits2MiB(bits):
    """
    Convert bits to MiB.

    :param bits: number of bits
    :type bits: int
    :return: MiB
    :rtype: float
    """

    return bits / (8 * 1024 * 1024)


def bits2MB(bits):
    """
    Convert bits to MB.

    :param bits: number of bits
    :type bits: int
    :return: MiB
    :rtype: float
    """

    return bits / (8 * 1000 * 1000)


def bytes2MiB(bytes):
    """
    Convert bytes to MiB.

    :param bytes: number of bytes
    :type bytes: int
    :return: MiB
    :rtype: float
    """

    return bytes / (1024 * 1024)


def bytes2MB(bytes):
    """
    Convert bytes to MB.

    :param bytes: number of bytes
    :type bytes: int
    :return: MiB
    :rtype: float
    """

    return bytes / (1000 * 1000)


bMiB = bits2MiB
BMiB = bytes2MiB
bMB = bits2MB
BMB = bytes2MB


def binary_labels(classes):
    """
    Convert 0,1 labels to -1,1 labels.

    :param classes: classes as B x 1
    :type classes: torch.autograd.Variable or torch.Tensor
    """

    classes[classes == 0] = -1
    return classes


def one_hot(classes, C):
    """
    Convert class labels to one-hot vectors.

    :param classes: classes as B x 1
    :type classes: torch.autograd.Variable or torch.Tensor
    :param C: number of classes
    :type C: int
    :return: one hot vector as B x C
    :rtype: torch.autograd.Variable or torch.Tensor
    """

    assert isinstance(classes, torch.autograd.Variable) or isinstance(
        classes, torch.Tensor), 'classes needs to be torch.autograd.Variable or torch.Tensor'
    assert len(classes.size()) == 2 or len(classes.size()
                                           ) == 1, 'classes needs to have rank 2 or 1'
    assert C > 0

    if len(classes.size()) < 2:
        classes = classes.view(-1, 1)

    one_hot = torch.Tensor(classes.size(0), C)
    if is_cuda(classes):
        one_hot = one_hot.cuda()

    if isinstance(classes, torch.autograd.Variable):
        one_hot = torch.autograd.Variable(one_hot)

    one_hot.zero_()
    one_hot.scatter_(1, classes, 1)

    return one_hot


def project_ball(tensor, epsilon=1, ord=2):
    """
    Compute the orthogonal projection of the input tensor (as vector) onto the L_ord epsilon-ball.

    **Assumes the first dimension to be batch dimension, which is preserved.**

    :param tensor: variable or tensor
    :type tensor: torch.autograd.Variable or torch.Tensor
    :param epsilon: radius of ball.
    :type epsilon: float
    :param ord: order of norm
    :type ord: int
    :return: projected vector
    :rtype: torch.autograd.Variable or torch.Tensor
    """

    assert isinstance(tensor, torch.Tensor) or isinstance(
        tensor, torch.autograd.Variable), 'given tensor should be torch.Tensor or torch.autograd.Variable'

    if ord == 0:
        assert epsilon >= 0
        sorted, _ = torch.sort(tensor.view(tensor.size()[0], -1), dim=1)
        k = int(math.ceil(epsilon))
        assert k > 0
        thresholds = sorted[:, -k]
        mask = (tensor >= expand_as(thresholds, tensor)).type(tensor.type())

        tensor *= mask
    elif ord == 1:
        # ! Does not allow differentiation obviously!
        cuda = is_cuda(tensor)
        array = tensor.detach().cpu().numpy()
        array = cnumpy.project_ball(array, epsilon=epsilon, ord=ord)
        tensor = torch.from_numpy(array)
        if cuda:
            tensor = tensor.cuda()
    elif ord == 2:
        size = tensor.size()
        flattened_size = numpy.prod(numpy.array(size[1:]))

        tensor = tensor.view(-1, flattened_size)
        clamped = torch.clamp(epsilon / torch.norm(tensor, 2, dim=1), max=1)
        clamped = clamped.view(-1, 1)

        tensor = tensor * clamped
        if len(size) == 4:
            tensor = tensor.view(-1, size[1], size[2], size[3])
        elif len(size) == 2:
            tensor = tensor.view(-1, size[1])
    elif ord == float('inf'):
        tensor = torch.clamp(tensor, min=-epsilon, max=epsilon)
    else:
        raise NotImplementedError()

    return tensor


def project_sphere(tensor, epsilon=1, ord=2):
    """
    Compute the orthogonal projection of the input tensor (as vector) onto the L_ord epsilon-ball.

    **Assumes the first dimension to be batch dimension, which is preserved.**

    :param tensor: variable or tensor
    :type tensor: torch.autograd.Variable or torch.Tensor
    :param epsilon: radius of ball.
    :type epsilon: float
    :param ord: order of norm
    :type ord: int
    :return: projected vector
    :rtype: torch.autograd.Variable or torch.Tensor
    """

    assert isinstance(tensor, torch.Tensor) or isinstance(
        tensor, torch.autograd.Variable), 'given tensor should be torch.Tensor or torch.autograd.Variable'

    size = tensor.size()
    flattened_size = numpy.prod(numpy.array(size[1:]))

    tensor = tensor.view(-1, flattened_size)
    tensor = tensor / torch.norm(tensor, dim=1, ord=ord).view(-1, 1)
    tensor *= epsilon

    if len(size) == 4:
        tensor = tensor.view(-1, size[1], size[2], size[3])
    elif len(size) == 2:
        tensor = tensor.view(-1, size[1])

    return tensor


def tensor_or_value(mixed):
    """
    Get tensor or single value.

    :param mixed: variable, tensor or value
    :type mixed: mixed
    :return: tensor or value
    :rtype: torch.Tensor or value
    """

    if isinstance(mixed, torch.Tensor):
        if mixed.numel() > 1:
            return mixed
        else:
            return mixed.item()
    elif isinstance(mixed, torch.autograd.Variable):
        return tensor_or_value(mixed.cpu().data)
    else:
        return mixed


def as_variable(mixed, cuda=False, requires_grad=False):
    """
    Get a tensor or numpy array as variable.

    :param mixed: input tensor
    :type mixed: torch.Tensor or numpy.ndarray
    :param device: gpu or not
    :type device: bool
    :param requires_grad: gradients
    :type requires_grad: bool
    :return: variable
    :rtype: torch.autograd.Variable
    """

    assert isinstance(mixed, numpy.ndarray) or isinstance(
        mixed, torch.Tensor), 'input needs to be numpy.ndarray or torch.Tensor'

    if isinstance(mixed, numpy.ndarray):
        mixed = torch.from_numpy(mixed)

    if cuda:
        mixed = mixed.cuda()
    return torch.autograd.Variable(mixed, requires_grad)


def tile(a, dim, n_tile):
    """
    Numpy-like tiling in torch.
    https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/2

    :param a: tensor
    :type a: torch.Tensor or torch.autograd.Variable
    :param dim: dimension to tile
    :type dim: int
    :param n_tile: number of tiles
    :type n_tile: int
    :return: tiled tensor
    :rtype: torch.Tensor or torch.autograd.Variable
    """

    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(numpy.concatenate(
        [init_dim * numpy.arange(n_tile) + i for i in range(init_dim)]))
    if is_cuda(a):
        order_index = order_index.cuda()
    return torch.index_select(a, dim, order_index)


def expand_as(tensor, tensor_as):
    """
    Expands the tensor using view to allow broadcasting.

    :param tensor: input tensor
    :type tensor: torch.Tensor or torch.autograd.Variable
    :param tensor_as: reference tensor
    :type tensor_as: torch.Tensor or torch.autograd.Variable
    :return: tensor expanded with singelton dimensions as tensor_as
    :rtype: torch.Tensor or torch.autograd.Variable
    """

    view = list(tensor.size())
    for i in range(len(tensor.size()), len(tensor_as.size())):
        view.append(1)

    return tensor.view(view)


def classification_error(logits, targets, reduction='mean'):
    """
    Accuracy.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduce: reduce to number or keep per element
    :type reduce: bool
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert logits.size()[0] == targets.size()[0]
    # or (len(list(targets.size())) == 2 and targets.size(1) == 1)
    assert len(list(targets.size())) == 1
    assert len(list(logits.size())) == 2

    if logits.size()[1] > 1:
        values, indices = torch.max(
            torch.nn.functional.softmax(logits, dim=1), dim=1)
    else:
        indices = torch.round(torch.nn.functional.sigmoid(logits)).view(-1)

    errors = torch.clamp(torch.abs(indices.long() - targets.long()), max=1)
    if reduction == 'mean':
        return torch.mean(errors.float())
    elif reduction == 'sum':
        return torch.sum(errors.float())
    else:
        return errors


def classification_loss(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert logits.size()[0] == targets.size()[0]
    # or (len(list(targets.size())) == 2 and targets.size(1) == 1)
    assert len(list(targets.size())) == 1
    assert len(list(logits.size())) == 2

    if logits.size()[1] > 1:
        return torch.nn.functional.cross_entropy(logits, targets, reduction=reduction)
    else:
        # probability 1 is class 1
        # probability 0 is class 0
        return torch.nn.functional.binary_cross_entropy(torch.nn.functional.sigmoid(logits).view(-1), targets.float(), reduction=reduction)


def reverse_cross_entropy_loss(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    reverse_targets = one_hot(targets, logits.size(1))
    #reverse_targets = 1 - reverse_targets

    loss = (classification_loss(logits, targets) -
            torch.sum(torch.log_softmax(logits, dim=1), dim=1)) / float(logits.size(1) - 1)
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    else:
        return loss


def max_p_loss(logits, targets=None, reduction='mean'):
    """
    Loss.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    max_log = torch.max(torch.nn.functional.softmax(logits, dim=1), dim=1)[0]
    if reduction == 'mean':
        return torch.mean(max_log)
    elif reduction == 'sum':
        return torch.sum(max_log)
    else:
        return max_log


def max_log_loss(logits, targets=None, reduction='mean'):
    """
    Loss.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    max_log = torch.max(
        torch.nn.functional.log_softmax(logits, dim=1), dim=1)[0]
    if reduction == 'mean':
        return torch.mean(max_log)
    elif reduction == 'sum':
        return torch.sum(max_log)
    else:
        return max_log


def entropy_loss(logits, targets=None, reduction='mean'):
    """
    Loss.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    entropy = torch.sum(torch.nn.functional.softmax(
        logits, dim=1) * torch.nn.functional.log_softmax(logits, dim=1), dim=1)
    if reduction == 'mean':
        return torch.mean(entropy)
    elif reduction == 'sum':
        return torch.sum(entropy)
    else:
        return entropy


def cross_entropy_divergence(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted logits
    :type logits: torch.autograd.Variable
    :param targets: target distributions
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert len(list(logits.size())) == len(list(targets.size()))
    assert logits.size()[0] == targets.size()[0]
    assert logits.size()[1] == targets.size()[1]
    assert logits.size()[1] > 1

    divergences = torch.sum(- targets *
                            torch.nn.functional.log_softmax(logits, dim=1), dim=1)
    if reduction == 'mean':
        return torch.mean(divergences)
    elif reduction == 'sum':
        return torch.sum(divergences)
    else:
        return divergences


def left_sided_kullback_leibler_divergence(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted logits
    :type logits: torch.autograd.Variable
    :param targets: target distributions
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert len(list(logits.size())) == len(list(targets.size()))
    assert logits.size()[0] == targets.size()[0]
    assert logits.size()[1] == targets.size()[1]
    assert logits.size()[1] > 1

    divergences = - torch.mul(targets, torch.log(targets + 1e-6) -
                              torch.nn.functional.log_softmax(logits, dim=1))
    if reduction == 'mean':
        return torch.mean(divergences)
    elif reduction == 'sum':
        return torch.sum(divergences)
    else:
        return divergences


def right_sided_kullback_leibler_divergence(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted logits
    :type logits: torch.autograd.Variable
    :param targets: target distributions
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert len(list(logits.size())) == len(list(targets.size()))
    assert logits.size()[0] == targets.size()[0]
    assert logits.size()[1] == targets.size()[1]
    assert logits.size()[1] > 1

    divergences = - torch.mul(torch.nn.functional.softmax(logits, dim=1),
                              torch.nn.functional.logsoftmax(logits, dim=1) - torch.log(targets + 1e-6))
    if reduction == 'mean':
        return torch.mean(divergences)
    elif reduction == 'sum':
        return torch.sum(divergences)
    else:
        return divergences


def jensen_shannon_divergence(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted logits
    :type logits: torch.autograd.Variable
    :param targets: target distributions
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    return 0.5 * left_sided_kullback_leibler_divergence(logits, targets) + 0.5 * left_sided_kullback_leibler_divergence(logits, targets)


def bhattacharyya_coefficient(logits, targets):
    """
    Loss.

    :param logits: predicted logits
    :type logits: torch.autograd.Variable
    :param targets: target distributions
    :type targets: torch.autograd.Variable
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert len(list(logits.size())) == len(list(targets.size()))
    assert logits.size()[0] == targets.size()[0]
    assert logits.size()[1] == targets.size()[1]
    assert logits.size()[1] > 1

    # http://www.cse.yorku.ca/~kosta/CompVis_Notes/bhattacharyya.pdf
    # torch.sqrt not differentiable at zero
    return torch.clamp(torch.sum(torch.sqrt(torch.nn.functional.softmax(logits, dim=1) * targets + 1e-8), dim=1), min=0, max=1)


def bhattacharyya_divergence(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted logits
    :type logits: torch.autograd.Variable
    :param targets: target distributions
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    divergences = - 2 * torch.log(bhattacharyya_coefficient(logits, targets))
    if reduction == 'mean':
        return torch.mean(divergences)
    elif reduction == 'sum':
        return torch.sum(divergences)
    else:
        return divergences


def hellinger_divergence(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted logits
    :type logits: torch.autograd.Variable
    :param targets: target distributions
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    # torch.sqrt not differentiable at zero
    divergences = torch.sqrt(
        1 - bhattacharyya_coefficient(logits, targets) + 1e-8)
    if reduction == 'mean':
        return torch.mean(divergences)
    elif reduction == 'sum':
        return torch.sum(divergences)
    else:
        return divergences


def bhattacharyya_angle_divergence(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted logits
    :type logits: torch.autograd.Variable
    :param targets: target distributions
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    # torch.sqrt not differentiable at zero
    divergences = torch.acos(
        0.005 + 0.99 * bhattacharyya_coefficient(logits, targets))
    if reduction == 'mean':
        return torch.mean(divergences)
    elif reduction == 'sum':
        return torch.sum(divergences)
    else:
        return divergences


def total_variation_divergence(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted logits
    :type logits: torch.autograd.Variable
    :param targets: target distributions
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert len(list(logits.size())) == len(list(targets.size()))
    assert logits.size()[0] == targets.size()[0]
    assert logits.size()[1] == targets.size()[1]
    assert logits.size()[1] > 1

    divergences = torch.sum(
        torch.abs(torch.nn.functional.softmax(logits, dim=1) - targets), dim=1)
    if reduction == 'mean':
        return torch.mean(divergences)
    elif reduction == 'sum':
        return torch.sum(divergences)
    else:
        return divergences


def constant_transition(perturbations, norm, epsilon=0.3, gamma=1):
    """
    Linear transition rule.

    :param perturbations: perturbations
    :type perturbations: torch.autograd.Variable
    :param norm: norm
    :type norm: attacks.norms.Norm
    :param epsilon: epsilon
    :type epsilon: float
    :param gamma: gamma
    :type gamma: float
    :return: gamma, norms
    :rtype: torch.autograd.Variable, torch.autograd.Variable
    """

    norms = norm(perturbations)
    ones = torch.ones_like(norms)
    ones[norms <= epsilon] = 0
    return ones, norms


def linear_transition(perturbations, norm, epsilon=0.3, gamma=1):
    """
    Linear transition rule.

    :param perturbations: perturbations
    :type perturbations: torch.autograd.Variable
    :param norm: norm
    :type norm: attacks.norms.Norm
    :param epsilon: epsilon
    :type epsilon: float
    :param gamma: gamma
    :type gamma: float
    :return: gamma, norms
    :rtype: torch.autograd.Variable, torch.autograd.Variable
    """

    norms = norm(perturbations)
    return torch.min(torch.ones_like(norms), gamma * norms / epsilon), norms


def inverse_power_transition(perturbations, norm, epsilon=0.3, gamma=1):
    """
    Power transition rule.

    :param perturbations: perturbations
    :type perturbations: torch.autograd.Variable
    :param norm: norm
    :type norm: attacks.norms.Norm
    :param epsilon: epsilon
    :type epsilon: float
    :param gamma: gamma
    :type gamma: float
    :return: gamma, norms
    :rtype: torch.autograd.Variable, torch.autograd.Variable
    """

    # returned value determines importance of uniform distribution:
    # (1 - ret)*one_hot + ret*uniform

    norms = norm(perturbations)
    return torch.pow(torch.min(torch.ones_like(norms), norms / epsilon), gamma), norms


def power_transition(perturbations, norm, epsilon=0.3, gamma=1):
    """
    Power transition rule.

    :param perturbations: perturbations
    :type perturbations: torch.autograd.Variable
    :param norm: norm
    :type norm: attacks.norms.Norm
    :param epsilon: epsilon
    :type epsilon: float
    :param gamma: gamma
    :type gamma: float
    :return: gamma, norms
    :rtype: torch.autograd.Variable, torch.autograd.Variable
    """

    # returned value determines importance of uniform distribution:
    # (1 - ret)*one_hot + ret*uniform

    norms = norm(perturbations)
    return 1 - torch.pow(1 - torch.min(torch.ones_like(norms), norms / epsilon), gamma), norms


def tanh_transition(perturbations, norm, epsilon=0.3, gamma=1):
    """
    Linear transition rule.

    :param perturbations: perturbations
    :type perturbations: torch.autograd.Variable
    :param norm: norm
    :type norm: attacks.norms.Norm
    :param epsilon: epsilon
    :type epsilon: float
    :param gamma: gamma
    :type gamma: float
    :return: gamma, norms
    :rtype: torch.autograd.Variable, torch.autograd.Variable
    """

    norms = norm(perturbations)
    return torch. tanh(1 / (epsilon / gamma) * norms)


def exponential_transition(perturbations, norm, epsilon=0.3, gamma=1):
    """
    Exponential transition rule.

    :param perturbations: perturbations
    :type perturbations: torch.autograd.Variable
    :param norm: norm
    :type norm: attacks.norms.Norm
    :param epsilon: epsilon
    :type epsilon: float
    :param gamma: gamma
    :type gamma: float
    :return: gamma, norms
    :rtype: torch.autograd.Variable, torch.autograd.Variable
    """

    norms = norm(perturbations)
    return 1 - torch.exp(-gamma * norms), norms


class View(torch.nn.Module):
    """
    Simple view layer.
    """

    def __init__(self, *args):
        """
        Constructor.

        :param args: shape
        :type args: [int]
        """

        super(View, self).__init__()

        self.shape = args

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return input.view(self.shape)


class Clamp(torch.nn.Module):
    """
    Wrapper for clamp.
    """

    def __init__(self, min=0, max=1):
        """
        Constructor.
        """

        super(Clamp, self).__init__()

        self.min = min
        """ (float) Min value. """

        self.max = max
        """ (float) Max value. """

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return torch.clamp(torch.clamp(input, min=self.min), max=self.max)


class Scale(torch.nn.Module):
    """
    Simply scaling layer, mainly to allow simple saving and loading.
    """

    def __init__(self, shape):
        """
        Constructor.

        :param shape: shape
        :type shape: [int]
        """

        super(Scale, self).__init__()

        self.weight = torch.nn.Parameter(torch.zeros(shape))  # min
        self.bias = torch.nn.Parameter(torch.ones(shape))  # max

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return expand_as(self.weight, input) + torch.mul(expand_as(self.bias, input) - expand_as(self.weight, input), input)



class Normalize(torch.nn.Module):
    """
    Normalization layer to be learned.
    """

    def __init__(self, n_channels):
        """
        Constructor.

        :param n_channels: number of channels
        :type n_channels: int
        """

        super(Normalize, self).__init__()

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = torch.nn.Parameter(torch.ones(n_channels))
        self.bias = torch.nn.Parameter(torch.zeros(n_channels))

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return (input - self.bias.view(1, -1, 1, 1)) / self.weight.view(1, -1, 1, 1)
