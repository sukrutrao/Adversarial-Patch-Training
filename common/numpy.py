import numpy
import scipy.stats
import math


def one_hot(array, N):
    """
    Convert an array of numbers to an array of one-hot vectors.

    :param array: classes to convert
    :type array: numpy.ndarray
    :param N: number of classes
    :type N: int
    :return: one-hot vectors
    :rtype: numpy.ndarray
    """

    array = array.astype(int)
    assert numpy.max(array) < N
    assert numpy.min(array) >= 0

    one_hot = numpy.zeros((array.shape[0], N))
    one_hot[numpy.arange(array.shape[0]), array] = 1
    return one_hot


def expand_as(array, array_as):
    """
    Expands the tensor using view to allow broadcasting.

    :param array: input tensor
    :type array: numpy.ndarray
    :param array_as: reference tensor
    :type array_as: torch.Tensor or torch.autograd.Variable
    :return: tensor expanded with singelton dimensions as tensor_as
    :rtype: torch.Tensor or torch.autograd.Variable
    """

    shape = list(array.shape)
    for i in range(len(array.shape), len(array_as.shape)):
        shape.append(1)

    return array.reshape(shape)


def concatenate(array1, array2, axis=0):
    """
    Basically a wrapper for numpy.concatenate, with the exception
    that the array itself is returned if its None or evaluates to False.

    :param array1: input array or None
    :type array1: mixed
    :param array2: input array
    :type array2: numpy.ndarray
    :param axis: axis to concatenate
    :type axis: int
    :return: concatenated array
    :rtype: numpy.ndarray
    """

    assert isinstance(array2, numpy.ndarray)
    if array1 is not None:
        assert isinstance(array1, numpy.ndarray)
        return numpy.concatenate((array1, array2), axis=axis)
    else:
        return array2


def exponential_norm(batch_size, dim, epsilon=1, ord=2):
    """
    Sample vectors uniformly by norm and direction separately.

    :param batch_size: how many vectors to sample
    :type batch_size: int
    :param dim: dimensionality of vectors
    :type dim: int
    :param epsilon: epsilon-ball
    :type epsilon: float
    :param ord: norm to use
    :type ord: int
    :return: batch_size x dim tensor
    :rtype: numpy.ndarray
    """

    random = numpy.random.randn(batch_size, dim)
    random /= numpy.repeat(numpy.linalg.norm(random, ord=ord, axis=1).reshape(-1, 1), axis=1, repeats=dim)
    random *= epsilon

    truncated_normal = scipy.stats.truncexpon.rvs(1, loc=0, scale=0.9, size=(batch_size, 1))
    random *= numpy.repeat(truncated_normal, axis=1, repeats=dim)

    return random


def uniform_norm(batch_size, dim, epsilon=1, ord=2):
    """
    Sample vectors uniformly by norm and direction separately.

    :param batch_size: how many vectors to sample
    :type batch_size: int
    :param dim: dimensionality of vectors
    :type dim: int
    :param epsilon: epsilon-ball
    :type epsilon: float
    :param ord: norm to use
    :type ord: int
    :return: batch_size x dim tensor
    :rtype: numpy.ndarray
    """

    random = numpy.random.randn(batch_size, dim)
    random /= numpy.repeat(numpy.linalg.norm(random, ord=ord, axis=1).reshape(-1, 1), axis=1, repeats=dim)
    random *= epsilon
    uniform = numpy.random.uniform(0, 1, (batch_size, 1))  # exponent is only difference!
    random *= numpy.repeat(uniform, axis=1, repeats=dim)

    return random


def uniform_ball(batch_size, dim, epsilon=1, ord=2):
    """
    Sample vectors uniformly in the n-ball.

    See Harman et al., On decompositional algorithms for uniform sampling from n-spheres and n-balls.

    :param batch_size: how many vectors to sample
    :type batch_size: int
    :param dim: dimensionality of vectors
    :type dim: int
    :param epsilon: epsilon-ball
    :type epsilon: float
    :param ord: norm to use
    :type ord: int
    :return: batch_size x dim tensor
    :rtype: numpy.ndarray
    """

    random = numpy.random.randn(batch_size, dim)
    random /= numpy.repeat(numpy.linalg.norm(random, ord=ord, axis=1).reshape(-1, 1), axis=1, repeats=dim)
    random *= epsilon
    uniform = numpy.random.uniform(0, 1, (batch_size, 1)) ** (1. / dim)
    random *= numpy.repeat(uniform, axis=1, repeats=dim)

    return random


def uniform_sphere(batch_size, dim, epsilon=1, ord=2):
    """
    Sample vectors uniformly on the n-sphere.

    See Harman et al., On decompositional algorithms for uniform sampling from n-spheres and n-balls.

    :param batch_size: how many vectors to sample
    :type batch_size: int
    :param dim: dimensionality of vectors
    :type dim: int
    :param epsilon: epsilon-ball
    :type epsilon: float
    :param ord: norm to use
    :type ord: int
    :return: batch_size x dim tensor
    :rtype: numpy.ndarray
    """

    random = numpy.random.randn(batch_size, dim)
    random /= numpy.repeat(numpy.linalg.norm(random, ord=ord, axis=1).reshape(-1, 1), axis=1, repeats=dim)
    random *= epsilon

    return random


def truncated_normal(size, lower=-2, upper=2):
    """
    Sample from truncated normal.

    See https://stackoverflow.com/questions/18441779/how-to-specify-upper-and-lower-limits-when-using-numpy-random-normal.

    :param size: size of vector
    :type size: [int]
    :param lower: lower bound
    :type lower: float
    :param upper: upper bound
    :type upper: float
    :return: batch_size x dim tensor
    :rtype: numpy.ndarray
    """

    return scipy.stats.truncnorm.rvs(lower, upper, size=size)


def project_simplex(v, s=1):
    """
    Taken from https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246.

    Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """

    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s

    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and numpy.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = numpy.sort(v)[::-1]
    cssv = numpy.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = numpy.nonzero(u * numpy.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - s) / rho
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = numpy.sort(v)[::-1]
    cssv = numpy.cumsum(u) - z
    ind = numpy.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = numpy.maximum(v - theta, 0)
    return w


def projection_simplex_pivot(v, z=1, random_state=None):
    rs = numpy.random.RandomState(random_state)
    n_features = len(v)
    U = numpy.arange(n_features)
    s = 0
    rho = 0
    while len(U) > 0:
        G = []
        L = []
        k = U[rs.randint(0, len(U))]
        ds = v[k]
        for j in U:
            if v[j] >= v[k]:
                if j != k:
                    ds += v[j]
                    G.append(j)
            elif v[j] < v[k]:
                L.append(j)
        drho = len(G) + 1
        if s + ds - (rho + drho) * v[k] < z:
            s += ds
            rho += drho
            U = L
        else:
            U = G
    theta = (s - z) / float(rho)
    return numpy.maximum(v - theta, 0)


def projection_simplex_bisection(v, z=1, tau=0.0001, max_iter=1000):
    lower = 0
    upper = numpy.max(v)
    current = numpy.inf

    for it in xrange(max_iter):
        if numpy.abs(current) / z < tau and current < 0:
            break

        theta = (upper + lower) / 2.0
        w = numpy.maximum(v - theta, 0)
        current = numpy.sum(w) - z
        if current <= 0:
            upper = theta
        else:
            lower = theta
    return w


def project_ball(array, epsilon=1, ord=2):
    """
    Compute the orthogonal projection of the input tensor (as vector) onto the L_ord epsilon-ball.

    **Assumes the first dimension to be batch dimension, which is preserved.**

    :param array: array
    :type array: numpy.ndarray
    :param epsilon: radius of ball.
    :type epsilon: float
    :param ord: order of norm
    :type ord: int
    :return: projected vector
    :rtype: torch.autograd.Variable or torch.Tensor
    """

    assert isinstance(array, numpy.ndarray), 'given tensor should be numpy.ndarray'

    if ord == 0:
        assert epsilon >= 1
        size = array.shape
        flattened_size = numpy.prod(numpy.array(size[1:]))

        array = array.reshape(-1, flattened_size)
        sorted = numpy.sort(array, axis=1)

        k = int(math.ceil(epsilon))
        thresholds = sorted[:, -k]

        mask = (array >= expand_as(thresholds, array)).astype(float)
        array *= mask
    elif ord == 1:
        size = array.shape
        flattened_size = numpy.prod(numpy.array(size[1:]))

        array = array.reshape(-1, flattened_size)

        for i in range(array.shape[0]):
            # compute the vector of absolute values
            u = numpy.abs(array[i])
            # check if v is already a solution
            if u.sum() <= epsilon:
                # L1-norm is <= s
                continue
            # v is not already a solution: optimum lies on the boundary (norm == s)
            # project *u* on the simplex
            #w = project_simplex(u, s=epsilon)
            w = projection_simplex_sort(u, z=epsilon)
            # compute the solution to the original problem on v
            w *= numpy.sign(array[i])
            array[i] = w

        if len(size) == 4:
            array = array.reshape(-1, size[1], size[2], size[3])
        elif len(size) == 2:
            array = array.reshape(-1, size[1])
    elif ord == 2:
        size = array.shape
        flattened_size = numpy.prod(numpy.array(size[1:]))

        array = array.reshape(-1, flattened_size)
        clamped = numpy.clip(epsilon/numpy.linalg.norm(array, 2, axis=1), a_min=None, a_max=1)
        clamped = clamped.reshape(-1, 1)

        array = array * clamped
        if len(size) == 4:
            array = array.reshape(-1, size[1], size[2], size[3])
        elif len(size) == 2:
            array = array.reshape(-1, size[1])
    elif ord == float('inf'):
        array = numpy.clip(array, a_min=-epsilon, a_max=epsilon)
    else:
        raise NotImplementedError()

    return array


def project_sphere(array, epsilon=1, ord=2):
    """
    Compute the orthogonal projection of the input tensor (as vector) onto the L_ord epsilon-ball.

    **Assumes the first dimension to be batch dimension, which is preserved.**

    :param array: variable or tensor
    :type array: torch.autograd.Variable or torch.Tensor
    :param epsilon: radius of ball.
    :type epsilon: float
    :param ord: order of norm
    :type ord: int
    :return: projected vector
    :rtype: torch.autograd.Variable or torch.Tensor
    """

    assert isinstance(array, numpy.ndarray), 'given tensor should be numpy.ndarray'

    size = array.shape
    flattened_size = numpy.prod(numpy.array(size[1:]))

    array = array.reshape(-1, flattened_size)
    array = array/numpy.linalg.norm(array, axis=1, ord=ord).reshape(-1, 1)
    array *= epsilon

    if len(size) == 4:
        array = array.reshape(-1, size[1], size[2], size[3])
    elif len(size) == 2:
        array = array.reshape(-1, size[1])

    return array


def project_orthogonal(basis, vectors, rank=None):
    """
    Project the given vectors on the basis using an orthogonal projection.

    :param basis: basis vectors to project on
    :type basis: numpy.ndarray
    :param vectors: vectors to project
    :type vectors: numpy.ndarray
    :return: projection
    :rtype: numpy.ndarray
    """

    # The columns of Q are an orthonormal basis of the columns of basis
    Q, R = numpy.linalg.qr(basis)
    if rank is not None and rank > 0:
        Q = Q[:, :rank]

    # As Q is orthogonal, the projection is
    beta = Q.T.dot(vectors)
    projection = Q.dot(beta)

    return projection


def project_lstsq(basis, vectors):
    """
    Project using least squares.

    :param basis: basis vectors to project on
    :type basis: numpy.ndarray
    :param vectors: vectors to project
    :type vectors: numpy.ndarray
    :return: projection
    :rtype: numpy.ndarray
    """

    x, _, _, _ = numpy.linalg.lstsq(basis, vectors)
    projection = basis.dot(x)

    return projection


def angles(vectors_a, vectors_b):
    """
    Compute angle between two sets of vectors.

    See https://people.eecs.berkeley.edu/~wkahan/Mindless.pdf.

    :param vectors_a:
    :param vectors_b:
    :return:
    """

    if len(vectors_b.shape) == 1:
        vectors_b = vectors_b.reshape(-1, 1)

    # Normalize vector
    norms_a = numpy.linalg.norm(vectors_a, ord=2, axis=0)
    norms_b = numpy.linalg.norm(vectors_b, ord=2, axis=0)

    norms_a = numpy.repeat(norms_a.reshape(1, -1), vectors_a.shape[0], axis=0)
    norms_b = numpy.repeat(norms_b.reshape(1, -1), vectors_b.shape[0], axis=0)

    vectors_a /= norms_a
    vectors_b /= norms_b

    term_1 = numpy.multiply(vectors_a, norms_b) - numpy.multiply(vectors_b, norms_a)
    term_1 = numpy.linalg.norm(term_1, ord=2, axis=0)

    term_2 = numpy.multiply(vectors_a, norms_b) + numpy.multiply(vectors_b, norms_a)
    term_2 = numpy.linalg.norm(term_2, ord=2, axis=0)
    angles = 2*numpy.arctan2(term_1, term_2)

    return angles


def normalized_non_maximum_entropy_detector(probabilities):
    indices = numpy.argmax(probabilities, axis=1)
    normalized_probabilities = numpy.copy(probabilities)
    normalized_probabilities[numpy.arange(probabilities.shape[0]), indices] = 0
    normalized_probabilities = normalized_probabilities/numpy.sum(normalized_probabilities, axis=1).reshape(-1, 1)
    from scipy.special import xlogy

    confidences = - numpy.sum(xlogy(normalized_probabilities, normalized_probabilities), axis=1)
    #confidences /= math.log(probabilities.shape[1])
    return confidences


def max_detector(probabilities):
    return numpy.max(probabilities, axis=1)
