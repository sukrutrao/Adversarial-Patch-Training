import torch
import numpy
import common.torch
import common.numpy
import common.summary
from common.log import log


def progress(batch, batches, epoch=None):
    """
    Report progress.

    :param epoch: epoch
    :type epoch: int
    :param batch: batch
    :type batch: int
    :param batches: batches
    :type batches: int
    """

    if batch == 0:
        if epoch is not None:
            log(' %d .' % epoch, end='')
        else:
            log(' .', end='')
    else:
        log('.', end='', context=False)

    if batch == batches - 1:
        log(' done', end="\n", context=False)


def test(model, testset, cuda=False, transform=None):
    """
    Test a model on a clean or adversarial dataset.

    :param model: model
    :type model: torch.nn.Module
    :param testset: test set
    :type testset: torch.utils.data.DataLoader
    :param cuda: use CUDA
    :type cuda: bool
    """

    assert model.training is False
    assert len(testset) > 0
    assert isinstance(testset, torch.utils.data.DataLoader)
    assert isinstance(testset.sampler, torch.utils.data.SequentialSampler)
    assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

    probabilities = None
    # should work with and without labels
    for b, data in enumerate(testset):
        if isinstance(data, tuple) or isinstance(data, list):
            inputs = data[0]
        else:
            inputs = data

        assert isinstance(inputs, torch.Tensor)

        inputs = common.torch.as_variable(inputs, cuda)
        inputs = inputs.permute(0, 3, 1, 2)

        if transform is not None:
            inputs = common.torch.as_variable(transform(inputs.detach().cpu().numpy()), cuda=True)
        logits = model(inputs)
        probabilities_ = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
        probabilities = common.numpy.concatenate(probabilities, probabilities_)

        progress(b, len(testset))

    assert probabilities.shape[0] == len(testset.dataset)

    return probabilities


def attack(model, testset, attack, objective, attempts=1, writer=common.summary.SummaryWriter(), cuda=False, transform=None):
    """
    Attack model.

    :param model: model
    :type model: torch.nn.Module
    :param testset: test set
    :type testset: torch.utils.data.DataLoader
    :param attack: attack
    :type attack: attacks.Attack
    :param objective: attack objective
    :type objective: attacks.Objective
    :param attempts: number of attempts
    :type attempts: int
    :param get_writer: summary writer or utility function to get writer
    :type get_writer: torch.utils.tensorboard.SummaryWriter or callable
    :param cuda: whether to use CUDA
    :type cuda: bool
    """

    assert model.training is False
    assert len(testset) > 0
    assert attempts >= 0
    assert isinstance(testset, torch.utils.data.DataLoader)
    assert isinstance(testset.sampler, torch.utils.data.SequentialSampler)
    assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

    perturbations = []
    probabilities = []
    errors = []

    # should work via subsets of datasets
    for a in range(attempts):
        perturbations_a = None
        probabilities_a = None
        errors_a = None

        for b, data in enumerate(testset):
            assert isinstance(data, tuple) or isinstance(data, list)

            inputs = common.torch.as_variable(data[0], cuda)
            # inputs = N x H x W x C
            inputs = inputs.permute(0, 3, 1, 2)
            # inputs = N x C x H x W
            labels = common.torch.as_variable(data[1], cuda)

            # attack target labels
            targets = None
            if len(list(data)) > 2:
                targets = common.torch.as_variable(data[2], cuda)

            objective.set(labels, targets)
            perturbations_b, errors_b = attack.run(model, inputs, objective,
                                                   writer=writer if not callable(
                                                       writer) else writer('%d-%d' % (a, b)),
                                                   prefix='%d/%d/' % (a, b) if not callable(writer) else '')

            inputs = inputs + common.torch.as_variable(perturbations_b, cuda)
            assert (not torch.any(inputs<0.0)) and (not torch.any(inputs>1.0)) ###########################################################

            if transform is not None:
                inputs = common.torch.as_variable(transform(inputs.detach().cpu().numpy()), cuda=True)
            logits = model(inputs)
            probabilities_b = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()

            perturbations_a = common.numpy.concatenate(perturbations_a, perturbations_b)
            probabilities_a = common.numpy.concatenate(probabilities_a, probabilities_b)
            errors_a = common.numpy.concatenate(errors_a, errors_b)

            progress(b, len(testset), epoch=a)

        perturbations.append(perturbations_a)
        probabilities.append(probabilities_a)
        errors.append(errors_a)

    perturbations = numpy.array(perturbations)
    probabilities = numpy.array(probabilities)
    errors = numpy.array(errors)

    assert perturbations.shape[1] == len(testset.dataset)
    assert probabilities.shape[1] == len(testset.dataset)
    assert errors.shape[1] == len(testset.dataset)

    # #attemps x N x H x W x C, #attempts x N x #classes, #attempts x N
    return perturbations, probabilities, errors


def test_attack_directions(model, testset, adversarialset, points=51, ord=float('inf'), cuda=False):
    """
    Test model along attack directions.

    :param model: model
    :type model: torch.nn.Module
    :param testset: test set
    :type testset: torch.utils.data.DataLoader
    :param adversarialset: adversarial set
    :type adversarialset: torch.utils.data.DataLoader
    :param cuda: whether to use CUDA
    :type cuda: bool
    """

    assert model.training is False
    assert len(testset) > 0
    assert isinstance(testset, torch.utils.data.DataLoader)
    assert isinstance(testset.sampler, torch.utils.data.SequentialSampler)
    assert len(adversarialset) > 0
    assert len(testset) >= len(adversarialset)
    assert isinstance(adversarialset, torch.utils.data.DataLoader)
    assert isinstance(adversarialset.sampler, torch.utils.data.SequentialSampler)
    assert points % 2 == 1
    assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

    probabilities = []
    norms = []

    for testdata, adversarialdata in zip(enumerate(testset), enumerate(adversarialset)):
        testb = testdata[0]
        adversarialb = adversarialdata[0]
        assert testb == adversarialb
        assert isinstance(testdata[1], list)
        assert isinstance(adversarialdata[1], list)

        testdata = testdata[1]
        adversarialdata = adversarialdata[1]
        inputs = testdata[0]
        adversarial_inputs = adversarialdata[0]

        batch_size = inputs.shape[0]
        adversarial_directions = adversarial_inputs - inputs

        adversarial_directions = adversarial_directions.numpy()
        adversarial_norms = numpy.linalg.norm(
            adversarial_directions.reshape(batch_size, -1), axis=1, ord=ord)

        for b in range(batch_size):
            factors = numpy.linspace(-2, 2, points).astype(numpy.float32)
            adversarial_input_sequence = numpy.repeat(numpy.expand_dims(
                adversarial_directions[b], axis=0), points, axis=0)
            adversarial_input_sequence = adversarial_input_sequence * \
                common.numpy.expand_as(factors, adversarial_input_sequence)

            adversarial_input_sequence = common.torch.as_variable(
                inputs[b], cuda) + common.torch.as_variable(adversarial_input_sequence, cuda)

            assert numpy.isclose(factors[0], -2)
            assert numpy.isclose(factors[points // 2], 0)
            assert numpy.isclose(factors[-1], 2)

            numpy.testing.assert_almost_equal(
                adversarial_input_sequence[points // 2].cpu().numpy(), inputs[b].cpu().numpy(), 4)
            numpy.testing.assert_almost_equal(adversarial_input_sequence[0].cpu(
            ).numpy(), inputs[b].cpu().numpy() - 2*adversarial_directions[b], 4)
            numpy.testing.assert_almost_equal(
                adversarial_input_sequence[-1].cpu().numpy(), inputs[b].cpu().numpy() + 2*adversarial_directions[b], 4)

            adversarial_input_sequence = adversarial_input_sequence.permute(0, 3, 1, 2)
            adversarial_input_sequence = torch.clamp(adversarial_input_sequence, min=0, max=1)

            logit_sequence = model.forward(adversarial_input_sequence)
            probability_sequence = torch.nn.functional.softmax(logit_sequence, dim=1)

            probabilities.append(probability_sequence.detach().cpu().numpy())
            norms.append(adversarial_norms[b]*factors)

            progress(adversarialb*batch_size + b, len(adversarialset)*batch_size)

    return numpy.array(probabilities), numpy.array(norms)
