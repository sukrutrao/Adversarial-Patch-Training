import torch
import numpy
from .attack import *
from common.log import log
import common.torch
from common.mask import MaskGenerator, MaskDirection


class AdversarialPatch(Attack):
    """
    Adversarial Patch attack
    """

    def __init__(self, mask_gen, epsilon, max_iterations, optimize_location=False, optimize_location_type=None, stride=None, signed_grad=False):
        """
        Constructor

        :param mask_gen: MaskGenerator object to generate masks
        :type mask_gen: MaskGenerator
        :param epsilon: learning rate
        :type epsilon: float
        :param max_iterations: total number of iterations to learn patch
        :type max_iterations: int
        :param optimize_location: flag to decide whether to optimize location of mask, defaults to False
        :type optimize_location: bool
        :param optimize_location_type: mode of optimizing location of mask if applicable, defaults to None
        :type optimize_location_type: str
        :param stride: number of pixels to move mask in each step when optimizing location if applicable, defaults to None
        :type stride: int
        :param signed_grad: flag to decide whether to use sign of gradient to update patch, defaults to False
        :type signed_grad: bool
        """

        super(AdversarialPatch, self).__init__()

        self.mask_gen = mask_gen
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.optimize_location = optimize_location
        self.optimize_location_type = optimize_location_type
        self.stride = stride
        self.signed_grad = signed_grad
        self.norm = None

    def run(self, model, images, objective, writer=common.summary.SummaryWriter(), prefix=''):
        """
        Run Adversarial Patch attack

        :param model: model to attack, must contain normalization layer to work correctly
        :type model: torch.nn.Module
        :param images: images
        :type images: torch.autograd.Variable
        :param objective: objective
        :type objective: UntargetedObjective
        :param writer: summary writer, defaults to common.summary.SummaryWriter()
        :type writer: common.summary.SummaryWriter, optional
        :param prefix: prefix for writer, defaults to ''
        :type prefix: str, optional
        """

        super(AdversarialPatch, self).run(
            model, images, objective, writer, prefix)

        assert model.training is False
        assert self.mask_gen is not None
        assert self.epsilon is not None
        assert self.max_iterations is not None and self.max_iterations >= 0
        assert self.optimize_location is False or self.stride is not None
        assert self.optimize_location is False or self.optimize_location_type is not None

        assert len(images.shape) == 4
        batch_size, channels, _, _ = images.shape

        assert (not torch.any(images < 0.0)) and (not torch.any(images > 1.0))

        is_cuda = common.torch.is_cuda(model)

        # Generate masks randomly from the allowed locations
        mask_coords = self.mask_gen.random_location(batch_size)
        masks = common.torch.as_variable(
            self.mask_gen.get_masks(mask_coords, channels), cuda=is_cuda)

        # Stores perturbations. Patched image = (1 - masks) * clean image + masks * patches
        patches = common.torch.as_variable(numpy.random.uniform(
            low=0.0, high=1.0, size=images.shape).astype(numpy.float32), cuda=is_cuda, requires_grad=True)

        current_iteration = 0

        # Best error found for each image
        success_errors = numpy.ones((batch_size), dtype=numpy.float32) * 1e12

        # Best patch found for each image
        success_perturbations = numpy.zeros(images.shape, dtype=numpy.float32)

        # Train the current batch
        while current_iteration < self.max_iterations:
            current_iteration += 1

            # Set all gradients in the model to zero at each iteration so that they do not add up
            model.zero_grad()

            # Apply patch
            inverse_masks = common.torch.as_variable(numpy.ones(
                masks.shape, dtype=numpy.float32), cuda=is_cuda) - masks
            imgs_patched = inverse_masks * images + masks * patches
            assert (not torch.any(imgs_patched < 0.0)) and (
                not torch.any(imgs_patched > 1.0))

            # Get predictions on patched images
            preds = model(imgs_patched)

            # Calculate error per image. `objective` is aware of the true labels.
            error = objective(preds)

            # Compute loss and gradients
            loss = torch.sum(error)
            loss.backward()

            # For every image, if the best known loss so far is found, update best loss and its
            # corresponding patch. Currently, not checking if flip occurs.
            for b in range(batch_size):
                if error[b].item() < success_errors[b]:
                    success_errors[b] = error[b].item()
                    success_perturbations[b] = (
                        masks[b] * (patches[b] - images[b])).detach().cpu().numpy()

            # Get gradient with respect to patches
            if self.signed_grad:
                loss_grad = torch.sign(patches.grad)
            else:
                loss_grad = patches.grad

            # Patch update direction aims to reduce activation of source class
            # patches.data is used on the left so that a new node is not created in the graph
            patches.data = patches - self.epsilon * masks * loss_grad

            # Clip patches, since it must always be in [0,1)
            patches.data.clamp_(0.0, 1.0)

            # Set gradients for patches to zero
            patches.grad.data.zero_()

            # Optimize location of patch, if applicable
            if self.optimize_location:

                # Best error values for each image
                best_errors = error.clone()

                # Best direction to move for each image
                best_directions = [None] * batch_size

                # In full mode, the patches are moved by `stride` pixels in each direction, and for
                # each image, the direction which resulted in the best error value is kept for the
                # next iteration
                if self.optimize_location_type == 'full':
                    with torch.no_grad():

                        # Find best direction by trying each. Only four extra forward passes are
                        # necessary.
                        for direction in MaskDirection:
                            moved_mask_coords = self.mask_gen.move_coords(
                                mask_coords, direction, self.stride)
                            moved_masks = common.torch.as_variable(
                                self.mask_gen.get_masks(moved_mask_coords, channels), cuda=is_cuda)
                            moved_patches = patches.clone()
                            moved_patches[torch.where(
                                moved_masks)] = patches[torch.where(masks)]
                            inverse_moved_masks = common.torch.as_variable(numpy.ones(
                                moved_masks.shape, dtype=numpy.float32), cuda=is_cuda) - moved_masks
                            moved_imgs_patched = inverse_moved_masks * images + moved_masks * moved_patches
                            assert (not torch.any(moved_imgs_patched < 0.0)) and (
                                not torch.any(moved_imgs_patched > 1.0))
                            moved_preds = model(moved_imgs_patched)
                            moved_errors = objective(moved_preds)
                            for b in range(batch_size):
                                if moved_errors[b] < best_errors[b]:
                                    best_errors[b] = moved_errors[b]
                                    best_directions[b] = direction

                # In random mode, the patch for each image is moved in a random direction by
                # `stride` pixels
                elif self.optimize_location_type == 'random':
                    with torch.no_grad():
                        moved_mask_coords, directions = self.mask_gen.move_coords_random(
                            mask_coords, self.stride)
                        moved_masks = common.torch.as_variable(
                            self.mask_gen.get_masks(moved_mask_coords, channels), cuda=is_cuda)
                        moved_patches = patches.clone()
                        moved_patches[torch.where(
                            moved_masks)] = patches[torch.where(masks)]
                        inverse_moved_masks = common.torch.as_variable(numpy.ones(
                            moved_masks.shape, dtype=numpy.float32), cuda=is_cuda) - moved_masks
                        moved_imgs_patched = inverse_moved_masks * images + moved_masks * moved_patches
                        assert (not torch.any(moved_imgs_patched < 0.0)) and (
                            not torch.any(moved_imgs_patched > 1.0))
                        moved_preds = model(moved_imgs_patched)
                        moved_errors = objective(moved_preds)
                        for b in range(batch_size):
                            if moved_errors[b] < best_errors[b]:
                                best_errors[b] = moved_errors[b]
                                best_directions[b] = directions[b]

                else:
                    raise ValueError

                # Move mask coordinates for each image if a direction is found. If all
                # directions lead to worse error, no move is made.
                for b in range(batch_size):
                    if best_directions[b] is not None:
                        mask_coords[b] = self.mask_gen.move_coords_single(
                            mask_coords[b], best_directions[b], self.stride)

                old_masks = masks

                # Update masks
                masks = common.torch.as_variable(
                    self.mask_gen.get_masks(mask_coords, channels), cuda=is_cuda)

                patches[torch.where(
                    masks)].data = patches[torch.where(old_masks)]

                # Set gradients for patches to zero. TODO: this should not be required
                patches.grad.data.zero_()

        return success_perturbations, success_errors
