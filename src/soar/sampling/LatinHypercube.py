"""
Samples points based on the Latin Hypercube procedure outlined in §5.2 of [1].
We assume that the distribution is the cartesian product of uniform distributions
on R.

Does this really need to be a class? I don't think so.

References:
    [1] T. J. Santner, B. J. Williams, and W. I. Notz, The Design and Analysis of
    Computer Experiments. in Springer Series in Statistics. New York, NY: Springer
    New York, 2003. doi: 10.1007/978-1-4757-3799-8.

"""
import torch


from .ExperimentalDesign import ExperimentalDesign


class LatinHypercube(ExperimentalDesign):
    _num_dimensions: int
    _support: torch.Tensor

    _strict_latin_hypercube: bool

    def __init__(self,
                 support: torch.Tensor,
                 strict_latin_hypercube: bool = False):
        self._support = support
        self._num_dimensions = support.shape[0]
        self._strict_latin_hypercube = strict_latin_hypercube

    def generate_samples(self, num_samples: int) -> torch.Tensor:
        permutations = []
        # check to see if num_dimensions < (num_samples)!
        if self._strict_latin_hypercube:
            k = num_samples
            quotient = self._num_dimensions
            sample_points = False
            while k > 1:
                quotient /= k
                k -= 1
                if quotient < 1:
                    sample_points = True
                    break
            if not sample_points:
                raise Exception("num samples too small for dimension of problem")
        while len(permutations) < self._num_dimensions:
            perm = torch.randperm(num_samples)
            if self._strict_latin_hypercube:
                already_sampled = False
                for perm2 in permutations:
                    already_sampled |= (perm2 == perm).all()
                if not already_sampled:
                    permutations.append(perm)
            else:
                permutations.append(perm)
        permutations = torch.stack(permutations).T

        # generate the normalized latin hypercube
        sample_points = torch.zeros((num_samples, self._num_dimensions))
        for idx, perm in enumerate(permutations):
            for component, division in enumerate(perm):
                sample_points[idx, component] = (division + torch.rand(1)) / num_samples

        # scale the hypercube
        scaling_factor = self._support[:, 1] - self._support[:, 0]
        scaled_samples = sample_points * scaling_factor + self._support[:, 0]
        return scaled_samples

    @property
    def support(self):
        return self._support

    @property
    def num_dimensions(self):
        return self._num_dimensions


if __name__ == '__main__':
    dimensions = torch.Tensor([
        [-1.0, 1.0],
        [1.0, 5.0],
        [0.0, 1.0]
    ])

    lh = LatinHypercube(dimensions, strict_latin_hypercube=True)
    samples = lh.generate_samples(4)
    print(samples)
