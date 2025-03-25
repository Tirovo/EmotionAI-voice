import torch
import random

class SpecAugmentTransform:
    def __init__(self, freq_mask_param=15, time_mask_param=25, num_masks=2):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks

    def __call__(self, spec):
        # spec shape: (1, 128, 128)
        cloned = spec.clone()

        _, freq_dim, time_dim = cloned.shape

        # Frequency masking
        for _ in range(self.num_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, freq_dim - f)
            cloned[0, f0:f0 + f, :] = 0

        # Time masking
        for _ in range(self.num_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, time_dim - t)
            cloned[0, :, t0:t0 + t] = 0

        return cloned
