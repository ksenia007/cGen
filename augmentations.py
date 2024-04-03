import numpy as np
from numpy.random import randint
from numpy.random import rand
from torchvision import transforms


class FlipSequence(object):
    """Flip the sequence left to right w/ some probability"""
    def __init__(self, prob, **kwargs):
        assert prob<=1
        self.prob = prob

    def __call__(self, sample, additional=None):
        # we shift a random subset of buffer to make a sequence length
        if rand()<self.prob:
            # flip left to right
            if len(sample.shape) == 3:
                sample = np.flip(sample, (1,))
            elif len(sample.shape) == 4:
                sample = np.flip(sample, (2,))
            else:
                sample = np.flip(sample, (0,))
            
            # compelement
            ax = np.where(np.array(sample.shape) == 4)[0]
            if isinstance(ax, np.ndarray):
                ax = ax[0]
            sample = np.flip(sample, (ax,))
        return sample.copy()


class ShiftSequence(object):
    """Crop the buffer and the sequence length"""

    def __init__(self, **kwargs):
        assert kwargs['buffer'] >= 0, "Bad buffer value, accept [0,inf) range: {}".format(kwargs['buffer'])
        if kwargs['buffer'] == 0: print('Careful, buffer is set to 0')
        self.buffer = int(kwargs['buffer']) 
        self.seq_length = int(kwargs['seq_length'])

    def __call__(self, sample, additional=None):
        # we shift a random subset of buffer to make a sequence length

        if len(sample)==2:
            use_additional = True
        else:
            use_additional = False
        if use_additional:
            sample, additional = sample

        if self.buffer == 0:
            shift_bp = 0
        else:
            shift_bp = randint(0, self.buffer) # buffer is the maximum amount we could shift on each side, and it is symmetric
        # (example) 300bp shift would starts from 300bp in the beginning, and attach 300bp on the other end
        half_seq_length = self.seq_length // 2
        
        sample = np.array(sample)

        # TODO: reformat into function, code chunk repeats
        if len(sample.shape) == 3:

            middle = sample.shape[1]//2
            left = middle - half_seq_length + shift_bp
            right = middle + half_seq_length + shift_bp
            assert left<right, "Bad buffer / sequence length combo: actual length={}, want seq length {} (1/2 is {}), middle={}, left={}, right={}".format(
                sample.shape[1], self.seq_length, half_seq_length, middle, left, right
            )
            ###

            sample = sample[:, left:right, :]
            if use_additional:
                additional = additional[:, left:right]

        elif len(sample.shape) == 4:

            middle = sample.shape[2]//2
            left = middle - half_seq_length + shift_bp
            right = middle + half_seq_length + shift_bp
            assert left<right, "Bad buffer / sequence length combo: actual length={}, want seq length {} (1/2 is {}), middle={}, left={}, right={}".format(
                sample.shape[1], self.seq_length, half_seq_length, middle, left, right
            )

            sample = sample[:, :, left:right, :]
            if use_additional:
                additional = additional[:, :, left:right]


        else:

            middle = sample.shape[0]//2
            left = middle - half_seq_length + shift_bp
            right = middle + half_seq_length + shift_bp
            assert left<right, "Bad buffer / sequence length combo: actual length={}, want seq length {} (1/2 is {}), middle={}, left={}, right={}".format(
                sample.shape[0], self.seq_length, half_seq_length, middle, left, right
            )
            ###

            sample = sample[left:right, :]
            if use_additional:
                additional = additional[left:right]

        return [sample, additional]


class MaskPart(object):
    """Mask part of the sequence, proportion_mask is the proportion to set to all 0s (BP resolution)
    """

    def __init__(self, proportion_mask, no_center=False, **kwargs):
        assert proportion_mask < 1
        self.proportion_mask = proportion_mask
        self.seq_length = kwargs['seq_length']
        self.no_center = no_center

    def __call__(self, sample):
        
        if len(sample)==2:
            if sample[1] is None:
                use_additional = False
            else:
                use_additional = True
            
            sample, additional = sample
        else:
            use_additional = False
            
        n_bases_mask = int(self.seq_length*self.proportion_mask)


        if not use_additional: 
            choice = np.arange(self.seq_length)
        else:
            choice = np.where(additional<0)[0]

            if len(choice)<n_bases_mask:
                #print('less options than we need to mask, have {}, need {}; updating needed'.format(len(choice), n_bases_mask))
                n_bases_mask = int(0.5*len(choice))

        mask_ids = np.random.choice(choice, size=n_bases_mask, replace=False)
        
        sample = np.array(sample)
        if len(sample.shape) == 3:
            sample[:, mask_ids, :] = [0, 0, 0, 0]
        elif len(sample.shape) == 4:
            sample[:, :, mask_ids, :] = [0,0,0,0]
        else:
            sample[mask_ids, :] = [0, 0, 0, 0]
        
        return sample


class ContrastiveTransformations:
    """Define how contrastive transforms are applied"""

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        # self.tensor = transforms.ToTensor() # need to add here (?)
        self.n_views = n_views

    def __call__(self, x, additional=None):
        if additional is None:
            # not using the additional scores
            return [self.base_transforms(x) for i in range(self.n_views)]
        else:
            # pass in additional as a [a, add] vector - one entry
            return [self.base_transforms([x, additional]) for i in range(self.n_views)]
