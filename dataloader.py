import os
from typing import Optional

import h5py
from pytorch_lightning import LightningDataModule
from selene_sdk.samplers import RandomPositionsSampler
from selene_sdk.samplers.dataloader import _H5Dataset
from selene_sdk.samplers.dataloader import H5DataLoader
from selene_sdk.samplers.dataloader import _SamplerDataset
from selene_sdk.samplers.dataloader import SamplerDataLoader
from selene_sdk.sequences import Genome
from selene_sdk.utils import load_features_list



class GenomicDataH5(LightningDataModule):
    """Define a DataModule as a wrapper on top of Selene dataloader
    for pretraining (with conservation)
    """

    def __init__(
        self,
        model_name_or_path: str,
        seq_length: int = 4096,
        transform=None,
        train_batch_size: int =  32,
        eval_batch_size: int = 32,
        data_dir="./ds_train_hg38/pretrain_dataset",
        use_additional=None,  # pass in the key 'conservation'
        n_cpus=1,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.seq_length = seq_length
        self.transform = transform
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.data_dir = data_dir
        self.use_additional = use_additional
        self.n_cpus = n_cpus
        self.seed = 313


    def prepare_data(self):
        self._train_h5 = os.path.join(
            self.data_dir, 'pretrain.trainset.1m.seqlen={0}.h5'.format(self.seq_length))
        #self._train_h5 = os.path.join(
        #    self.data_dir, 'pretrain.trainset.4m.seqlen={0}.h5'.format(self.seq_length))

        self._valid_h5 = os.path.join(
            self.data_dir, 'pretrain.validset.32k.seqlen={0}.h5'.format(self.seq_length))
        print(self._train_h5, self._valid_h5)
        print('transform', self.transform)


    def setup(self, stage: Optional[str] = None):
        self.train = _H5Dataset(self._train_h5,
                                transform=self.transform,
                                use_additional=self.use_additional)
        self.validate = _H5Dataset(self._valid_h5,
                                   transform=self.transform,
                                   use_additional=self.use_additional)
        # NO TEST DATASET RIGHT NOW
        self.test = _H5Dataset(self._valid_h5,
                               transform=self.transform,
                               use_additional=self.use_additional)

    def train_dataloader(self):
        return H5DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            num_workers=self.n_cpus,
            seed=self.seed,
            shuffle=True)

    def val_dataloader(self):
        return H5DataLoader(
            self.validate,
            batch_size=self.eval_batch_size,
            num_workers=self.n_cpus,
            seed=self.seed,
            shuffle=True)

    def test_dataloader(self):
        return H5DataLoader(
            self.test,
            batch_size=self.eval_batch_size,
            num_workers=self.n_cpus,
            seed=self.seed,
            shuffle=True)


class GenomicData(LightningDataModule):
    """Define a DataModule as a wrapper on top of Selene dataloader"""

    def __init__(
        self,
        model_name_or_path: str,  # this param is not used!! remove later... or use it :)
        seq_length: int = 1024,
        transform=None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        data_dir="./ds_train_hg38",
        targets_file='sorted_deepsea.hg38.bed.gz',
        txt_file='distinct_features.txt',
        n_cpus=1,
        return_coords=False,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.seq_length = seq_length
        self.transform = transform
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.data_dir = data_dir
        self.targets_file = targets_file
        self.txt_file = txt_file
        self.n_cpus = n_cpus
        self.seed = 313
        self.return_coords = return_coords


    def prepare_data(self):
        self._genome = Genome(
            os.path.join(self.data_dir, 'data/hg38.fa'),
            blacklist_regions='hg38')
        self._targets_file = os.path.join(self.data_dir, self.targets_file)
        self._targets = load_features_list(
            os.path.join(self.data_dir, self.txt_file))

    def setup(self, stage: Optional[str] = None):
        self.train_sampler = RandomPositionsSampler(
            self._genome,
            self._targets_file,
            self._targets,
            sequence_length=self.seq_length,
            center_bin_to_predict=[self.seq_length // 2,
                                   self.seq_length // 2 + 1],
            feature_thresholds=None,
            mode='train',
            cache_modes=['train'],
            seed=self.seed
        )
        self.validate_sampler = RandomPositionsSampler(
            self._genome,
            self._targets_file,
            self._targets,
            sequence_length=self.seq_length,
            center_bin_to_predict=[self.seq_length // 2,
                                   self.seq_length // 2 + 1],
            feature_thresholds=None,
            mode='validate',
            cache_modes=['validate'],
            seed=self.seed
        )
        self.test_sampler = RandomPositionsSampler(
            self._genome,
            self._targets_file,
            self._targets,
            sequence_length=self.seq_length,
            center_bin_to_predict=[self.seq_length // 2,
                                   self.seq_length // 2 + 1],
            feature_thresholds=None,
            mode='test',
            cache_modes=['test'],
            seed=self.seed
        )
        self.train = _SamplerDataset(
            self.train_sampler, transform=self.transform, mode="train", return_coords=self.return_coords)
        self.validate = _SamplerDataset(
            self.validate_sampler, transform=self.transform, mode="validate", return_coords=self.return_coords)
        self.test = _SamplerDataset(
            self.test_sampler, transform=self.transform, mode="test", return_coords=self.return_coords)


    def train_dataloader(self):
        return SamplerDataLoader(
            self.train,
            mode="train",
            batch_size=self.train_batch_size,
            num_workers=self.n_cpus,
            seed=self.seed)

    def val_dataloader(self):
        return SamplerDataLoader(
            self.validate,
            mode="validate",
            batch_size=self.eval_batch_size,
            num_workers=self.n_cpus,
            seed=self.seed)

    def test_dataloader(self):
        return SamplerDataLoader(
            self.test,
            mode="test",
            batch_size=self.eval_batch_size,
            num_workers=self.n_cpus,
            seed=self.seed)



class ExpressionData(LightningDataModule):
    """Define a DataModule as a wrapper on top of Selene dataloader
    for gene expression TSS task
    """

    def __init__(
        self,
        model_name_or_path: str,
        h5_filepath: str,
        seq_length: int = 4096,
        transform=None,
        train_batch_size: int =  32,
        eval_batch_size: int = 32,
        data_dir="./expression_task",
        use_additional=None,  # now pass in the key 'additional' if you use it
        n_cpus=1,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.h5_filepath = h5_filepath
        self.seq_length = seq_length
        self.transform = transform
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.data_dir = data_dir
        self.use_additional = use_additional
        self.n_cpus = n_cpus
        self.seed = 313


    def prepare_data(self):
        self._train_h5 = os.path.join(self.data_dir, self.h5_filepath)

        h5_fileprefix = None
        if 'shift' in self._train_h5:
            h5_fileprefix = self._train_h5.split('.shift')[0]
        else:
            h5_fileprefix = self._train_h5.split('.train')[0]

        self._valid_h5 = h5_fileprefix + '.valid.h5'
        self._test_h5 = h5_fileprefix + '.test.h5'

        print("Train H5", self._train_h5)
        print("Valid H5", self._valid_h5)


    def setup(self, stage: Optional[str] = None):
        self.train = _H5Dataset(self._train_h5,
                                transform=self.transform,
                                use_additional=self.use_additional,
                                sequence_len=self.seq_length)
        self.validate = _H5Dataset(self._valid_h5,
                                   transform=self.transform,
                                   use_additional=self.use_additional,
                                   sequence_len=self.seq_length)
        self.test = _H5Dataset(self._test_h5,
                               transform=self.transform,
                               use_additional=self.use_additional,
                               sequence_len=self.seq_length)


    def train_dataloader(self):
        return H5DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            num_workers=self.n_cpus,
            seed=self.seed,
            shuffle=True)

    def val_dataloader(self):
        return H5DataLoader(
            self.validate,
            batch_size=self.eval_batch_size,
            num_workers=self.n_cpus,
            seed=self.seed,
            shuffle=False)

    def test_dataloader(self):
        return H5DataLoader(
            self.test,
            batch_size=self.eval_batch_size,
            num_workers=self.n_cpus,
            seed=self.seed,
            shuffle=False)




class GenomicBenchmarks(LightningDataModule):
    """Define a DataModule as a wrapper on top of Selene dataloader
    for Genomic Benchmarks datasets
    """

    def __init__(
        self,
        model_name_or_path: str,
        dataset: str,
        transform=None,
        train_batch_size: int =  32,
        eval_batch_size: int = 32,
        data_dir="./genomic_benchmarks",
        n_cpus=1,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.dataset = dataset
        self.transform = transform
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.data_dir = data_dir
        self.n_cpus = n_cpus
        self.seed = 313


    def prepare_data(self):
        self._train_file = os.path.join(self.data_dir, self.dataset, 'train_dataset.h5')
        self._valid_file = os.path.join(self.data_dir, self.dataset, 'valid_dataset.h5')
        self._test_file = os.path.join(self.data_dir, self.dataset, 'test_dataset.h5')
        print(self._train_file)
        print(self._valid_file)
        print(self._test_file)


    def setup(self, stage: Optional[str] = None):
        self.train = _H5Dataset(self._train_file,
                                transform=self.transform)
        self.validate = _H5Dataset(self._valid_file,
                                   transform=self.transform)
        self.test = _H5Dataset(self._test_file,
                               transform=self.transform)


    def train_dataloader(self):
        return H5DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            num_workers=self.n_cpus,
            seed=self.seed,
            shuffle=True)

    def val_dataloader(self):
        return H5DataLoader(
            self.validate,
            batch_size=self.eval_batch_size,
            num_workers=self.n_cpus,
            seed=self.seed,
            shuffle=False)

    def test_dataloader(self):
        return H5DataLoader(
            self.test,
            batch_size=self.eval_batch_size,
            num_workers=self.n_cpus,
            seed=self.seed,
            shuffle=False)
