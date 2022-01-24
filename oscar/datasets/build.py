# Code is modified from https://github.com/microsoft/Oscar
# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

import os.path as op
import torch

from oscar.utils.misc import get_world_size
from .caption_tsv import CaptionTSVDataset, CaptionTSVDatasetWithConstraints


def build_dataset(yaml_file, tokenizer, args, is_train=True):
    if not op.isfile(yaml_file):
        yaml_file = op.join(args.data_dir, yaml_file)
        assert op.isfile(yaml_file)

    if is_train:
        return CaptionTSVDataset(yaml_file, tokenizer=tokenizer,
            add_od_labels=args.add_od_labels, max_img_seq_length=args.max_img_seq_length,
            max_seq_length=args.max_seq_length, max_seq_a_length=args.max_seq_a_length,
            is_train=True, mask_prob=args.mask_prob, max_masked_tokens=args.max_masked_tokens)
    if args.use_cbs:
        dataset_class = CaptionTSVDatasetWithConstraints
    else:
        dataset_class = CaptionTSVDataset
    return dataset_class(yaml_file, tokenizer=tokenizer,
            add_od_labels=args.add_od_labels, max_img_seq_length=args.max_img_seq_length,
            max_seq_length=args.max_seq_length, max_seq_a_length=args.max_gen_length,
            is_train=False)


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_data_loader(logger, args, yaml_file, tokenizer, is_distributed=True, 
        is_train=True):
    dataset = build_dataset(yaml_file, tokenizer, args, 
        is_train=(is_train and not args.scst))
    if is_train:
        shuffle = True
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        logger.info("Train with {} images per GPU.".format(images_per_gpu))
        logger.info("Total batch size {}".format(images_per_batch))
        logger.info("Total training steps {}".format(num_iters))
    else:
        shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, sampler=sampler,
        batch_size=images_per_gpu,
        pin_memory=True,
    )
    return data_loader
