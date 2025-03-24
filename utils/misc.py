import copy
import logging
import pickle
import random
from collections import defaultdict
from typing import Tuple, List

import numpy as np
import torch
from deepspeed.profiling.flops_profiler import get_model_profile
from termcolor import colored


class ObjDict(dict):
    """
    Makes a dictionary behave like an object, with attribute-style access.
    """

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __deepcopy__(self, name):
        copy_dict = dict()
        for key, value in self.items():
            if hasattr(value, '__deepcopy__'):
                copy_dict[key] = copy.deepcopy(value)
            else:
                copy_dict[key] = value
        return ObjDict(copy_dict)

    def __getstate__(self):
        return pickle.dumps(self.__dict__)

    def __setstate__(self, state):
        self.__dict__ = pickle.loads(state)

    def __exists__(self, name):
        return name in self.__dict__


def set_random_seed(seed=0, benchmark=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = True


def resume_state(model, ckpt, **args):
    state = torch.load(ckpt)
    model.load_state_dict(state['model'], strict=True)
    logging.info(f'loaded model state from {ckpt}')

    for i in args.keys():
        if hasattr(args[i], "load_state_dict"):
            args[i].load_state_dict(state[i])
            logging.info(f'loaded {i} state from {ckpt}')
    return state


def load_state(model, ckpt, **args):
    state = torch.load(ckpt)
    if hasattr(model, 'module'):
        incompatible = model.module.load_state_dict(state['model'], strict=False)
    else:
        incompatible = model.load_state_dict(state['model'], strict=False)
    logging.info(f'loaded model state from {ckpt}')

    if incompatible.missing_keys:
        logging.warning('missing_keys')
        logging.warning(
            get_missing_parameters_message(incompatible.missing_keys),
        )
    if incompatible.unexpected_keys:
        logging.warning('unexpected_keys')
        logging.warning(
            get_unexpected_parameters_message(incompatible.unexpected_keys)
        )
    for i in args.keys():
        if i not in state.keys():
            logging.warning(f'missing {i} state in state_dict, just skipped')
            continue
        if hasattr(args[i], "load_state_dict"):
            args[i].load_state_dict(state[i])
            logging.info(f'loaded {i} state from {ckpt}')
    return state


def save_state(ckpt, **args):
    state = {}
    for i in args.keys():
        item = args[i].state_dict() if hasattr(args[i], "state_dict") else args[i]
        state[i] = item
    torch.save(state, ckpt)


def cal_model_params(model):
    total = sum([param.nelement() for param in model.parameters()])
    trainable = sum([param.nelement() for param in model.parameters() if param.requires_grad])

    return total, trainable


def cal_model_flops(model, inputs: [List | Tuple] = None, profile=True, warmup=10):
    flops, macs, params = get_model_profile(
        model=model,
        args=inputs,
        print_profile=profile,  # prints the model graph with the measured profile attached to each module
        detailed=True,  # print the detailed profile
        warm_up=warmup,  # the number of warm-ups before measuring the time of each module
        as_string=False,  # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
        output_file=None,  # path to the output file. If None, the profiler prints to stdout.
        ignore_modules=None)  # the list of modules to ignore in the profiling
    return flops, macs, params


def get_missing_parameters_message(keys):
    groups = _group_checkpoint_keys(keys)
    msg = "Some model parameters or buffers are not found in the checkpoint:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "blue") for k, v in groups.items()
    )
    return msg


def get_unexpected_parameters_message(keys):
    groups = _group_checkpoint_keys(keys)
    msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "magenta") for k, v in groups.items()
    )
    return msg


def _group_checkpoint_keys(keys):
    groups = defaultdict(list)
    for key in keys:
        pos = key.rfind(".")
        if pos >= 0:
            head, tail = key[:pos], [key[pos + 1:]]
        else:
            head, tail = key, []
        groups[head].extend(tail)
    return groups


def _group_to_str(group):
    if len(group) == 0:
        return ""

    if len(group) == 1:
        return "." + group[0]

    return ".{" + ", ".join(group) + "}"
