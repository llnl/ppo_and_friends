import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Tuple, Box, Discrete, MultiDiscrete, MultiBinary, Dict, Sequence, Graph
from ppo_and_friends.utils.spaces import ShapelyDiscrete, SparseFlatteningDict, SparseFlatteningTuple, FlatteningTuple
from typing import Any
import numpy as np
from copy import deepcopy

class InfArraySpace(spaces.Space[Any]):

    def __init__(self, n, *args, **kw_args):
        """
        """
        self.n = n
        super().__init__(*args, **kw_args)

    def sample(self):
        return np.array([np.inf] * self.n)

def get_all_inf_dict_0():
    d_0 = {
        "tuple1" : Tuple([InfArraySpace(5), InfArraySpace(2)]),
    }

def get_dict_0():
    d_0 = {
        "tuple1" : Tuple([Box(-1, 1, (2,)), Discrete(3), InfArraySpace(2)]),
    }

    return Dict(d_0)

def get_dict_1(inner_dict):
    d_1 = {
        "box1" : Box(-1, 1, (2,)),
        "discrete1" : Discrete(2),
        "infarray2" : InfArraySpace(8),
        "tuple2" : Tuple([Dict(inner_dict), Discrete(4), InfArraySpace(4)]),
    }
    
    return Dict(d_1)

def get_dict_2(inner_dict):
    d_2 = {
        "box2" : Box(-1, 1, (4,)),
        "discrete1" : Discrete(5),
        "multidiscrete1" : MultiDiscrete([5, 3, 4]),
        "multibinary1": MultiBinary(3),
        "dict1": Dict(inner_dict),
        "infarray4" : InfArraySpace(5),
    }
    
    return Dict(d_2)

def get_dict_3():
    d_2 = get_dict_2()
    
    return Dict(d_2)


def test_dense_dict(rebase=False):
    gym_dict = get_dict_2(get_dict_1(get_dict_0()))
    sfd = SparseFlatteningDict(deepcopy(gym_dict.spaces), seed = gym_dict._np_random)

    #
    # Make sure our sparse tree looks right.
    #
    sfd.mode  = "sparse"
    tree_str  = sfd.get_tree_str()

    if rebase:
        with open("baselines/sparse_tree_baseline.txt", "w") as out_f:
            out_f.write(tree_str)

    with open("baselines/sparse_tree_baseline.txt", "r") as in_f:
        tree_baseline = in_f.read()

    err_msg = f"""
    {tree_str}

    VS

    {tree_baseline}
    """
    assert tree_str == tree_baseline, err_msg

    #
    # Make sure our dense tree looks right after switching from
    # sparse to dense.
    #
    sfd.auto_flatten = False
    sfd.mode         = "dense"

    if rebase:
        with open("baselines/dense_tree_baseline.txt", "w") as out_f:
            out_f.write(tree_str)

    with open("baselines/dense_tree_baseline.txt", "r") as in_f:
        tree_baseline = in_f.read()

    err_msg = f"""
    {tree_str}

    VS

    {tree_baseline}
    """
    assert tree_str == tree_baseline, err_msg

    #
    # Make sure a dense sampling matches.
    #
    gym_dict.seed(1)
    gym_sample = gym_dict.sample()

    sfd.seed(1)
    sample = sfd.sample()
    err_msg = f"""
    Expected sfd.sample() to match gym_dict.sample().
    sfd sample: {sample}
    gym sample: {gym_sample}
    """
    assert (str(sample) == str(gym_sample)), err_msg
    
    #
    # Make sure we can flatten a dense sample.
    #
    flattened_sample = sfd.sparse_flatten_sample(sample)

    err_msg = f"""
    Expected flattened_size attribute to match size of flattened sample.
    flatttened_size: {sfd._flattened_size}
    flatttened_size: {flattened_sample.size}
    """
    assert flattened_sample.size == sfd._flattened_size, err_msg

    err_msg = f"""
    Expected flattened_size attribute to match the shape attribute.
    flatttened_size: {sfd._flattened_size}
    shape: {sfd.shape}
    """
    assert flattened_sample.shape == (sfd._flattened_size,), err_msg

    err_msg = f"""
    Expected sparse flattened sample to not have any inf values, but found:
    {flattened_sample}
    """
    assert np.inf not in flattened_sample

    if rebase:
        np.save("baselines/flattened_dense", flattened_sample)

    baseline = np.load("baselines/flattened_dense.npy")

    err_msg = f"""
    Expected: {baseline}
    Found: {flattened_sample}
    """
    assert np.isclose(flattened_sample, baseline).all(), err_msg

def test_sparse_dict(rebase=False):
    gym_dict = get_dict_2(get_dict_1(get_dict_0()))
    sfd = SparseFlatteningDict(deepcopy(gym_dict.spaces), seed = gym_dict._np_random)

    sfd.auto_flatten = False
    sfd.mode         = "sparse"

    sfd.seed(1)
    sample = sfd.sample()

    if rebase:
        np.save("baselines/flattened_sparse", sample)

    baseline = np.load("baselines/flattened_sparse.npy")

    err_msg = f"""
    Expected: {baseline}
    Found: {sample}
    """
    assert np.isclose(sample, baseline).all(), err_msg
    
    flattened_sample = sfd.sparse_flatten_sample(sample)

    err_msg = f"""
    Expected flattened_size attribute to match the shape attribute.
    flatttened_size: {sfd._flattened_size}
    shape: {sfd.shape}
    """
    assert flattened_sample.shape == (sfd._flattened_size,), err_msg

    err_msg = f"""
    Expected sparse flattened sample to not have any inf values, but found:
    {flattened_sample}
    """
    assert np.inf not in flattened_sample

    err_msg = f"""
    Expected: {baseline}
    Found: {flattened_sample}
    """
    assert np.isclose(flattened_sample, baseline).all(), err_msg

def test_omit_entry(rebase=False):
    gym_dict = get_dict_2(get_dict_1(get_all_inf_dict_0()))
    sfd = SparseFlatteningDict(deepcopy(gym_dict.spaces), seed = gym_dict._np_random)

    #
    # Make sure our sparse tree looks right.
    #
    sfd.mode  = "sparse"
    tree_str  = sfd.get_tree_str()

    if rebase:
        with open("baselines/sparse_tree_omit_baseline.txt", "w") as out_f:
            out_f.write(tree_str)

    with open("baselines/sparse_tree_omit_baseline.txt", "r") as in_f:
        tree_baseline = in_f.read()

    err_msg = f"""
    {tree_str}

    VS

    {tree_baseline}
    """
    assert tree_str == tree_baseline, err_msg

    #
    # Make sure our dense tree looks right after switching from
    # sparse to dense.
    #
    sfd.auto_flatten = False
    sfd.mode         = "dense"

    if rebase:
        with open("baselines/dense_tree_omit_baseline.txt", "w") as out_f:
            out_f.write(tree_str)

    with open("baselines/dense_tree_omit_baseline.txt", "r") as in_f:
        tree_baseline = in_f.read()

    err_msg = f"""
    {tree_str}

    VS

    {tree_baseline}
    """
    assert tree_str == tree_baseline, err_msg

    #
    # Make sure a dense sampling matches.
    #
    gym_dict.seed(1)
    gym_sample = gym_dict.sample()

    sfd.seed(1)
    sample = sfd.sample()
    err_msg = f"""
    Expected sfd.sample() to match gym_dict.sample().
    sfd sample: {sample}
    gym sample: {gym_sample}
    """
    assert (str(sample) == str(gym_sample)), err_msg

    #
    # Make sure we can flatten a dense sample.
    #
    flattened_sample = sfd.sparse_flatten_sample(sample)

    err_msg = f"""
    Expected flattened_size attribute to match size of flattened sample.
    flatttened_size: {sfd._flattened_size}
    flatttened_size: {flattened_sample.size}
    """
    assert flattened_sample.size == sfd._flattened_size, err_msg

    err_msg = f"""
    Expected sparse flattened sample to not have any inf values, but found:
    {flattened_sample}
    """
    assert np.inf not in flattened_sample

    if rebase:
        np.save("baselines/omit_entry", flattened_sample)

    baseline = np.load("baselines/omit_entry.npy")

    err_msg = f"""
    Expected: {baseline}
    Found: {flattened_sample}
    """
    assert np.isclose(flattened_sample, baseline).all(), err_msg
