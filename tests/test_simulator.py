import pytest

import networkx as nx

import tsgm


def test_sine_cosine_simulator():
    data = tsgm.dataset.DatasetProperties(N=10, T=12, D=23)
    sin_cosine_sim = tsgm.simulator.SineConstSimulator(data)
    syn_dataset = sin_cosine_sim.generate(10)
    assert type(syn_dataset) is tsgm.dataset.Dataset

    assert syn_dataset.X.shape == (10, 12, 23)
    assert syn_dataset.y.shape == (10,)
