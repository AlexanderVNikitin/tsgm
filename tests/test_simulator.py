import numpy as np
from unittest.mock import MagicMock
import tsgm


def test_sine_cosine_simulator():
    data = tsgm.dataset.DatasetProperties(N=10, T=12, D=23)
    sin_cosine_sim = tsgm.simulator.SineConstSimulator(data)
    syn_dataset = sin_cosine_sim.generate(10)
    assert type(syn_dataset) is tsgm.dataset.Dataset
    params = sin_cosine_sim.params() 
    assert params["max_scale"] == 10.0 and params["max_const"] == 5.0
    assert syn_dataset.X.shape == (10, 12, 23)
    assert syn_dataset.y.shape == (10,)

    sin_cosine_sim.set_params(max_scale=10, max_const=123)
    params = sin_cosine_sim.params()
    assert params["max_scale"] == 10.0 and params["max_const"] == 123

    new_sim = sin_cosine_sim.clone()
    params1 = sin_cosine_sim.params()
    params2 = new_sim.params()
    assert params1.keys() == params2.keys() and params1["max_scale"] == params2["max_scale"] and params1["max_const"] == params2["max_const"]


def test_simulator_base():
    MockDriver = MagicMock()
    data = tsgm.dataset.Dataset(x=np.ones((3, 2, 1)), y=np.ones(3))
    s = tsgm.simulator.Simulator(data, MockDriver)
    cloned = s.clone()
    assert np.array_equal(s._data.X, cloned._data.X) and np.array_equal(s._data.y, cloned._data.y)

    s.fit()
    MockDriver.fit.assert_called_once_with(s._data.X, s._data.y)


def test_pdm_simulator():
    n_samples = 10
    data = tsgm.dataset.DatasetProperties(N=100, T=12, D=23)
    pdm_simulator = tsgm.simulator.PredictiveMaintenanceSimulator(data)
    syn_dataset, equipment = pdm_simulator.generate(n_samples)
    assert len(equipment) == 10
    assert len(syn_dataset) == 10
    for d in equipment:
        assert isinstance(d, dict)

    new_sim = pdm_simulator.clone()
    params1 = pdm_simulator.params()
    params2 = new_sim.params()
    assert params1["switches"] == params2["switches"]
    assert params1["m_norms"] == params2["m_norms"]
    assert params1["sigma_norms"] == params2["sigma_norms"]


def test_lv_simulator():
    n_samples = 10
    data = tsgm.dataset.DatasetProperties(N=100, T=12, D=23)
    lv_simulator = tsgm.simulator.LotkaVolterraSimulator(data)
    syn_dataset = lv_simulator.generate(n_samples)
    assert syn_dataset.shape == (10, 2)

    new_sim = lv_simulator.clone()
    params1 = lv_simulator.params()
    params2 = new_sim.params()
    for k in ["alpha", "beta", "gamma", "delta", "x0", "y0"]:
        assert params1[k] == params2[k]
