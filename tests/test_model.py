"""Model behaviour on the packaged model (skipped if it hasn't been trained)."""

import pytest

from soccer_xg import config, model, scene

pytestmark = pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="models/xg_model.joblib not built")


@pytest.fixture(scope="module")
def xg():
    fitted = model.load_model()
    return lambda *a, **k: float(fitted.predict(scene.scene_to_shot(*a, **k))[0])


def test_closer_and_more_central_is_better(xg):
    assert xg((112, 40)) > xg((105, 40)) > xg((95, 40)) > xg((85, 40))
    for x in (95, 105, 112):
        assert xg((x, 40)) >= xg((x, 30)) >= xg((x, 20)) >= xg((x, 5))


def test_defenders_and_keeper_matter(xg):
    assert xg((108, 40), keeper=False) > xg((108, 40)) > xg((108, 40), defenders=[(113, 39), (114, 41)])


def test_penalty_gets_the_constant(xg):
    fitted = model.load_model()
    assert xg((108, 40), shot_type="Penalty") == pytest.approx(fitted.penalty_xg)
    assert 0.7 < fitted.penalty_xg < 0.85
