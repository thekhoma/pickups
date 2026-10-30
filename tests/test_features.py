from pickups import features, models_catalog
from pickups.tasks import recast_ltx


def test_the_contest_build_offers_no_car_swap(monkeypatch):
    monkeypatch.delenv("PICKUPS_CARS", raising=False)
    assert not features.cars()
    assert "car" not in models_catalog.workflows() and "character" in models_catalog.workflows()
    assert "isn't in this version" in recast_ltx.NO_CARS
    monkeypatch.setenv("PICKUPS_CARS", "1")
    assert features.cars() and "car" in models_catalog.workflows()


def test_car_words_never_show_the_car_steps_when_cars_are_off(monkeypatch):
    from pickups.tasks import TASKS
    monkeypatch.delenv("PICKUPS_CARS", raising=False)
    labels = [st.label for st in TASKS["recast"].stages_of({"text": "the SUV", "frame": 0, "points": [[1, 1]]})]
    person = [st.label for st in TASKS["recast"].stages_of({"text": "the man", "frame": 0, "points": [[1, 1]]})]
    assert labels == person  # the job refuses the car (NO_CARS) when it runs; until then it looks like any recast
