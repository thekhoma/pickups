import shutil
from pathlib import Path

import pytest

from pickups.app import comfy, machines
from pickups.tasks import common

_REAL_PORT_OPEN = comfy._port_open
WEB = Path(__file__).resolve().parents[1] / "web"


@pytest.fixture()
def web_node() -> Path:
    """The web folder, for a test that runs the page's own TypeScript with Node. Skips when Node isn't installed."""
    if not shutil.which("node"):
        pytest.skip("Node.js isn't installed (needed to run the web code)")
    return WEB


@pytest.fixture()
def web_modules(web_node) -> Path:
    """The web folder, for a test whose TypeScript imports npm packages (zod). Skips until `cd web && npm install`."""
    if not (WEB / "node_modules").is_dir():
        pytest.skip("web/node_modules is missing: run `cd web && npm install` to run this test")
    return web_node


@pytest.fixture(autouse=True)
def _never_a_real_comfyui(monkeypatch):
    """A test never reaches a real ComfyUI: with a live proxy on this computer's proxy port (a running app's), a test
    client would send its graphs through it to that GPU. Only a test that fakes the ports (or the proxy) gets one."""
    real = comfy.Connection.ensure_proxy

    def guarded(self):
        if comfy._port_open is _REAL_PORT_OPEN:
            raise RuntimeError("Tests never reach a real ComfyUI or proxy.")
        return real(self)

    monkeypatch.setattr(comfy.Connection, "ensure_proxy", guarded)


@pytest.fixture(autouse=True)
def _machine_id_is_per_test(tmp_path, monkeypatch):
    """The unsaved machine's id file lives in the test's own folder, never in a real Pickups home."""
    monkeypatch.setattr(machines, "_id_path", lambda: tmp_path / "machine_id")


@pytest.fixture(autouse=True)
def _max_frames_is_per_test():
    """The longest take is set from the connected GPU (common.use_gpu): a test with a big fake GPU must not change it for the next."""
    keep = common.MAX_FRAMES
    yield
    common.MAX_FRAMES = keep


CAR_TESTS = {"test_vehicle_swap", "test_car_blend", "test_recast_vehicle"}


@pytest.fixture(autouse=True)
def _cars_on_for_car_tests(request, monkeypatch):
    """Car swap is off in the contest build (pickups.features); its own tests run with it on."""
    if request.module.__name__.rsplit(".", 1)[-1] in CAR_TESTS:
        monkeypatch.setenv("PICKUPS_CARS", "1")
