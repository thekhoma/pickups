"""Connection to ComfyUI through the Comfy Developer Platform.

Pickups talks to ComfyUI only through the Comfy SDK. For a local or LAN ComfyUI, the SDK goes through Comfy's
`comfy-api-proxy` (the Comfy API v2 in front of any ComfyUI), which Pickups starts for you. Read-only machine
facts for the setup check (GPU, VRAM, installed models) come from ComfyUI's /system_stats and /object_info.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from comfy_sdk import Comfy, Progress, StatusChange

HOME = Path(os.environ.get("PICKUPS_HOME", os.path.expanduser("~/.pickups")))
SETTINGS = HOME / "settings.json"
PROXY_PORT = int(os.environ.get("PICKUPS_PROXY_PORT", "8189"))

# RunPod's HTTP proxy (Cloudflare) answers 403 to Python's default "Python-urllib" user agent, so every urllib read of
# a ComfyUI (status, model lists, history, /free) says who it is instead.
_opener = urllib.request.build_opener()
_opener.addheaders = [("User-Agent", "Pickups")]
urllib.request.install_opener(_opener)


class Canceled(Exception):
    pass


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text())
    except (OSError, ValueError):
        return {}


def save_settings(data: dict) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    SETTINGS.write_text(json.dumps(data, indent=2))


def _get_json(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - user-configured ComfyUI URL
        return json.loads(r.read())


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class Connection:
    comfy_url: str = "http://127.0.0.1:8188"
    proxy_port: int = PROXY_PORT
    _proxy: subprocess.Popen | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    # Bumped on every switch, so cached SDK clients (they hold the proxy's address) know to rebuild.
    generation: int = 0

    @classmethod
    def from_settings(cls, override: str | None = None) -> "Connection":
        url = override or os.environ.get("PICKUPS_COMFY_URL") or load_settings().get("comfy_url") or "http://127.0.0.1:8188"
        return cls(comfy_url=url.rstrip("/"))

    # --- the API proxy ------------------------------------------------------------------------------------------
    def ensure_proxy(self) -> str:
        """Start comfy-api-proxy for this ComfyUI if nothing is listening on the proxy port. Returns its base URL."""
        with self._lock:
            if not _port_open(self.proxy_port):
                exe = shutil.which("comfy-api-proxy")
                if not exe:
                    raise RuntimeError("comfy-api-proxy is not installed (pip install comfy-api-proxy)")
                state = HOME / "proxy"
                state.mkdir(parents=True, exist_ok=True)
                log = open(HOME / "proxy.log", "ab")  # noqa: SIM115 - lives as long as the proxy
                self._proxy = subprocess.Popen([exe, "run", "--comfyui", self.comfy_url, "--port", str(self.proxy_port),
                                                "--state-dir", str(state)], stdout=log, stderr=log)
                for _ in range(50):
                    if _port_open(self.proxy_port):
                        break
                    time.sleep(0.2)
        return f"http://127.0.0.1:{self.proxy_port}"

    def stop_proxy(self) -> None:
        if self._proxy and self._proxy.poll() is None:
            self._proxy.terminate()
            try:
                self._proxy.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proxy.kill()
        self._proxy = None

    def switch(self, url: str) -> str:
        """Point Pickups at another ComfyUI. comfy-api-proxy serves one ComfyUI, so stop the one this app started and
        start a fresh one for the new address. A proxy this app didn't start (still on the port) is left alone and
        the new one takes a free port. Returns the new proxy's base URL."""
        url = url.rstrip("/")
        with self._lock:
            self.stop_proxy()
            for _ in range(10):  # the old proxy lets go of its port
                if not _port_open(self.proxy_port):
                    break
                time.sleep(0.1)
            else:
                self.proxy_port = _free_port()
            self.comfy_url = url
            self.generation += 1
        # The new proxy starts in the background: the setup check talks to ComfyUI directly, so Connect doesn't wait
        # for it (up to 10 s on a cold start). A job's client() waits on the same lock until it is up.
        threading.Thread(target=self._start_proxy_quietly, daemon=True).start()
        return f"http://127.0.0.1:{self.proxy_port}"

    def _start_proxy_quietly(self) -> None:
        try:
            self.ensure_proxy()
        except RuntimeError:  # not installed: client() raises it again when a job needs the proxy
            pass

    def client(self) -> Comfy:
        os.environ["COMFY_BASE_URL"] = self.ensure_proxy()
        # Uploading a long 4K work clip to a remote GPU (a cloud pod through a tunnel) can take minutes; the SDK's 30 s
        # default fails those uploads. Local and LAN connections never get near this.
        return Comfy(client_info="pickups", timeout=float(os.environ.get("PICKUPS_HTTP_TIMEOUT", 600)))

    # --- machine facts ------------------------------------------------------------------------------------------
    def system_stats(self) -> dict | None:
        try:
            return _get_json(f"{self.comfy_url}/system_stats")
        except Exception:  # noqa: BLE001 - offline is a normal state
            return None

    # The loader whose choice list shows each models folder (so a folder whose loader didn't answer is unknown, not missing).
    LOADERS = {"UNETLoader": ("unet_name", "diffusion_models"), "CLIPLoader": ("clip_name", "text_encoders"),
               "VAELoader": ("vae_name", "vae"), "LoraLoaderModelOnly": ("lora_name", "loras"),
               "CheckpointLoaderSimple": ("ckpt_name", "checkpoints"), "UpscaleModelLoader": ("model_name", "upscale_models"),
               "LatentUpscaleModelLoader": ("model_name", "latent_upscale_models"), "CLIPVisionLoader": ("clip_name", "clip_vision"),
               "LoadDA3Model": ("model_name", "geometry_estimation"), "ModelPatchLoader": ("name", "model_patches")}

    def model_scan(self) -> tuple[set[str], set[str]]:
        """Every model file name ComfyUI can see (from the loader nodes' choice lists), and the models folders it couldn't
        list (that loader's query failed or timed out): their files are unknown, not missing."""
        names: set[str] = set()
        unknown: set[str] = set()
        for node, (field_name, folder) in self.LOADERS.items():
            try:
                spec = _get_json(f"{self.comfy_url}/object_info/{node}")[node]["input"]["required"][field_name]
                opts = spec[0] if isinstance(spec[0], list) else spec[1].get("options", [])
                names.update(opts)
            except Exception:  # noqa: BLE001
                unknown.add(folder)
        return names, unknown

    def model_files(self) -> set[str]:
        """Every model file name ComfyUI can see (from the loader nodes' choice lists)."""
        return self.model_scan()[0]

    def node_types(self) -> set[str]:
        try:
            return set(_get_json(f"{self.comfy_url}/object_info", timeout=20).keys())
        except Exception:  # noqa: BLE001
            return set()

    def cancel_prompt(self, prompt_id: str) -> str | None:
        """Stop one job this app sent, by its prompt id, and nothing else: still waiting in ComfyUI's queue, it is
        removed ("removed"); the one running, it is interrupted ("interrupted"). Anything else (finished, unknown, or
        another job running) is left alone (None). Never a bare /interrupt: that stops whatever is running."""
        if not prompt_id or not is_ours(prompt_id):
            return None  # only ever a prompt this app sent
        try:
            q = _get_json(f"{self.comfy_url}/queue")
        except Exception:  # noqa: BLE001 - offline: nothing to stop there
            return None

        def ids(items) -> set[str]:
            return {str(it[1]) for it in items or [] if isinstance(it, (list, tuple)) and len(it) > 1}

        if prompt_id in ids(q.get("queue_pending")):
            self._post("/queue", {"delete": [prompt_id]})
            return "removed"
        if prompt_id in ids(q.get("queue_running")):
            # ComfyUI interrupts only when the running prompt is this one (newer builds check prompt_id themselves).
            self._post("/interrupt", {"prompt_id": prompt_id})
            return "interrupted"
        return None

    def wait_stopped(self, prompt_id: str | None, timeout: float = 30.0, every: float = 0.5) -> bool:
        """Wait until ComfyUI no longer runs or queues this prompt (an interrupt lands at the next node: R&D's Run 9
        Finish kept loading a VAE for 8 s). True once it's gone, or when ComfyUI can't be asked; False on the timeout."""
        if not prompt_id:
            return True
        end = time.monotonic() + timeout
        while True:
            try:
                q = _get_json(f"{self.comfy_url}/queue")
            except Exception:  # noqa: BLE001 - offline: nothing is running there
                return True
            busy = {str(it[1]) for k in ("queue_running", "queue_pending") for it in q.get(k) or []
                    if isinstance(it, (list, tuple)) and len(it) > 1}
            if str(prompt_id) not in busy:
                return True
            if time.monotonic() >= end:
                return False
            time.sleep(every)

    def _post(self, path: str, body: dict) -> None:
        req = urllib.request.Request(f"{self.comfy_url}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=10).read()  # noqa: S310 - user-configured ComfyUI URL
        except Exception:  # noqa: BLE001 - best effort; the job's own cancel still lands
            pass

    def free_memory(self) -> None:
        """Ask ComfyUI to unload models between heavy stages (keeps 24 GB cards out of trouble)."""
        try:
            req = urllib.request.Request(f"{self.comfy_url}/free", data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=10)  # noqa: S310
        except Exception:  # noqa: BLE001
            pass


REJECTED = "ComfyUI rejected the workflow{what}. A node or model it needs may be missing or out of date."
UNREACHABLE = "Couldn't reach ComfyUI. Check that it's running, then try again."
UPLOAD_REFUSED = "ComfyUI turned down a file this job sent. Try again; if it repeats, restart ComfyUI."
_REJECT_RX = r"(?i)failed validation|invalid ?workflow|node_errors|value not in list|required input is missing|exception during inner validation"
_UNREACH_RX = r"(?i)failed to reach comfyui|connection (refused|reset|aborted)|cannot connect|connecterror|name or service not known|timed? ?out|unreachable"


def _first_node_error(details) -> str:
    """": <node> (<its message>)" from a validation error's per-node detail, else ""."""
    nodes = details.get("node_errors", details) if isinstance(details, dict) else {}
    for nid, n in (nodes.items() if isinstance(nodes, dict) else []):
        if not isinstance(n, dict):
            continue
        errs = n.get("errors") or []
        msg = next((str(x.get("message") or x.get("details") or "") for x in errs if isinstance(x, dict)), "")
        who = n.get("class_type") or f"node {nid}"
        return f": {who}" + (f" ({msg})" if msg else "")
    return ""


def plain_error(e: BaseException, reachable=None) -> tuple[str, str]:
    """(the sentence a person reads, the raw detail for Details) for a failed job (REVIEW F3). A workflow ComfyUI
    rejected says so with its node; "can't reach" is said only when ComfyUI really doesn't answer (reachable(): a
    quick check, so the proxy's "Failed to reach ComfyUI for upload" after a rejection isn't taken at its word)."""
    raw = str(e) or e.__class__.__name__
    details = getattr(e, "details", None)
    detail = raw + (f"\n{json.dumps(details, indent=1, default=str)[:4000]}" if details else "")
    name = e.__class__.__name__
    if name in ("InvalidWorkflow", "WorkflowFormatUi") or re.search(_REJECT_RX, raw):
        return REJECTED.format(what=_first_node_error(details)), detail
    if re.search(_UNREACH_RX, raw):
        up = True
        try:
            up = bool(reachable()) if reachable else False
        except Exception:  # noqa: BLE001 - a failed check means not reachable
            up = False
        return (UPLOAD_REFUSED if up and "upload" in raw.lower() else REJECTED.format(what="") if up else UNREACHABLE), detail
    if re.search(r"(?i)rejected the upload", raw):
        return UPLOAD_REFUSED, detail
    return raw[:500], detail


class Stopper:
    """What a run carries so Cancel reaches it: the cancel flag, the prompt id ComfyUI gave the graph running now, and
    how to stop that one prompt (Connection.cancel_prompt)."""

    def __init__(self, cancel: threading.Event | None = None, stop_prompt=None):
        self.cancel = cancel or threading.Event()
        self.stop_prompt = stop_prompt
        self.prompt_id: str | None = None


# Every prompt id ComfyUI gave a graph this app submitted (run_graph). Cancel only ever stops one of these: another app's
# job on the same ComfyUI is never touched, even if its id is passed in.
_ours: set[str] = set()
_ours_lock = threading.Lock()


def mark_ours(prompt_id: str) -> None:
    with _ours_lock:
        _ours.add(str(prompt_id))


def is_ours(prompt_id: str) -> bool:
    with _ours_lock:
        return str(prompt_id) in _ours


# The inline lane (the start frame, the look frame) runs in a request's own thread: its Stopper rides here, so the quick
# runner's graphs can be canceled without passing it through every helper.
_current = threading.local()


def current_stopper() -> Stopper | None:
    return getattr(_current, "stopper", None)


def set_current_stopper(stopper: Stopper | None) -> None:
    _current.stopper = stopper


def check_canceled() -> None:
    """Raise Canceled if this thread's job was canceled (local work between graphs: downloads, merges, encodes)."""
    s = current_stopper()
    if s is not None and s.cancel.is_set():
        raise Canceled()


def run_local(args: list[str], **kw) -> None:
    """subprocess.run(args, check=True) that a Cancel stops: the process is killed within a quarter second and Canceled
    is raised (a 4K ProRes encode can take minutes after the last graph)."""
    import subprocess

    check_canceled()
    proc = subprocess.Popen(args, **kw)
    while True:
        try:
            code = proc.wait(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            s = current_stopper()
            if s is not None and s.cancel.is_set():
                proc.kill()
                proc.wait()
                raise Canceled() from None
    if code:
        raise subprocess.CalledProcessError(code, args)


_END = object()


def _stop(job, stopper: Stopper | None) -> None:
    """Stop this job's prompt in ComfyUI (removed from the queue, or interrupted if it is the one running), then tell the
    proxy so its record ends too. Best effort: Canceled is raised either way."""
    if stopper is not None and stopper.stop_prompt:
        try:
            stopper.stop_prompt(job.id)
        except Exception:  # noqa: BLE001
            pass
    try:
        job.cancel()  # the proxy's per-id cancel (never a bare interrupt)
    except Exception:  # noqa: BLE001
        pass


def run_graph(client: Comfy, graph: dict, on_progress=None, cancel: threading.Event | None = None, stopper: Stopper | None = None):
    """Submit an API-format graph and wait. Reports fractional progress; honours cancel within a quarter second, even
    while ComfyUI sends nothing (a job still waiting in its queue), and then stops this graph's own prompt by its id
    (the thread's Stopper: the queue worker and the inline lane set one). Returns the finished job."""
    stopper = stopper or current_stopper()  # a queued job's or an inline row's, set on this thread
    if stopper is not None and cancel is not None and stopper.cancel is not cancel:
        stopper = None  # someone else's flag: honour it, but don't stop by a prompt id it doesn't own
    cancel = cancel if cancel is not None else (stopper.cancel if stopper else None)
    if cancel is not None and cancel.is_set():
        raise Canceled()
    job = client.submit(client.workflows.from_json(graph))
    mark_ours(job.id)
    if stopper is not None:
        stopper.prompt_id = job.id
    events: queue.Queue = queue.Queue()

    def pump() -> None:
        try:
            for ev in job.events():
                events.put(ev)
        except BaseException as e:  # noqa: BLE001 - handed to the waiting thread
            events.put(e)
        events.put(_END)

    threading.Thread(target=pump, daemon=True, name="pickups-events").start()
    while True:
        if cancel is not None and cancel.is_set():
            _stop(job, stopper)
            raise Canceled()
        try:
            ev = events.get(timeout=0.25)
        except queue.Empty:
            continue
        if ev is _END:
            break
        if isinstance(ev, BaseException):
            raise ev
        if isinstance(ev, Progress) and on_progress:
            frac = ev.value if ev.value is not None else 0.0
            if ev.steps:
                frac = (ev.step or 0) / ev.steps
            on_progress(max(0.0, min(1.0, float(frac))))
        if isinstance(ev, StatusChange) and ev.status in ("succeeded", "failed", "canceled", "expired"):
            break
    try:
        return job.result()
    except Exception:
        if cancel is not None and cancel.is_set():
            raise Canceled() from None
        raise
