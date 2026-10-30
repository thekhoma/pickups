"""`pickups --port`: a taken port says how to start on another one, before anything starts."""

import socket
import sys

from pickups import cli


def test_a_taken_port_says_to_use_another(monkeypatch, capsys):
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen()
        port = held.getsockname()[1]
        monkeypatch.setattr(sys, "argv", ["pickups", "--no-browser", "--port", str(port)])
        assert cli.main() == 1
    out = capsys.readouterr().out
    assert f"Port {port} is already in use" in out and f"uv run pickups --port {port + 2}" in out


def test_a_free_port_is_free():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert cli._port_free("127.0.0.1", port)
