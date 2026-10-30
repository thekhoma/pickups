import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from pickups import director


def _stub(replies):
    calls = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            text = replies[min(len(calls) - 1, len(replies) - 1)]
            out = json.dumps({"choices": [{"message": {"content": text}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, calls


def test_director_retries_when_the_answer_is_not_english():
    srv, calls = _stub(["一位老司机坐在车里。", "An old driver sits in his car as the camera pushes in."])
    cfg = {"url": f"http://127.0.0.1:{srv.server_port}", "model": "m"}
    out = director.enhance_openai(cfg, "http://127.0.0.1:9", "ltx", "old driver in his car")
    assert out == "An old driver sits in his car as the camera pushes in."
    assert len(calls) == 2
    assert "English" in calls[0]["messages"][0]["content"]  # the system prompt asks for English
    srv.shutdown()


def test_director_cleans_thinking_and_aspect_lines():
    assert director.clean("<think>hmm</think>A red car on a coast road.\nAspect ratio: 16:9") == "A red car on a coast road."
