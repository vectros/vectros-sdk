import socket
import threading
import time

from vectros import _tracing
from vectros._tracing import TraceSpan, flush_spans


def test_flush_waits_until_queued_spans_are_sent(tmp_path, monkeypatch):
    path = str(tmp_path / "ingest.sock")
    monkeypatch.setattr(_tracing, "SOCKET_PATH", path)
    received = []
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(8)

    def accept(count):
        for _ in range(count):
            conn, _ = server.accept()
            with conn:
                time.sleep(0.1)  # A collector that is slower than the agent.
                received.append(conn.recv(65536))

    thread = threading.Thread(target=accept, args=(2,), daemon=True)
    thread.start()
    root = TraceSpan.root("agent.flush-test", "AGENT", 42)
    root.child("aios.planning", "AGENT").finish("step")
    root.finish("answer")

    assert flush_spans(timeout=5)
    thread.join(timeout=5)
    assert len(received) == 2
    assert b"agent.flush-test" in received[1]
    server.close()


def test_flush_gives_up_after_its_timeout(tmp_path, monkeypatch):
    path = str(tmp_path / "ingest.sock")
    monkeypatch.setattr(_tracing, "SOCKET_PATH", path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(8)
    release = threading.Event()
    sender = _tracing._BestEffortSender()
    monkeypatch.setattr(sender, "_run", lambda: release.wait())
    sender.send({"name": "stuck", "attributes": {"aios.agent.id": 42}})

    started = time.monotonic()
    assert not sender.flush(timeout=0.2)
    assert time.monotonic() - started < 2
    release.set()
    server.close()
