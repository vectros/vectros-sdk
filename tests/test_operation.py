import socket
import tempfile
import threading
import unittest

from vectros_sdk import OperationClient, OperationInterfaceError


class OperationClientTests(unittest.TestCase):
    def _server(self, replies):
        directory = tempfile.TemporaryDirectory()
        path = f"{directory.name}/aios.sock"
        ready = threading.Event()

        def run():
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(path)
                server.listen()
                ready.set()
                for expected, response in replies:
                    connection, _ = server.accept()
                    with connection:
                        self.assertEqual(connection.recv(4096).decode(), expected)
                        connection.sendall(response.encode())

        thread = threading.Thread(target=run)
        thread.start()
        ready.wait(timeout=2)
        return directory, path, thread

    def test_lifecycle_and_payload_free_watch(self):
        directory, path, thread = self._server([
            ("SUBMIT agent:one instance:one run:one operation:one infer read_only\n", "OK queued operation:one\n"),
            ("ADMIT\n", "OK admitted operation:one\n"),
            ("START operation:one\n", "OK running operation:one\n"),
            ("WATCH agent:one run:one 0\n", "EVENT 1 OperationQueued ignored-payload\nEVENT 2 OperationStarted ignored-payload\n"),
        ])
        try:
            client = OperationClient(path)
            self.assertEqual(client.submit(agent_id="agent:one", instance_id="instance:one", run_id="run:one", operation_id="operation:one", kind="infer", side_effect="read_only").state, "queued")
            self.assertEqual(client.admit_next().state, "admitted")
            self.assertEqual(client.start("operation:one").state, "running")
            self.assertEqual([(event.cursor, event.event_type) for event in client.watch("agent:one", "run:one")], [(1, "OperationQueued"), (2, "OperationStarted")])
        finally:
            thread.join(timeout=2)
            directory.cleanup()

    def test_tool_and_malformed_commands_are_rejected_before_connecting(self):
        client = OperationClient("/tmp/operation-client-test.sock")
        with self.assertRaises(ValueError):
            client.submit(agent_id="agent:one", instance_id="instance:one", run_id="run:one", operation_id="operation:one", kind="tool", side_effect="read_only")
        with self.assertRaises(ValueError):
            client.get("operation with space")
        with self.assertRaises(ValueError):
            OperationClient("relative.sock")

    def test_daemon_error_is_stable_and_payload_free(self):
        directory, path, thread = self._server([("GET operation:missing\n", "ERR NOT_FOUND\n")])
        try:
            with self.assertRaises(OperationInterfaceError) as context:
                OperationClient(path).get("operation:missing")
            self.assertEqual(context.exception.code, "NOT_FOUND")
        finally:
            thread.join(timeout=2)
            directory.cleanup()
