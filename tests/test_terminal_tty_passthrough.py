"""TERM.15: real, pty-driven proof of TERM.7's interactive-program path.

This does not simulate a keypress reaching the child instead of the wrapper
directly -- there is no real human at a real keyboard in a test. It proves
the two OS-level facts that *guarantee* that outcome, which the kernel's own
tty driver enforces unconditionally once they hold: (1) the interactive
child ends up in its own, new process group, and (2) that group is the real
controlling terminal's foreground process group while the child runs,
restored back to the wrapper's own group once the child exits. Given those
two facts, correct signal routing (Ctrl-C/Ctrl-Z/Ctrl-\\ reaching only the
child) is a direct, kernel-enforced consequence, not something this test
needs to separately re-verify by sending real key sequences.

Uses `os.forkpty()` to give the "wrapper" process (standing in for the real
`aios-terminal` REPL process) a genuine controlling terminal, since
`execution._run_with_terminal_control` refuses to attempt job control at all
without a real tty on stdin (see that function's own doc comment) -- this is
exactly the condition it needs to actually exercise the code path under
test, not the piped/captured-stdio fallback.
"""

import json
import os
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from vectros_sdk.terminal import execution
from vectros_sdk.terminal.execution import _run_with_terminal_control


class TestInteractiveTerminalControl(unittest.TestCase):
    def test_an_interactive_command_becomes_and_then_relinquishes_the_real_foreground_group(self):
        self._assert_real_job_control(force_pre_311_fallback=False)

    def test_the_pre_3_11_preexec_fn_fallback_path_also_works_for_real(self):
        # `_run_with_terminal_control` picks its mechanism (`process_group=0`
        # vs `preexec_fn=os.setpgrp`) by `sys.version_info` -- forcing the
        # older branch here proves *that* path too, live, rather than
        # trusting it by inspection just because it is the smaller/older
        # code path.
        self._assert_real_job_control(force_pre_311_fallback=True)

    def _assert_real_job_control(self, *, force_pre_311_fallback: bool) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_path = os.path.join(tmp, "results.json")
            pid, master_fd = os.forkpty()
            if pid == 0:
                self._run_wrapper_child(results_path, force_pre_311_fallback)
            else:
                self._drive_and_assert(pid, master_fd, results_path)

    def _run_wrapper_child(self, results_path: str, force_pre_311_fallback: bool) -> None:
        """Runs post-fork inside the new pty session -- a real controlling
        terminal and a real session leader, exactly like a real interactive
        shell process. Never returns to the caller (always `os._exit`)."""
        try:
            wrapper_pgid = os.getpgid(0)
            probe_script = textwrap.dedent(
                f"""
                import json, os
                result = {{
                    "child_pgid": os.getpgid(0),
                    "child_is_foreground": os.tcgetpgrp(0) == os.getpgid(0),
                    "wrapper_pgid": {wrapper_pgid},
                }}
                with open({results_path!r}, "w") as f:
                    json.dump(result, f)
                """
            )
            argv = [sys.executable, "-c", probe_script]
            if force_pre_311_fallback:
                with mock.patch.object(execution.sys, "version_info", (3, 10, 0)):
                    _run_with_terminal_control(argv)
            else:
                _run_with_terminal_control(argv)
            restored = os.tcgetpgrp(0) == os.getpgid(0)
            with open(results_path) as f:
                result = json.load(f)
            result["wrapper_restored_as_foreground_afterward"] = restored
            with open(results_path, "w") as f:
                json.dump(result, f)
        finally:
            os._exit(0)  # noqa: SLF001 - real exit of a forked child, not the test process

    def _drive_and_assert(self, pid: int, master_fd: int, results_path: str) -> None:
        # Drain the pty master (the child's stdio all points at the pty
        # slave) so the child never blocks on a full pty output buffer;
        # content is not asserted on here, only the results file is.
        try:
            while True:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
        finally:
            os.close(master_fd)
        os.waitpid(pid, 0)

        self.assertTrue(os.path.exists(results_path), "wrapper child never wrote its results")
        with open(results_path) as f:
            result = json.load(f)

        self.assertTrue(
            result["child_is_foreground"],
            "the interactive program must become the terminal's real foreground process group",
        )
        self.assertNotEqual(
            result["child_pgid"],
            result["wrapper_pgid"],
            "the interactive program must run in its own process group, not the wrapper's",
        )
        self.assertTrue(
            result["wrapper_restored_as_foreground_afterward"],
            "the wrapper must regain foreground status once the interactive program exits",
        )


if __name__ == "__main__":
    unittest.main()
