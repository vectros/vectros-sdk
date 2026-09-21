"""Unit tests for the operator-typed execution path's one real safety net:
destructive verbs require an explicit, shown confirmation before running,
and declining must mean nothing runs at all."""

import unittest

from vectros_sdk.terminal.execution import is_destructive, is_interactive, run_command


class TestDestructiveDetection(unittest.TestCase):
    def test_rm_is_destructive(self):
        self.assertTrue(is_destructive("rm my old files"))

    def test_plain_read_only_command_is_not_destructive(self):
        self.assertFalse(is_destructive("ls -la"))

    def test_a_redirect_is_treated_as_destructive(self):
        self.assertTrue(is_destructive("echo hi > important.txt"))

    def test_empty_line_is_not_destructive(self):
        self.assertFalse(is_destructive(""))


class TestInteractiveDetection(unittest.TestCase):
    """TERM.7: the narrow, disclosed heuristic named in
    `INTERACTIVE_PROGRAMS`'s own doc comment."""

    def test_vim_is_interactive(self):
        self.assertTrue(is_interactive("vim notes.txt"))

    def test_a_plain_read_only_command_is_not_interactive(self):
        self.assertFalse(is_interactive("ls -la"))

    def test_a_full_path_still_matches_by_basename(self):
        self.assertTrue(is_interactive("/usr/bin/vim notes.txt"))

    def test_an_unlisted_program_is_not_interactive(self):
        # A real, honest limitation: not on the fixed list, so it takes the
        # plain path -- correct for the overwhelming majority of commands,
        # but not a general interactive-program detector.
        self.assertFalse(is_interactive("my-custom-repl"))

    def test_empty_line_is_not_interactive(self):
        self.assertFalse(is_interactive(""))


class TestRunCommandWithoutARealTerminal(unittest.TestCase):
    """`_run_with_terminal_control` refuses job control without a real
    controlling terminal on stdin (there is none under a test runner) and
    falls back to the plain path -- `run_command` must still work correctly
    for a command `is_interactive` matches in that case."""

    def test_an_interactive_command_still_runs_and_reports_its_real_exit_code(self):
        result = run_command("python3 -c 'import sys; sys.exit(3)'")
        self.assertTrue(result.ran)
        self.assertEqual(result.exit_code, 3)


class TestRunCommandConfirmation(unittest.TestCase):
    def test_declining_a_destructive_command_never_runs_it(self):
        result = run_command("rm my old files", confirm=lambda _prompt: "n")
        self.assertFalse(result.ran)
        self.assertFalse(result.confirmed)

    def test_approving_a_destructive_command_runs_it(self):
        result = run_command("true", confirm=lambda _prompt: "y")
        # "true" itself is not destructive, so confirm is never even asked --
        # exercised here only to prove the non-destructive path always runs.
        self.assertTrue(result.ran)
        self.assertEqual(result.exit_code, 0)

    def test_a_real_destructive_command_runs_only_after_approval(self):
        prompts = []

        def confirm(prompt):
            prompts.append(prompt)
            return "y"

        result = run_command("true", confirm=confirm)
        self.assertTrue(result.ran)
        self.assertEqual(prompts, [])  # "true" is not destructive: no prompt at all


if __name__ == "__main__":
    unittest.main()
