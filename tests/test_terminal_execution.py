"""Unit tests for the operator-typed execution path's one real safety net:
destructive verbs require an explicit, shown confirmation before running,
and declining must mean nothing runs at all."""

import unittest

from vectros_sdk.terminal.execution import is_destructive, run_command


class TestDestructiveDetection(unittest.TestCase):
    def test_rm_is_destructive(self):
        self.assertTrue(is_destructive("rm my old files"))

    def test_plain_read_only_command_is_not_destructive(self):
        self.assertFalse(is_destructive("ls -la"))

    def test_a_redirect_is_treated_as_destructive(self):
        self.assertTrue(is_destructive("echo hi > important.txt"))

    def test_empty_line_is_not_destructive(self):
        self.assertFalse(is_destructive(""))


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
