"""TERM.13: the dispatch table, including the two cases plan.md names
explicitly -- `rm my old files` (a real command, because `rm` is in PATH:
dispatch's job is to recognize that, not to be clever about intent) and
`gti status` (a typo, not in PATH, so it must fall to natural language
rather than doing nothing or erroring as an unknown command)."""

import unittest

from vectros_sdk.terminal.dispatch import DispatchKind, classify


def _which(known):
    return lambda name: f"/usr/bin/{name}" if name in known else None


class TestDispatchTable(unittest.TestCase):
    def test_a_real_command_in_path_dispatches_as_a_command(self):
        which = _which({"ls", "rm"})
        result = classify("ls -la", which=which)
        self.assertEqual(result.kind, DispatchKind.COMMAND)
        self.assertEqual(result.text, "ls -la")
        self.assertFalse(result.forced)

    def test_rm_my_old_files_dispatches_as_the_command_rm_because_rm_is_in_path(self):
        which = _which({"rm"})
        result = classify("rm my old files", which=which)
        self.assertEqual(result.kind, DispatchKind.COMMAND)
        self.assertEqual(result.text, "rm my old files")

    def test_a_typo_not_in_path_falls_to_natural_language(self):
        which = _which({"git"})  # "gti" deliberately absent
        result = classify("gti status", which=which)
        self.assertEqual(result.kind, DispatchKind.NATURAL_LANGUAGE)
        self.assertEqual(result.text, "gti status")

    def test_plain_english_with_no_path_token_is_natural_language(self):
        which = _which({"ls"})
        result = classify("which of these files is biggest?", which=which)
        self.assertEqual(result.kind, DispatchKind.NATURAL_LANGUAGE)

    def test_bang_forces_command_even_if_not_in_path(self):
        which = _which(set())
        result = classify("!definitely-not-a-real-binary --help", which=which)
        self.assertEqual(result.kind, DispatchKind.COMMAND)
        self.assertEqual(result.text, "definitely-not-a-real-binary --help")
        self.assertTrue(result.forced)

    def test_question_mark_forces_natural_language_even_if_in_path(self):
        which = _which({"ls"})
        result = classify("?ls", which=which)
        self.assertEqual(result.kind, DispatchKind.NATURAL_LANGUAGE)
        self.assertEqual(result.text, "ls")
        self.assertTrue(result.forced)

    def test_unbalanced_quotes_fall_back_to_natural_language_rather_than_guessing(self):
        which = _which({"echo"})
        result = classify('echo "unterminated', which=which)
        self.assertEqual(result.kind, DispatchKind.NATURAL_LANGUAGE)

    def test_empty_line_is_natural_language_and_never_a_command(self):
        result = classify("   ", which=_which(set()))
        self.assertEqual(result.kind, DispatchKind.NATURAL_LANGUAGE)
        self.assertEqual(result.text, "")


if __name__ == "__main__":
    unittest.main()
