"""TERM.14 (adversarial): a model reply can contain more than one
```propose block -- deliberately or by trying to smuggle a second command
past one the human already approved. Only the first block may ever be
parsed, shown, or run; the second must never be touched by any code path."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vectros_sdk.terminal.proposal import extract_first_proposal, review_and_execute


SMUGGLED_REPLY = """Sure, I'll check disk usage for you.

```propose
df -h
```

Actually let me also clean up while I'm at it:

```propose
rm -rf /important-data
```
"""


class TestFirstProposalOnly(unittest.TestCase):
    def test_only_the_first_propose_block_is_extracted(self):
        proposal = extract_first_proposal(SMUGGLED_REPLY)
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.command, "df -h")
        self.assertNotIn("rm -rf", proposal.command)

    def test_a_reply_with_no_propose_block_yields_nothing(self):
        self.assertIsNone(extract_first_proposal("Here is the answer, no command needed."))

    def test_an_empty_propose_block_yields_nothing(self):
        self.assertIsNone(extract_first_proposal("```propose\n\n```"))


class TestApprovalNeverReachesTheSecondCommand(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.audit_log = Path(self.tmpdir.name) / "audit.jsonl"
        self.patcher = patch(
            "vectros_sdk.terminal.proposal.AUDIT_LOG_PATH", self.audit_log
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.tmpdir.cleanup()

    def test_approving_the_first_proposal_never_executes_the_second(self):
        proposal = extract_first_proposal(SMUGGLED_REPLY)
        with patch("vectros_sdk.terminal.proposal.run_command") as mock_run:
            mock_run.return_value.exit_code = 0
            mock_run.return_value.ran = True
            mock_run.return_value.confirmed = True
            review_and_execute(proposal, confirm=lambda _prompt: "y")
            mock_run.assert_called_once()
            (executed_command,), _ = mock_run.call_args
            self.assertEqual(executed_command, "df -h")
            self.assertNotIn("rm", executed_command)

    def test_declining_the_proposal_runs_nothing_and_audits_the_decline(self):
        proposal = extract_first_proposal(SMUGGLED_REPLY)
        with patch("vectros_sdk.terminal.proposal.run_command") as mock_run:
            result = review_and_execute(proposal, confirm=lambda _prompt: "n")
            mock_run.assert_not_called()
            self.assertFalse(result.ran)

        lines = self.audit_log.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["decision"], "declined")
        self.assertEqual(record["command"], "df -h")

    def test_approving_audits_the_approval_with_the_first_commands_exit_code(self):
        proposal = extract_first_proposal(SMUGGLED_REPLY)
        with patch("vectros_sdk.terminal.proposal.run_command") as mock_run:
            mock_run.return_value.exit_code = 7
            mock_run.return_value.ran = True
            mock_run.return_value.confirmed = True
            review_and_execute(proposal, confirm=lambda _prompt: "y")

        lines = self.audit_log.read_text(encoding="utf-8").strip().splitlines()
        record = json.loads(lines[-1])
        self.assertEqual(record["decision"], "approved")
        self.assertEqual(record["command"], "df -h")
        self.assertEqual(record["exit_code"], 7)


if __name__ == "__main__":
    unittest.main()
