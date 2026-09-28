"""
`real_kernel.messages_to_append`: which messages of an `llm_chat` call are
new relative to what the agent's server-side context already holds.

Pure function, no server needed. The real-server half (the context really
receiving the system prompt and the model's replies) is
`test_real_kernel_client.py::test_llm_chat_records_system_prompt_and_replies_in_the_context`.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vectros_sdk.client.real_kernel import _message_key, messages_to_append  # noqa: E402

SYSTEM = {"role": "system", "content": "Be brief."}
USER_1 = {"role": "user", "content": "My name is Ravi."}
REPLY_1 = {"role": "assistant", "content": "Hello Ravi."}
USER_2 = {"role": "user", "content": "What is my name?"}


def held(*messages):
    return [_message_key(m) for m in messages]


class TestMessagesToAppend(unittest.TestCase):
    def test_first_call_appends_everything_including_the_system_prompt(self) -> None:
        self.assertEqual(messages_to_append([], [SYSTEM, USER_1]), [SYSTEM, USER_1])

    def test_full_growing_history_appends_only_the_new_turn(self) -> None:
        history = held(SYSTEM, USER_1, REPLY_1)
        self.assertEqual(messages_to_append(history, [SYSTEM, USER_1, REPLY_1, USER_2]), [USER_2])

    def test_reply_resent_with_whitespace_stripped_is_not_duplicated(self) -> None:
        history = [*held(SYSTEM, USER_1), ("assistant", "Hello Ravi.")]
        resent = {"role": "assistant", "content": "\n Hello Ravi. \n"}
        self.assertEqual(messages_to_append(history, [SYSTEM, USER_1, resent, USER_2]), [USER_2])

    def test_system_plus_newest_message_appends_only_the_newest(self) -> None:
        history = held(SYSTEM, USER_1, REPLY_1)
        self.assertEqual(messages_to_append(history, [SYSTEM, USER_2]), [USER_2])

    def test_newest_message_alone_is_appended(self) -> None:
        history = held(SYSTEM, USER_1, REPLY_1)
        self.assertEqual(messages_to_append(history, [USER_2]), [USER_2])

    def test_growing_history_after_an_earlier_conversation_appends_only_the_new_turn(self) -> None:
        earlier = held({"role": "user", "content": "Say PASS"}, {"role": "assistant", "content": "PASS"})
        history = earlier + held(SYSTEM, USER_1, REPLY_1)
        self.assertEqual(messages_to_append(history, [SYSTEM, USER_1, REPLY_1, USER_2]), [USER_2])

    def test_new_conversation_after_an_earlier_one_is_appended_in_full(self) -> None:
        history = held(SYSTEM, USER_1, REPLY_1)
        other = {"role": "system", "content": "You are a translator."}
        self.assertEqual(messages_to_append(history, [other, USER_2]), [other, USER_2])

    def test_role_is_part_of_the_match(self) -> None:
        history = held(SYSTEM)
        as_user = {"role": "user", "content": SYSTEM["content"]}
        self.assertEqual(messages_to_append(history, [as_user]), [as_user])

    def test_identical_history_appends_nothing(self) -> None:
        history = held(SYSTEM, USER_1, REPLY_1)
        self.assertEqual(messages_to_append(history, [SYSTEM, USER_1, REPLY_1]), [])


if __name__ == "__main__":
    unittest.main()
