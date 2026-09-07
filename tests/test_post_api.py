import unittest
from unittest.mock import MagicMock, patch

from vectros_sdk import (
    AIOSKernelError,
    PostQuery,
    PostResponse,
    broadcast_post,
    publish_to_topic,
    receive_posts,
    send_post,
    subscribe_topic,
)


class TestPostModels(unittest.TestCase):
    """Test PostQuery and PostResponse data models."""

    def test_post_query_defaults(self):
        query = PostQuery(
            agent_name="agent_alice",
            recipient="agent_bob",
            message="Hello Bob!",
        )
        self.assertEqual(query.query_class, "post")
        self.assertEqual(query.agent_name, "agent_alice")
        self.assertEqual(query.action_type, "send")
        self.assertEqual(query.recipient, "agent_bob")
        self.assertEqual(query.message, "Hello Bob!")
        self.assertTrue(query.mark_as_read)

        # Dict subscripting
        self.assertEqual(query["agent_name"], "agent_alice")
        self.assertEqual(query["message"], "Hello Bob!")

    def test_post_response_defaults(self):
        resp = PostResponse(
            response_message="Message delivered",
            message_id="msg_123",
        )
        self.assertEqual(resp.response_class, "post")
        self.assertEqual(resp.response_message, "Message delivered")
        self.assertEqual(resp.message_id, "msg_123")
        self.assertTrue(resp.finished)
        self.assertIsNone(resp.error)
        self.assertEqual(resp.status_code, 200)

        # Dual access
        self.assertEqual(resp["message_id"], "msg_123")
        self.assertEqual(resp["response"]["message_id"], "msg_123")


class TestPostAPIFunctions(unittest.TestCase):
    """Test all 5 Agent-to-Agent Communication functions."""

    @patch("vectros_sdk.post.api.send_request")
    def test_send_post(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "response_class": "post",
                "response_message": "Message sent",
                "message_id": "msg_001",
                "finished": True,
                "status_code": 200,
            }
        }

        resp = send_post(
            sender="alice",
            recipient="bob",
            message={"action": "collaborate", "topic": "research"},
            metadata={"priority": "high"},
            base_url="http://custom-kernel:9000",
        )

        self.assertIsInstance(resp, PostResponse)
        self.assertEqual(resp.message_id, "msg_001")
        self.assertEqual(resp["response"]["response_message"], "Message sent")

        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.agent_name, "alice")
        self.assertEqual(called_query.action_type, "send")
        self.assertEqual(called_query.recipient, "bob")
        self.assertEqual(called_query.message, {"action": "collaborate", "topic": "research"})
        self.assertEqual(called_query.metadata, {"priority": "high"})
        self.assertEqual(mock_send_request.call_args[1]["base_url"], "http://custom-kernel:9000")

    @patch("vectros_sdk.post.api.send_request")
    def test_receive_posts(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "response_class": "post",
                "messages": [
                    {"message_id": "msg_1", "sender": "bob", "content": "Hi Alice"},
                    {"message_id": "msg_2", "sender": "charlie", "content": "Meeting in 5"},
                ],
                "finished": True,
            }
        }

        resp = receive_posts(agent_name="alice", limit=5, mark_as_read=True)

        self.assertIsInstance(resp, PostResponse)
        self.assertEqual(len(resp.messages), 2)
        self.assertEqual(resp.messages[0]["sender"], "bob")

        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.agent_name, "alice")
        self.assertEqual(called_query.action_type, "receive")
        self.assertEqual(called_query.limit, 5)
        self.assertTrue(called_query.mark_as_read)

    @patch("vectros_sdk.post.api.send_request")
    def test_broadcast_post(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "response_class": "post",
                "response_message": "Broadcast sent to all agents",
                "finished": True,
            }
        }

        resp = broadcast_post(sender="alice", message="System reboot at 2 AM", topic="announcements")

        self.assertIsInstance(resp, PostResponse)
        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.agent_name, "alice")
        self.assertEqual(called_query.action_type, "broadcast")
        self.assertEqual(called_query.topic, "announcements")
        self.assertEqual(called_query.message, "System reboot at 2 AM")

    @patch("vectros_sdk.post.api.send_request")
    def test_publish_to_topic(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "response_class": "post",
                "response_message": "Published to topic news",
                "finished": True,
            }
        }

        resp = publish_to_topic(sender="sensor_bot", topic="news", message={"temp": 24.5})

        self.assertIsInstance(resp, PostResponse)
        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.action_type, "publish")
        self.assertEqual(called_query.topic, "news")
        self.assertEqual(called_query.message, {"temp": 24.5})

    @patch("vectros_sdk.post.api.send_request")
    def test_subscribe_topic(self, mock_send_request):
        mock_send_request.return_value = {
            "response": {
                "response_class": "post",
                "response_message": "Subscribed to news",
                "finished": True,
            }
        }

        resp = subscribe_topic(agent_name="listener_bot", topic="news")

        self.assertIsInstance(resp, PostResponse)
        called_query = mock_send_request.call_args[0][0]
        self.assertEqual(called_query.agent_name, "listener_bot")
        self.assertEqual(called_query.action_type, "subscribe")
        self.assertEqual(called_query.topic, "news")

    @patch("vectros_sdk.post.api.send_request")
    def test_post_api_error_propagation(self, mock_send_request):
        mock_send_request.side_effect = AIOSKernelError("Post broker unreachable", status_code=503)

        with self.assertRaises(AIOSKernelError) as ctx:
            send_post(sender="alice", recipient="bob", message="Hello")
        self.assertEqual(ctx.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()

