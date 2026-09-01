import unittest
from unittest.mock import AsyncMock

from attached_chromium import _checkbox_target, page_is_blocked


class AttachedChromiumTest(unittest.IsolatedAsyncioTestCase):
    def test_checkbox_target_uses_text_left_edge(self):
        words = [
            {"key": "verify", "left": 120, "top": 80, "width": 35, "height": 16},
            {"key": "you", "left": 160, "top": 80, "width": 20, "height": 16},
            {"key": "are", "left": 185, "top": 80, "width": 22, "height": 16},
            {"key": "human", "left": 212, "top": 80, "width": 45, "height": 16},
        ]
        self.assertEqual(_checkbox_target(words), (99, 88))

    def test_checkbox_target_ignores_unrelated_text(self):
        self.assertIsNone(_checkbox_target([
            {"key": "welcome", "left": 20, "top": 20, "width": 50, "height": 16}
        ]))

    async def test_page_is_blocked_recognizes_cloudflare(self):
        page = AsyncMock()
        page.evaluate.return_value = {
            "title": "Just a moment...",
            "body": "Performing security verification",
        }
        self.assertTrue(await page_is_blocked(page))

    async def test_page_is_blocked_accepts_regular_page(self):
        page = AsyncMock()
        page.evaluate.return_value = {"title": "Video", "body": "Ready"}
        self.assertFalse(await page_is_blocked(page))


if __name__ == "__main__":
    unittest.main()
