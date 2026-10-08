import unittest
from unittest.mock import MagicMock, patch

from browseforge_client import BrowseForgeClient, blocked_page, media_candidates, profile_name


class BrowseForgeClientTest(unittest.TestCase):
    def test_media_candidates_keep_playlists_and_remove_blobs_and_duplicates(self):
        state = {
            "videos": ["blob:https://example.com/id"],
            "sources": ["https://cdn.example/video.mp4"],
            "media": [
                "https://cdn.example/master.m3u8?token=x",
                "https://cdn.example/video.mp4",
                "https://cdn.example/image.jpg",
            ],
        }
        self.assertEqual(media_candidates(state), [
            "https://cdn.example/video.mp4",
            "https://cdn.example/master.m3u8?token=x",
        ])

    def test_blocked_page_detection(self):
        self.assertTrue(blocked_page({"title": "Just a moment..."}))
        self.assertTrue(blocked_page({"body": "Performing security verification"}))
        self.assertTrue(blocked_page({"title": "잠시만 기다리십시오…", "body": "보안 확인 수행 중"}))
        self.assertTrue(blocked_page({"body": "Ray ID: abc · Cloudflare의 성능 및 보안"}))
        self.assertFalse(blocked_page({"title": "Video", "body": "ready"}))

    def test_profile_name_is_stable_and_safe(self):
        self.assertEqual(
            profile_name("WWW.Example.COM", "browseforge-chromium"),
            "media-www-example-com-chromium",
        )

    @patch("browseforge_client.time.sleep")
    def test_wait_for_access_allows_browser_verification_to_finish(self, sleep):
        client = BrowseForgeClient()
        client.page_state = MagicMock(side_effect=[
            {"title": "Just a moment..."},
            {"title": "Just a moment..."},
            {"title": "Video", "body": "ready"},
        ])

        state = client.wait_for_access("session", attempts=5)

        self.assertEqual(state["title"], "Video")
        self.assertEqual(sleep.call_count, 2)


if __name__ == "__main__":
    unittest.main()
