import unittest
from unittest.mock import Mock, patch

from site_adapters import adapter_for_url


class SiteAdapterDispatchTest(unittest.TestCase):
    def test_known_hosts_are_dispatched_independently(self):
        cases = {
            "https://supjav.com/a": "supjav.com",
            "https://www.supjav.com/a": "supjav.com",
            "https://missav123.com/a": "missav123.com",
            "https://missav888.com/a": "missav888.com",
            "https://www.missav888.net/a": "missav888.net",
            "https://missav888.org/a": "missav888.org",
            "https://player.vimeo.com/video/1": "vimeo",
        }
        for url, expected in cases.items():
            with self.subTest(url=url):
                self.assertEqual(adapter_for_url(url).name, expected)

    def test_similar_or_malicious_hosts_do_not_match(self):
        for url in (
            "https://missav888.com.example/a",
            "https://notmissav888.com/a",
            "https://supjav.com.example/a",
            "https://example.com/a",
        ):
            with self.subTest(url=url):
                self.assertEqual(adapter_for_url(url).name, "generic")

    def test_dispatch_does_not_rewrite_url(self):
        url = "https://missav888.com/fc2-ppv-4574266"
        adapter = adapter_for_url(url)
        self.assertEqual(adapter.name, "missav888.com")
        self.assertNotEqual(adapter.name, "missav888.net")

    def test_waf_sites_prefer_browseforge(self):
        self.assertTrue(adapter_for_url("https://supjav.com/a").prefer_browseforge)
        self.assertTrue(adapter_for_url("https://missav123.com/a").prefer_browseforge)
        self.assertFalse(adapter_for_url("https://example.com/a").prefer_browseforge)

    def test_supjav_rejects_live_widget_media(self):
        adapter = adapter_for_url("https://supjav.com/145372.html")
        self.assertFalse(adapter.browseforge_accept_candidate(
            "https://media-hls.growcdnssedge.com/live/example.m3u8"
        ))
        self.assertTrue(adapter.browseforge_accept_candidate(
            "https://streamtape.com/get_video?id=target"
        ))

    def test_supjav_exposes_parts_and_sources(self):
        adapter = adapter_for_url("https://supjav.com/206680.html")
        options = adapter.browseforge_options({"parts": [
            {"id": "1", "label": "1", "sources": [{"name": "TV"}, {"name": "ST"}]},
            {"id": "2", "label": "2", "sources": [{"name": "TV"}, {"name": "DS"}]},
        ]})
        self.assertEqual(options, {
            "site": "supjav.com",
            "parts": [
                {"id": "1", "label": "1", "sources": [{"id": "TV", "label": "TV"}, {"id": "ST", "label": "ST"}]},
                {"id": "2", "label": "2", "sources": [{"id": "TV", "label": "TV"}, {"id": "DS", "label": "DS"}]},
            ],
        })

    def test_supjav_exposes_sources_without_numbered_parts(self):
        adapter = adapter_for_url("https://supjav.com/145372.html")
        options = adapter.browseforge_options({
            "servers": [{"name": "TV"}, {"name": "ST"}, {"name": "DS"}],
        })
        self.assertEqual(options["parts"], [{
            "id": "1", "label": "1", "sources": [
                {"id": "TV", "label": "TV"},
                {"id": "ST", "label": "ST"},
                {"id": "DS", "label": "DS"},
            ],
        }])

    def test_supjav_preserves_full_source_names(self):
        adapter = adapter_for_url("https://supjav.com/27066.html")
        options = adapter.browseforge_options({
            "servers": [
                {"name": "SERVER: Streamtape"},
                {"name": "SERVER: Mixdrop"},
                {"name": "SERVER: NinjaStream"},
            ],
        })
        self.assertEqual(
            [source["label"] for source in options["parts"][0]["sources"]],
            ["Streamtape", "Mixdrop", "NinjaStream"],
        )

    @patch("site_adapters.urlopen")
    def test_supjav_prefers_full_stream_tape_name(self, urlopen):
        response = Mock()
        response.read.return_value = b"https://video.example/selected.m3u8"
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        urlopen.return_value = response
        adapter = adapter_for_url("https://supjav.com/27066.html")
        adapter.browseforge_extra_candidates({
            "servers": [
                {"index": 0, "name": "SERVER: Mixdrop", "link": "mixdrop"},
                {"index": 1, "name": "SERVER: Stream Tape", "link": "stream-tape"},
            ],
        }, {"User-Agent": "test"}, 5)
        self.assertIn("c=epat-maerts", urlopen.call_args.args[0].full_url)

    @patch("site_adapters.urlopen")
    def test_supjav_matches_selected_full_name_after_server_prefix(self, urlopen):
        response = Mock()
        response.read.return_value = b"https://video.example/ninja.m3u8"
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        urlopen.return_value = response
        adapter = adapter_for_url("https://supjav.com/27066.html")
        adapter.browseforge_extra_candidates({
            "servers": [
                {"index": 0, "name": "SERVER: Mixdrop", "link": "mixdrop"},
                {"index": 1, "name": "SERVER: NinjaStream", "link": "ninja"},
            ],
        }, {"User-Agent": "test"}, 5, {"part": "1", "source": "NinjaStream"})
        self.assertIn("c=ajnin", urlopen.call_args.args[0].full_url)

    @patch("site_adapters.urlopen")
    def test_supjav_honors_selected_part_and_source(self, urlopen):
        response = Mock()
        response.read.return_value = b"https://video.example/selected.m3u8"
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        urlopen.return_value = response
        adapter = adapter_for_url("https://supjav.com/206680.html")
        candidates = adapter.browseforge_extra_candidates({
            "url": "https://supjav.com/206680.html",
            "parts": [
                {"id": "1", "sources": [{"index": 0, "name": "DS", "link": "part-one"}]},
                {"id": "2", "sources": [{"index": 0, "name": "TV", "link": "wrong"}, {"index": 1, "name": "DS", "link": "part-two"}]},
            ],
        }, {"User-Agent": "test"}, 5, {"part": "2", "source": "DS"})
        self.assertEqual(candidates, ["https://video.example/selected.m3u8"])
        self.assertIn("c=owt-trap", urlopen.call_args.args[0].full_url)

    @patch("site_adapters.urlopen")
    def test_supjav_prefers_st_over_dom_order(self, urlopen):
        response = Mock()
        response.read.return_value = (
            b"?id=target&expires=123&ip=abc&token=decoy "
            b"?id=target&expires=123&ip=abc&token=usable"
        )
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        urlopen.return_value = response
        adapter = adapter_for_url("https://supjav.com/145372.html")
        candidates = adapter.browseforge_extra_candidates({
            "url": "https://supjav.com/145372.html",
            "servers": [
                {"index": 0, "name": "TV", "link": "tv-token"},
                {"index": 1, "name": "ST", "link": "st-token"},
            ],
        }, {"User-Agent": "test"}, 5)
        self.assertEqual(candidates, [
            "https://streamtape.com/get_video?id=target&expires=123&ip=abc&token=usable"
        ])
        self.assertIn("c=nekot-ts", urlopen.call_args.args[0].full_url)

    @patch("site_adapters.urlopen")
    def test_supjav_tries_next_server_when_first_is_missing(self, urlopen):
        response = Mock()
        response.read.return_value = b'https://video.example/target.m3u8'
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        urlopen.side_effect = [OSError("missing"), response]
        adapter = adapter_for_url("https://supjav.com/145372.html")
        candidates = adapter.browseforge_extra_candidates({
            "url": "https://supjav.com/145372.html",
            "servers": [
                {"index": 0, "name": "TV", "link": "first"},
                {"index": 1, "name": "DS", "link": "second"},
            ],
        }, {"User-Agent": "test"}, 5)
        self.assertEqual(candidates, ["https://video.example/target.m3u8"])
        self.assertEqual(urlopen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
