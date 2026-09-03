# Third-party notices

This container installs Ubuntu's FFmpeg packages and their codec libraries.
The Go HTTP wrapper in this directory does not change the licenses of those
components.

- Ubuntu package copyright files are installed in `/usr/share/doc/*/copyright`
  inside the image.
- Ubuntu package source search: https://packages.ubuntu.com/source/noble/ffmpeg
- FFmpeg legal and license information:
  https://ffmpeg.org/legal.html
- FFmpeg source:
  https://ffmpeg.org/download.html#get-sources

When redistributing the built image, retain the installed copyright files and
comply with the exact Ubuntu FFmpeg build configuration's LGPL/GPL source
requirements.

The container also includes the official yt-dlp standalone executable and Deno.

- yt-dlp source and license information: https://github.com/yt-dlp/yt-dlp
- yt-dlp bundled third-party licenses are embedded in the official executable.
- Deno source and MIT license: https://github.com/denoland/deno

The SSH service uses Go's `golang.org/x/crypto/ssh` package under the BSD
3-Clause license. Source and license: https://pkg.go.dev/golang.org/x/crypto

The Collector image installs Debian Chromium and uses `chromedp` to control it.

- Debian package copyright files remain in `/usr/share/doc/*/copyright`.
- Chromium source and license information: https://www.chromium.org/Home/
- chromedp source and BSD 3-Clause license: https://github.com/chromedp/chromedp
