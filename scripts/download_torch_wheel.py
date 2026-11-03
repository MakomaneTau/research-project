"""Fetch the exact official CUDA wheel with resumable verified range requests.

The complete SHA-256 must match the official package index before installation.
This is useful on connections where a single large transfer is unusually slow.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import time
import urllib.parse
import urllib.request

BASE = "https://download.pytorch.org/whl/cu128/torch/"
FILENAME = "torch-2.11.0+cu128-cp311-cp311-win_amd64.whl"
ROOT = Path(__file__).resolve().parents[1] / ".build" / "wheels"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(value for key, value in attrs if key == "href")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    parser = Links()
    with urllib.request.urlopen(BASE, timeout=60) as response:
        parser.feed(response.read().decode())
    found = [link for link in parser.links if urllib.parse.unquote(urllib.parse.urlparse(link).path).endswith("/" + FILENAME)]
    if len(found) != 1:
        raise RuntimeError("Exact official wheel was not uniquely identified")
    url, fragment = urllib.parse.urldefrag(urllib.parse.urljoin(BASE, found[0]))
    expected = urllib.parse.parse_qs(fragment)["sha256"][0]
    if urllib.parse.urlparse(url).hostname not in {"download.pytorch.org", "download-r2.pytorch.org"}:
        raise RuntimeError("Unexpected wheel download host")
    candidates = [url.replace("https://download-r2.pytorch.org", "https://download.pytorch.org"), url]
    for candidate in dict.fromkeys(candidates):
        try:
            with urllib.request.urlopen(urllib.request.Request(candidate, headers={"Range": "bytes=0-0"}), timeout=60) as response:
                if response.status != 206:
                    raise RuntimeError("Server does not support verified byte ranges")
                size = int(response.headers["Content-Range"].split("/")[-1])
                url = candidate
                break
        except Exception:
            if candidate == candidates[-1]:
                raise
    chunk = 32 * 1024 * 1024
    parts = [(i, start, min(size, start + chunk) - 1) for i, start in enumerate(range(0, size, chunk))]

    def fetch(spec):
        index, start, stop = spec
        path = ROOT / f"{FILENAME}.part{index:03}"
        if path.exists() and path.stat().st_size == stop - start + 1:
            return index
        for attempt in range(8):
            try:
                completed = path.stat().st_size if path.exists() else 0
                if completed > stop - start + 1:
                    raise RuntimeError("Existing partial range exceeds its expected size")
                offset = start + completed
                request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{stop}"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206 or response.headers["Content-Range"] != f"bytes {offset}-{stop}/{size}":
                        raise RuntimeError("Server returned an unexpected range")
                    with path.open("ab") as output:
                        while data := response.read(1024 * 1024):
                            output.write(data)
                if path.stat().st_size != stop - start + 1:
                    raise RuntimeError("Incomplete range")
                return index
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(min(2 ** attempt, 30))

    print(f"Official wheel: {size / 1e9:.2f} GB in {len(parts)} ranges", flush=True)
    with ThreadPoolExecutor(max_workers=12) as executor:
        for completed, future in enumerate(as_completed([executor.submit(fetch, part) for part in parts]), 1):
            future.result()
            print(f"Downloaded ranges {completed}/{len(parts)}", flush=True)
    digest = hashlib.sha256()
    target = ROOT / FILENAME
    temporary = target.with_suffix(".assembling")
    with temporary.open("wb") as output:
        for index, _, _ in parts:
            with (ROOT / f"{FILENAME}.part{index:03}").open("rb") as source:
                while data := source.read(1024 * 1024):
                    digest.update(data)
                    output.write(data)
    if digest.hexdigest() != expected:
        raise RuntimeError("Official SHA-256 mismatch; wheel will not be installed")
    temporary.replace(target)
    target.with_suffix(".sha256").write_text(expected + "\n", encoding="ascii")
    print(f"Verified {target}", flush=True)


if __name__ == "__main__":
    main()
