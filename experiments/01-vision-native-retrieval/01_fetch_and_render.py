"""01_fetch_and_render.py -- download 3 PMC open-access PDFs and render pages to PNGs.

Strategy (per verified INGEST stream):
  1. NCBI OA API (oa.fcgi?id=PMCxxxx) is the OA/license GATE only -- a <record>
     element means the article is open-access. We do NOT try to fetch the tgz
     href from it (that href is ftp:// and blocked; constructing an https tgz
     path 404s).
  2. Download the real PDF over HTTPS from Europe PMC's render endpoint
     https://europepmc.org/articles/PMCxxxx?pdf=render (returns application/pdf,
     redirects to /api/getPdf). Verified live 200 for all 3 IDs on 2026-07-03.
  3. Render each page with PyMuPDF 1.28.0 at 150 dpi, RGB, to PNG.

Output layout:
  ./pdfs/<pmcid>.pdf
  ./pages/<pmcid>/<pmcid>_p000.png, _p001.png, ...
"""
import os
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET

import fitz  # PyMuPDF 1.28.0 ; `import pymupdf` also works

PMCIDS = ["PMC11153305", "PMC10867699", "PMC11271409"]
WORKDIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(WORKDIR, "pdfs")
PAGES_DIR = os.path.join(WORKDIR, "pages")
OA_API = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
UA = {"User-Agent": "vision-retrieval-poc/1.0 (mailto:segalleon@hotmail.com)"}
DPI = 150


def confirm_oa(pmcid):
    """Return (is_oa: bool, license_or_reason: str). Raises on transport error."""
    url = f"{OA_API}?{urllib.parse.urlencode({'id': pmcid})}"
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            root = ET.fromstring(r.read())
    except urllib.error.URLError as e:
        raise RuntimeError(f"{pmcid}: OA API unreachable: {e}") from e
    err = root.find(".//error")
    if err is not None:
        return False, (err.get("code") or err.text or "error")
    rec = root.find(".//record")
    if rec is None:
        return False, "no <record> (not open-access)"
    return True, (rec.get("license") or "unknown-license")


def download_pdf(pmcid, dest):
    """Download a real PDF from Europe PMC render endpoint. Raises on any failure."""
    url = f"https://europepmc.org/articles/{pmcid}?pdf=render"
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{pmcid}: HTTP {e.code} fetching PDF from {url}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"{pmcid}: network error fetching PDF: {e}") from e
    if not data.startswith(b"%PDF"):
        raise RuntimeError(
            f"{pmcid}: response is not a PDF (first bytes {data[:16]!r}). "
            f"Europe PMC render may be down; retry later."
        )
    with open(dest, "wb") as f:
        f.write(data)
    return len(data)


def render_pdf(pdf_path, out_dir, pmcid, dpi=DPI):
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    try:
        paths = []
        for i, page in enumerate(doc):
            # dpi= sets zoom (>=1.19.2); csRGB -> 3-channel RGB (drops alpha)
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
            p = os.path.join(out_dir, f"{pmcid}_p{i:03d}.png")
            pix.save(p)  # PNG chosen by .png extension
            paths.append(p)
    finally:
        doc.close()
    return paths


def main():
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(PAGES_DIR, exist_ok=True)
    total_pages = 0
    failures = []
    for pmcid in PMCIDS:
        print(f"[{pmcid}] checking open-access status...")
        try:
            ok, lic = confirm_oa(pmcid)
        except RuntimeError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            failures.append(pmcid)
            continue
        print(f"  OA={ok} license={lic}")
        if not ok:
            print(f"  SKIP: {pmcid} is not open-access ({lic}).", file=sys.stderr)
            failures.append(pmcid)
            continue
        pdf_path = os.path.join(PDF_DIR, f"{pmcid}.pdf")
        try:
            n = download_pdf(pmcid, pdf_path)
            print(f"  downloaded {n:,} bytes -> {pdf_path}")
        except RuntimeError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            failures.append(pmcid)
            continue
        out_dir = os.path.join(PAGES_DIR, pmcid)
        imgs = render_pdf(pdf_path, out_dir, pmcid, dpi=DPI)
        total_pages += len(imgs)
        print(f"  rendered {len(imgs)} page(s) @{DPI}dpi RGB -> {out_dir}")
        time.sleep(1)  # be polite
    print(f"\nDONE: {total_pages} page image(s) total under {PAGES_DIR}")
    if failures:
        print(f"FAILED for: {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
