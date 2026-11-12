#Download candidate GEO metadata and processed expression files.
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
REPORT_FILE = REPO_ROOT / "data" / "metadata" / "geo_downloads.json"
BLOCK_SIZE = 1024 * 1024
USER_AGENT = "osimertinib-response-ml/0.1 (public GEO research; Python urllib)"

@dataclass(frozen=True)
class FileSpec:
    accession: str
    filename: str
    kind: str
    approx_mb: float | None = None
    optional: bool = False
    comment: str = ""

    @property
    def url(self) -> str:
        stem = self.accession[:-3] + "nnn"
        root = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stem}/{self.accession}"
        if self.kind == "soft":
            return f"{root}/soft/{self.filename}"
        return f"{root}/suppl/{self.filename}"

    @property
    def destination(self) -> Path:
        return RAW_ROOT / self.accession / self.filename


def candidate_files() -> list[FileSpec]:
    """Manually checked filenames from individual public GEO series pages.

    'RAW.tar' in GEO may contain per-sample processed RESULTS: we download it
    as an archive, but DO NOT assume it contains raw FASTQ files or unpack it.
    """
    expression_files = [
        FileSpec("GSE193258", "GSE193258_RNAseq_estimated_counts.tsv.gz", "processed", 8.0),
        FileSpec("GSE193258", "GSE193258_RNAseq_log2TPM_abundance.tsv.gz", "processed", 8.0),
        FileSpec("GSE255958", "GSE255958_RAW.tar", "results_archive", 50.0,
                 comment="GEO describes TAR of RESULTS; inspect contents after download"),
        FileSpec("GSE254485", "GSE254485_RNAseq_featureCounts_final.txt.gz", "processed", 1.0),
        FileSpec("GSE202859", "GSE202859_TPMs_allsamples.txt.gz", "processed", 3.1),
        FileSpec("GSE335846", "GSE335846_RNA-seq_for_GEO.txt.gz", "processed", 4.4),
        FileSpec("GSE282074", "GSE282074_RAW.tar", "results_archive", 140.0,
                 comment="Mixed study: isolate relevant osimertinib results later"),
        FileSpec("GSE150949", "GSE150949_metaData_with_lineage.txt.gz", "sample_metadata", 1.7),
        FileSpec("GSE150949", "GSE150949_pc9_count_matrix.csv.gz", "processed", 117.4,
                 comment="Single-cell count matrix: NOT independent biological replicates"),
        # Other large matrices are available, but not needed for initial PC9 audit.
        FileSpec("GSE150949", "GSE150949_pooled_watermelon.count.matrix.csv.gz",
                 "processed", 56.0, optional=True),
        FileSpec("GSE150949", "GSE150949_pooled_watermelon.data.matrix.csv.gz",
                 "processed", 229.9, optional=True),
        FileSpec("GSE150949", "GSE150949_pooled_watermelon.metadata.matrix.csv.gz",
                 "sample_metadata", 1.1, optional=True),
    ]
    # SOFT family files preserve the GEO titles and submitter sample annotations.
    series = sorted({item.accession for item in expression_files})
    soft_files = [FileSpec(acc, f"{acc}_family.soft.gz", "soft") for acc in series]
    return soft_files + expression_files

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK_SIZE), b""):
            h.update(block)
    return h.hexdigest()

def load_report() -> dict[str, object]:
    if not REPORT_FILE.is_file():
        return {"files": {}}
    try:
        report = json.loads(REPORT_FILE.read_text(encoding="utf-8"))
        if isinstance(report, dict) and isinstance(report.get("files"), dict):
            return report
    except (ValueError, OSError):
        pass
    print(f"WARNING: invalid {REPORT_FILE}; a fresh report will be created", file=sys.stderr)
    return {"files": {}}

def save_report(report: dict[str, object]) -> None:
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    partial = REPORT_FILE.with_suffix(".json.tmp")
    partial.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    partial.replace(REPORT_FILE)

def _download_once(spec: FileSpec, max_bytes: int) -> int:
    dst = spec.destination
    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_name(dst.name + ".part")
    have = temporary.stat().st_size if temporary.exists() else 0
    if have > max_bytes:
        raise ValueError("existing partial file exceeds safety limit")
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if have:
        headers["Range"] = f"bytes={have}-"
    request = Request(spec.url, headers=headers)
    try:
        response = urlopen(request, timeout=60)
    except HTTPError as exc:
        if exc.code == 416 and have:
            # Server says no remaining range. Restart rather than accepting a possibly bad file.
            temporary.unlink(missing_ok=True)
        raise

    with response:
        status = response.status
        if status not in (200, 206):
            raise ValueError(f"unexpected HTTP status {status}")
        # Some download servers ignore Range; then restart from byte zero.
        append = have > 0 and status == 206
        start_at = have if append else 0
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            expected_total = start_at + int(content_length)
            if expected_total > max_bytes:
                raise ValueError(f"remote file exceeds {max_bytes / 1048576:.0f} MiB safety limit")
        if status == 206:
            content_range = response.headers.get("Content-Range", "")
            match = re.match(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range)
            if not match or int(match.group(1)) != start_at:
                raise ValueError("unexpected Content-Range; refusing unsafe resume")
            if match.group(3) != "*" and int(match.group(3)) > max_bytes:
                raise ValueError("remote file exceeds safety limit")

        mode = "ab" if append else "wb"
        nbytes = start_at
        last_notice = nbytes // (25 * BLOCK_SIZE)
        with temporary.open(mode) as out:
            while True:
                chunk = response.read(BLOCK_SIZE)
                if not chunk:
                    break
                nbytes += len(chunk)
                if nbytes > max_bytes:
                    raise ValueError("stream exceeded safety limit; partial file left in place")
                out.write(chunk)
                if nbytes // (25 * BLOCK_SIZE) > last_notice:
                    last_notice = nbytes // (25 * BLOCK_SIZE)
                    print(f"       {nbytes / BLOCK_SIZE:.0f} MiB received", flush=True)

        if content_length is not None and nbytes != expected_total:
            raise OSError(f"incomplete download ({nbytes:,} of {expected_total:,} bytes)")
        if nbytes == 0:
            raise OSError("server returned an empty file")

    temporary.replace(dst)
    return nbytes


def download(spec: FileSpec, max_bytes: int, report: dict[str, object]) -> bool:
    dst = spec.destination
    key = f"{spec.accession}/{spec.filename}"
    files = report["files"]
    assert isinstance(files, dict)
    if dst.exists():
        # Protect existing data; never overwrite or silently re-download.
        digest = sha256_file(dst)
        print(f"[SKIP] {key}  ({dst.stat().st_size:,} bytes, already present)")
        files[key] = {"status": "existing", "url": spec.url,
                      "path": str(dst.relative_to(REPO_ROOT)),
                      "size_bytes": dst.stat().st_size, "sha256": digest,
                      "checked_utc": datetime.now(timezone.utc).isoformat()}
        save_report(report)
        return True

    for attempt in range(1, 4):
        print(f"[GET {attempt}/3] {key}")
        try:
            nbytes = _download_once(spec, max_bytes)
            digest = sha256_file(dst)
            files[key] = {"status": "downloaded", "url": spec.url,
                          "path": str(dst.relative_to(REPO_ROOT)),
                          "size_bytes": nbytes, "sha256": digest,
                          "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                          "kind": spec.kind, "comment": spec.comment}
            save_report(report)
            print(f"[OK] {key} ({nbytes / BLOCK_SIZE:.1f} MiB) SHA256: {digest[:16]}...")
            return True
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            print(f"[ERROR] {key}: {exc}", file=sys.stderr)
            if attempt < 3:
                time.sleep(attempt * 2)

    files[key] = {"status": "failed", "url": spec.url,
                  "last_attempt_utc": datetime.now(timezone.utc).isoformat()}
    save_report(report)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true",
                        help="actually download; without this flag only print the plan")
    parser.add_argument("--only", action="append", metavar="GSEXXXXXX",
                        help="download only this accession; can repeat")
    parser.add_argument("--all-processed", action="store_true",
                        help="also download large supplementary pooled single-cell matrices")
    parser.add_argument("--max-file-mb", type=int, default=350,
                        help="individual file safety limit in MiB (default 350)")
    args = parser.parse_args()
    if args.max_file_mb <= 0:
        parser.error("--max-file-mb must be positive")

    specs = candidate_files()
    known = {s.accession for s in specs}
    if args.only:
        selected = {x.upper() for x in args.only}
        unknown = selected - known
        if unknown:
            parser.error(f"unknown GEO accession(s): {', '.join(sorted(unknown))}")
        specs = [s for s in specs if s.accession in selected]
    if not args.all_processed:
        specs = [s for s in specs if not s.optional]

    selected_studies = sorted({s.accession for s in specs})
    print(f"GEO studies: {', '.join(selected_studies)}")
    print(f"Files: {len(specs)}; destination: {RAW_ROOT}")
    print("Note: metadata and processed expression data only; no FASTQ/SRA downloads.")
    for spec in specs:
        approx = f"~{spec.approx_mb:g} MB" if spec.approx_mb else "unknown size"
        print(f"  [{spec.kind:15}] {spec.accession}/{spec.filename} ({approx})")
    known_mb = sum(s.approx_mb or 0 for s in specs)
    print(f"Known listed file sizes total ~{known_mb:.0f} MB, PLUS GEO SOFT metadata.")
    if not args.download:
        print("\nPreview only. To download: python scripts/download_geo.py --download")
        return 0

    print("\nDownloading. Existing files are not overwritten; .part files allow resume.")
    report = load_report()
    report["project"] = "osimertinib-response-ml"
    report["description"] = "GEO downloads only; sample labels and eligibility NOT yet verified"
    failures = [s for s in specs if not download(s, args.max_file_mb * BLOCK_SIZE, report)]
    if failures:
        print(f"\nFinished with {len(failures)} failure(s). Re-run to retry:", file=sys.stderr)
        for spec in failures:
            print(f"  - {spec.accession}/{spec.filename}", file=sys.stderr)
        return 1
    print(f"\nAll {len(specs)} files downloaded or already present.")
    print(f"Report: {REPORT_FILE}")
    print("IMPORTANT: downloading does not confirm biological sample-label comparability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
