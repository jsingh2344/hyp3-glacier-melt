#!/usr/bin/env python3
"""
Download OPERA-S1 RTC files from ASF/Vertex for one OPERA burst ID and date range.

Typical use for your cube workflow:

    python download_opera_burst_asf.py \
      --opera-burst-id T050-105620-IW1 \
      --start 2017-01-01 \
      --end 2024-12-31 \
      --output-dir /home/ubuntu/glacier-inputs/opera \
      --polarization VH \
      --asset-mode all \
      --processes 4

This will:
  1. Search ASF for OPERA-S1 RTC products matching the OPERA burst ID/date range.
  2. Download the selected product URLs using asf_search.
  3. Write:
       - download_manifest.csv: all selected/downloaded URLs
       - search_results.geojson: ASF search result metadata
       - file_list.txt: quoted local paths usable by generate_opera_cube.py.
         The file_list.txt includes selected-polarization GeoTIFFs. The cube workflow
         obtains its Copernicus GLO-30 DEM separately through hyp3lib unless a local
         DEM is explicitly supplied.

Authentication:
  Preferred: use a ~/.netrc file with Earthdata credentials.
  Or pass --username/--password.
  Or set EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables.

Notes:
  - OPERA burst IDs look like: T078-165486-IW2 or T078_165486_IW2.
  - ASF accepts the operaBurstID keyword for OPERA-S1 products.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import asf_search as asf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download OPERA-S1 RTC products from ASF for one OPERA burst ID and date range."
    )

    parser.add_argument(
        "--opera-burst-id",
        required=True,
        help="OPERA burst ID, e.g. T050-105620-IW1 or T050_105620_IW1.",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start acquisition date/time. Example: 2017-01-01 or 2017-01-01T00:00:00Z.",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End acquisition date/time. Example: 2024-12-31 or 2024-12-31T23:59:59Z.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where downloaded files and manifests will be written.",
    )

    parser.add_argument(
        "--processing-level",
        default="RTC",
        choices=["RTC", "CSLC", "RTC_STATIC", "CSLS_STATIC"],
        help="OPERA-S1 processing level to search. Default: RTC.",
    )
    parser.add_argument(
        "--polarization",
        default="VH",
        choices=["VV", "VH", "HH", "HV"],
        help="Polarization GeoTIFF to include in file_list.txt, and to prefer in cube mode. Default: VH.",
    )
    parser.add_argument(
        "--asset-mode",
        default="all",
        choices=["all", "main", "geotiff", "cube"],
        help=(
            "Which URLs to download. "
            "'main' downloads only primary product URLs, usually .h5. "
            "'geotiff' downloads all additional .tif/.tiff URLs. "
            "'cube' downloads only selected-polarization GeoTIFFs. "
            "'all' downloads primary product URLs plus all additional URLs. Default: all."
        ),
    )
    parser.add_argument(
        "--asset-regex",
        default=None,
        help=(
            "Optional regex applied to filenames after asset-mode selection. "
            "Example: '(_VH\\.tif$|dem\\.tif$)'."
        ),
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=2000,
        help="Maximum ASF search results. Default: 2000.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Parallel download processes passed to asf_search.download_urls. Default: 1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files even if a local file with the same name already exists.",
    )

    parser.add_argument(
        "--username",
        default=os.environ.get("EARTHDATA_USERNAME"),
        help="Earthdata Login username. Defaults to EARTHDATA_USERNAME env var if set.",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("EARTHDATA_PASSWORD"),
        help="Earthdata Login password. Defaults to EARTHDATA_PASSWORD env var if set.",
    )
    parser.add_argument(
        "--edl-token",
        default=os.environ.get("EARTHDATA_TOKEN"),
        help="Earthdata bearer token. Defaults to EARTHDATA_TOKEN env var if set.",
    )

    return parser.parse_args()


def normalize_date(value: str, *, is_end: bool) -> str:
    """
    ASF accepts several date formats, but this makes simple YYYY-MM-DD args explicit UTC.
    """
    value = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return f"{value}T23:59:59Z" if is_end else f"{value}T00:00:00Z"
    return value


def normalize_opera_burst_id(value: str) -> str:
    """
    ASF examples show both T078-165486-IW2 and T078_165486_IW2 forms.
    Keep the user-provided ID, but normalize whitespace.
    """
    return value.strip()


def filename_from_url(url: str) -> str:
    path = urlsplit(url).path
    return Path(unquote(path)).name


def get_properties(product: Any) -> dict[str, Any]:
    props = getattr(product, "properties", None)
    if props is None:
        return {}
    return dict(props)


def product_id(props: dict[str, Any]) -> str:
    for key in ("fileID", "sceneName", "granuleName", "productName", "displayName"):
        val = props.get(key)
        if val:
            return str(val)
    return ""


def main_url_from_product(product: Any, props: dict[str, Any]) -> str | None:
    # Most asf_search results expose the downloadable URL in properties["url"].
    url = props.get("url")
    if url:
        return str(url)

    # Defensive fallbacks for different asf_search/ASFProduct versions.
    for attr in ("url", "downloadUrl"):
        val = getattr(product, attr, None)
        if val:
            return str(val)

    return None


def additional_urls_from_props(props: dict[str, Any]) -> list[str]:
    urls = props.get("additionalUrls") or props.get("additional_urls") or []

    if isinstance(urls, str):
        # Usually this is already a list, but handle comma/newline-separated strings.
        parts = re.split(r"[\n,]+", urls)
        return [p.strip() for p in parts if p.strip()]

    if isinstance(urls, (list, tuple)):
        return [str(u) for u in urls if u]

    return []


def is_geotiff_url(url: str) -> bool:
    name = filename_from_url(url).lower()
    return name.endswith(".tif") or name.endswith(".tiff")


def is_pol_geotiff_url(url: str, pol: str) -> bool:
    name = filename_from_url(url)
    pol = pol.upper()
    return bool(re.search(rf"_{pol}\.tiff?$", name, flags=re.IGNORECASE))


def select_urls_for_product(
    product: Any,
    asset_mode: str,
    polarization: str,
    asset_regex: str | None,
) -> list[tuple[str, str]]:
    """
    Return list of (url, source_kind) tuples.
    source_kind is one of: main, additional.
    """
    props = get_properties(product)
    main_url = main_url_from_product(product, props)
    additional_urls = additional_urls_from_props(props)

    selected: list[tuple[str, str]] = []

    if main_url:
        if asset_mode in ("main", "all"):
            selected.append((main_url, "main"))

        elif asset_mode == "cube":
            if is_pol_geotiff_url(main_url, polarization):
                selected.append((main_url, "main"))

    if asset_mode in ("geotiff", "cube", "all"):
        for url in additional_urls:
            include = False

            if asset_mode == "all":
                include = True
            elif asset_mode == "geotiff":
                include = is_geotiff_url(url)
            elif asset_mode == "cube":
                include = is_pol_geotiff_url(url, polarization)

            if include:
                selected.append((url, "additional"))

    if asset_regex:
        rx = re.compile(asset_regex)
        selected = [(u, k) for (u, k) in selected if rx.search(filename_from_url(u))]

    # Preserve order but drop duplicates.
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for url, kind in selected:
        if url not in seen:
            seen.add(url)
            unique.append((url, kind))

    return unique


def make_session(args: argparse.Namespace):
    """
    Return an authenticated ASFSession if explicit credentials/token were provided.
    If not, return None and let asf_search use ~/.netrc/default auth behavior.
    """
    if args.username and args.password:
        return asf.ASFSession().auth_with_creds(args.username, args.password)

    if args.edl_token:
        return asf.ASFSession().auth_with_token(args.edl_token)

    return None


def search_opera_products(args: argparse.Namespace):
    start = normalize_date(args.start, is_end=False)
    end = normalize_date(args.end, is_end=True)
    burst_id = normalize_opera_burst_id(args.opera_burst_id)

    print("Searching ASF...")
    print("  dataset          = OPERA-S1")
    print(f"  processingLevel  = {args.processing_level}")
    print(f"  operaBurstID     = {burst_id}")
    print(f"  start            = {start}")
    print(f"  end              = {end}")
    print(f"  maxResults       = {args.max_results}")

    results = asf.search(
        dataset="OPERA-S1",
        processingLevel=args.processing_level,
        operaBurstID=burst_id,
        start=start,
        end=end,
        maxResults=args.max_results,
    )

    print(f"Found {len(results)} ASF result(s).")
    return results


def local_path_for_url(output_dir: Path, url: str) -> Path:
    return output_dir / filename_from_url(url)


def write_geojson(results: Any, path: Path) -> None:
    """
    asf_search result objects usually provide geojson(); fall back to per-product .geojson.
    """
    try:
        geojson = results.geojson()
    except Exception:
        features = []
        for product in results:
            try:
                features.append(product.geojson())
            except Exception:
                pass
        geojson = {"type": "FeatureCollection", "features": features}

    path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    fieldnames = [
        "product_id",
        "source_kind",
        "filename",
        "url",
        "local_path",
        "already_existed",
        "selected_for_cube_file_list",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_cube_file_list(rows: list[dict[str, str]], path: Path) -> None:
    cube_paths = [
        row["local_path"]
        for row in rows
        if row["selected_for_cube_file_list"].lower() == "true"
    ]

    cube_paths = sorted(cube_paths, key=lambda p: Path(p).name)

    with path.open("w", encoding="utf-8") as f:
        for p in cube_paths:
            f.write(f'"{p}"\n')


def run_download_opera_burst(args) -> int:

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = search_opera_products(args)

    if len(results) == 0:
        print("No matching ASF results found.", file=sys.stderr)
        return 2

    write_geojson(results, args.output_dir / "search_results.geojson")

    url_rows: list[dict[str, str]] = []
    urls_to_download: list[str] = []

    for product in results:
        props = get_properties(product)
        pid = product_id(props)

        selected_urls = select_urls_for_product(
            product=product,
            asset_mode=args.asset_mode,
            polarization=args.polarization,
            asset_regex=args.asset_regex,
        )

        for url, source_kind in selected_urls:
            local_path = local_path_for_url(args.output_dir, url)
            already_existed = local_path.exists() and local_path.stat().st_size > 0

            selected_for_cube = is_pol_geotiff_url(url, args.polarization)

            url_rows.append(
                {
                    "product_id": pid,
                    "source_kind": source_kind,
                    "filename": local_path.name,
                    "url": url,
                    "local_path": str(local_path),
                    "already_existed": str(already_existed),
                    "selected_for_cube_file_list": str(selected_for_cube),
                }
            )

            if args.overwrite or not already_existed:
                urls_to_download.append(url)

    if not url_rows:
        print(
            "Search succeeded, but no downloadable URLs matched your selection. "
            "Try --asset-mode all or remove --asset-regex.",
            file=sys.stderr,
        )
        return 3

    # Deduplicate download URLs while preserving order.
    urls_to_download = list(dict.fromkeys(urls_to_download))

    manifest_path = args.output_dir / "download_manifest.csv"
    file_list_path = args.output_dir / "file_list.txt"

    write_manifest(url_rows, manifest_path)
    write_cube_file_list(url_rows, file_list_path)

    print(f"Selected {len(url_rows)} URL row(s).")
    print(f"Need to download {len(urls_to_download)} file(s); already-present files are skipped unless --overwrite is set.")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote cube file list: {file_list_path}")

    if not urls_to_download:
        print("Nothing to download.")
        return 0

    session = make_session(args)

    print("Downloading with asf_search...")
    if session is None:
        asf.download_urls(
            urls=urls_to_download,
            path=str(args.output_dir),
            processes=args.processes,
        )
    else:
        asf.download_urls(
            urls=urls_to_download,
            path=str(args.output_dir),
            session=session,
            processes=args.processes,
        )

    print("Done.")
    return 0

def main() -> int:
    args = parse_args()
    return run_download_opera_burst(args)

if __name__ == "__main__":
    raise SystemExit(main())
