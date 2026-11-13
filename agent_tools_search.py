import io
import ipaddress
import math
import os
import re
import sqlite3
import zipfile
from pathlib import Path
from urllib.parse import urlparse, urlencode, urljoin
import pandas as pd
import pdfplumber
import requests
from langchain.tools import tool
from playwright.sync_api import sync_playwright
from tavily import TavilyClient
import time
from bs4 import BeautifulSoup


### Helper functions for tools

# Kg in each measurement unit
KG_PER_UNIT = {
    "kg": 1, "kilogram": 1, "kilograms": 1,
    "t": 1000, "mt": 1000, "tonne": 1000, "tonnes": 1000,
    "1000 t": 1000000, "1000 tonnes": 1000000,
    "thousand tonnes": 1000000, "million tonnes": 1000000000,
    "centner": 100,
    "centners": 100,
    "thsd. centner": 100000,
    "thousand centners": 100000,
    "1000 centners": 100000,
    "1 000 t": 1_000_000,
    "1,000 t": 1_000_000,
    "tsd. t": 1_000_000,
    "kt": 1_000_000,
}



def marketing_year_start(period: str) -> int | None:
    """Extract the first year from a standardized marketing-year period."""
    match = re.match(
        r"(?i)^\s*marketing year\s*:?\s*((?:19|20)\d{2})\s*[/\-–—]\s*((?:19|20)\d{2}|\d{2})(?!\d)",
        str(period).strip()
    )

    if not match:
        return None

    first = int(match.group(1))
    last_text = match.group(2)

    last = int(last_text)

    if len(last_text) == 2:
        last += (first // 100) * 100
        if last < first:
            last += 100

    return first if last == first + 1 else None


# Perform conversion
def convert_mass(value, from_unit, to_unit):
    """Convert known mass units; None means the comparison is unsafe."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    src, dst = str(from_unit).strip().lower(), str(to_unit).strip().lower()
    if src == dst:
        return value
    if src in KG_PER_UNIT and dst in KG_PER_UNIT:
        return value * KG_PER_UNIT[src] / KG_PER_UNIT[dst]
    return None

# Check if two values match
def _same(a, b, unit):
    # SUA export mt quantities to at most two decimals
    if str(unit).strip().lower() == "t":
        return round(float(a), 2) == round(float(b), 2)
    return math.isclose(
        float(a), float(b), rel_tol=1e-9, abs_tol=1e-8
    )

# Check website addresses (in order that the agent accesses public statistical websites)
def _safe_url(url):
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not host:
        raise ValueError("A public HTTP(S) URL is required.")
    if host in ("localhost", "localhost.localdomain") or host.endswith(".local"):
        raise ValueError("Local/private URLs are not allowed.")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        pass  # Ordinary hostname.
    else:
        if not ip.is_global:
            raise ValueError("Local/private URLs are not allowed.")
    return url

# Download PDFs and spreadsheets, for read_pdf() and read_excel()
def _download(url, return_url=False):
    """Download up to 25 MB, validating each redirect destination."""
    for _ in range(5):
        _safe_url(url)
        with requests.get(url, timeout=35, stream=True, allow_redirects=False) as response:
            if response.status_code in (301, 302, 303, 307, 308):
                target = response.headers.get("Location")
                if not target:
                    raise ValueError("Redirect has no destination.")
                url = urljoin(url, target)
                continue
            response.raise_for_status()
            chunks, total = [], 0
            for chunk in response.iter_content(65536):
                total += len(chunk)
                if total > 25 * 1024 * 1024:
                    raise ValueError("Download exceeds 25 MB.")
                chunks.append(chunk)
            data = b"".join(chunks)
            return (data, url) if return_url else data
    raise ValueError("Too many redirects.")

# Identify spreadsheet
def _excel_column(index):
    name = ""
    while index >= 0:
        index, remainder = divmod(index, 26)
        name = chr(65 + remainder) + name
        index -= 1
    return name




### Tools are created inside a function (some tools need information about current investigation, 
# agent does not need to repeatedly pass country code, HS mappings or entire dataset)
# -> supplying different investigation data to the tools without global variables /closure

def make_search_tools(sua_rows, reporter_m49, hs_by_commodity, app_db="data/app.sqlite"):
    """Build exactly six tools for the user's currently selected SUA data.

    `sua_rows` is a list of records with FAOSTAT column names Area, Item,
    Year, Element, Value, Unit (or the lowercase aliases country, commodity,
    year, element, value, unit). `reporter_m49` and `hs_by_commodity` come from
    deterministic preprocessing in agent.py, NOT from the model.
    """
    def field(row, fao_name, alias):
        return row[alias] if alias in row else row[fao_name]

    scope = {}
    for row in sua_rows:
        key = (str(field(row, "Area", "country")),
               str(field(row, "Item", "commodity")),
               int(field(row, "Year", "year")),
               str(field(row, "Element", "element")))
        scope[key] = {
            "value": field(row, "Value", "value"),
            "unit": str(field(row, "Unit", "unit")),
        }
    if not scope:
        raise ValueError("No SUA observations selected.")
    if len({k[0] for k in scope}) != 1:
        raise ValueError("Select one country per investigation.")

    reporter = str(reporter_m49).strip()
    if not reporter.isdigit():
        raise ValueError("Prepare the reporter M49 code before starting the agent.")
    hs_codes = {
        commodity: list(dict.fromkeys(str(code).strip().zfill(6) for code in codes))
        for commodity, codes in hs_by_commodity.items()
    }
    db_path = Path(app_db)

    @tool
    def search_web(query: str, include_domains: list[str] | None = None,
                   deep: bool = False) -> dict:
        """Search for official statistical publications and datasets.

        Use deep=True for detailed statistical searches.
        Use deep=False for quick discovery of government websites.
        Restrict searches to verified official domains when known.
        Search in national languages when useful.
        Search results are leads, not verified numerical evidence.
        """
        try:
            domains = [
                d.lower().strip()
                for d in (include_domains or [])
                if d.strip()
            ]

            params = {
                "query": query,
                "search_depth": "advanced" if deep else "basic",
                "max_results": 6,
                "include_answer": False,
                "timeout": 30
            }

            if domains:
                params["include_domains"] = domains
                params["include_domains_mode"] = "restrict"

            raw = TavilyClient(
                api_key=os.environ["TAVILY_API_KEY"]
            ).search(**params)

            hits = []
            seen = set()

            for hit in raw.get("results", []):
                url = hit.get("url", "")
                host = (urlparse(url).hostname or "").lower()

                if not host or url in seen:
                    continue

                if domains and not any(
                    host == d or host.endswith("." + d)
                    for d in domains
                ):
                    continue

                seen.add(url)

                hits.append({
                    "title": hit.get("title", ""),
                    "url": url,
                    "description": (hit.get("content") or "")[:450]
                })

            return {
                "results": hits,
                "note": (
                    "Search results are discovery leads only. "
                    "Open the original official source to verify figures."
                )
            }

        except Exception as error:
            return {"error": str(error)}

    
    @tool
    def read_webpage(
        url: str,
        actions: list[dict[str, str]] | None = None,
        use_browser: bool = False,
        keyword: str = "",
        start_char: int = 0
    ) -> dict:
        """Read a statistical webpage and discover its documents.

        By default, use fast HTTP extraction.
        Use Playwright when interaction or JavaScript is required.
        Search specific text using keyword and start_char.
        """
        try:
            _safe_url(url)

            # 1. Open webpage
            if use_browser or actions:

                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True)

                    try:
                        page = browser.new_page()
                        page.set_default_timeout(10000)

                        response = page.goto(
                            url,
                            wait_until="domcontentloaded",
                            timeout=25000
                        )

                        if response and response.status >= 400:
                            return {"error": f"HTTP {response.status}"}

                        _safe_url(page.url)

                        for step in actions or []:
                            action = step.get("action")
                            label = step.get("label", "")

                            if action == "fill":
                                page.get_by_label(
                                    label, exact=True
                                ).fill(step["value"])

                            elif action == "select":
                                page.get_by_label(
                                    label, exact=True
                                ).select_option(label=step["value"])

                            elif action == "click":
                                page.get_by_role(
                                    step.get("role", "button"),
                                    name=label,
                                    exact=True
                                ).click()

                            elif action == "wait":
                                page.get_by_text(
                                    label, exact=True
                                ).wait_for(state="visible")

                            else:
                                return {"error": f"Unsupported action: {action}"}

                            _safe_url(page.url)

                        html = page.content()
                        page_url = page.url

                    finally:
                        browser.close()

                method = "browser"

            else:
                # Fast HTTP download, with redirect checks
                html, page_url = _download(url, return_url=True)
                method = "http"

                if html.lstrip().startswith(b"%PDF-"):
                    return {
                        "url": page_url,
                        "error": "This is a PDF. Use read_pdf instead."
                    }

                if html.startswith(b"PK\x03\x04"):
                    return {
                        "url": page_url,
                        "error": "This is an Office/ZIP file. Use read_excel."
                    }

            # 2. Parse HTML
            soup = BeautifulSoup(html, "html.parser")

            # Find all links, resolving relative addresses
            links = []
            seen = set()

            for a in soup.find_all("a", href=True):
                link_url = urljoin(page_url, a["href"])

                if not link_url.startswith(("http://", "https://")):
                    continue

                if link_url in seen:
                    continue

                seen.add(link_url)

                links.append({
                    "title": a.get_text(" ", strip=True)[:150],
                    "url": link_url
                })

            # Put downloadable documents first
            extensions = (
                ".pdf", ".xlsx", ".xls",
                ".csv", ".zip", ".ods"
            )

            links.sort(
                key=lambda link: not (
                    urlparse(link["url"]).path.lower().endswith(extensions)
                    or "download" in link["title"].lower()
                )
            )

            # 3. Extract readable text
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            text = "\n".join(soup.stripped_strings)

            # 4. Find relevant text and support pagination
            start = max(0, start_char)
            found = None

            if keyword:
                found = text.casefold().find(
                    keyword.casefold(), start
                )

                if found >= 0:
                    start = max(0, found - 800)

            end = start + 16000

            result = {
                "url": page_url,
                "method": method,
                "text": text[start:end],
                "total_characters": len(text),
                "next_start_char": (
                    end if end < len(text) else None
                ),
                "links": links[:100]
            }

            if keyword:
                result["keyword_found"] = found >= 0

            if method == "http" and len(text) < 200:
                result["hint"] = (
                    "Little readable HTML was found. "
                    "If this is a JavaScript website, "
                    "retry with use_browser=True."
                )

            return result

        except Exception as error:
            return {
                "error": str(error),
                "url": url,
                "hint": (
                    "If HTTP extraction failed, try "
                    "use_browser=True for an interactive site."
                )
            }


    @tool
    def read_pdf(url: str, keyword: str = "", page: int = 0,  start_page: int = 1) -> dict:
        """Find relevant pages of a PDF or inspect one numbered page (1-based).
        Return extracted text and tables, keeping page numbers for citations.
        For more keyword matches, call again with start_page set to next_page.
        """
        try:
            contents = _download(url)
            if not contents.lstrip().startswith(b"%PDF-"):
                return {"error": "URL did not return a PDF."}
            with pdfplumber.open(io.BytesIO(contents)) as pdf:
                total = len(pdf.pages)
                if page and not 1 <= page <= total:
                    return {"error": f"Page must be between 1 and {total}."}
                page_indices = ([page - 1] if page else range(max(1, start_page) - 1, total))
                results = []
                for index in page_indices:
                    p = pdf.pages[index]
                    text = p.extract_text() or ""
                    if keyword and keyword.casefold() not in text.casefold():
                        continue
                    tables = p.extract_tables()[:3]
                    results.append({"page": index + 1, "text": text[:10000],
                                    "tables": [[row[:14] for row in table[:45]]
                                               for table in tables]})
                    if len(results) >= (1 if page else 3):
                        break
                next_page = None
                if not page and len(results) == 3:
                    if results[-1]["page"] < total:
                        next_page = results[-1]["page"] + 1
                return {
                    "url": url,
                    "total_pages": total,
                    "pages": results,
                    "next_page": next_page,
                    "note": "Scanned PDFs may contain no machine-readable text.",}
        except Exception as error:
            return {"error": str(error), "url": url}

    @tool
    def read_excel(url: str, file: str = "", sheet: str = "", keyword: str = "",
                   start_row: int = 1) -> dict:
        """Inspect CSV/Excel spreadsheets, including those in ZIP files.
        Return sheet names and original row/column coordinates; paginate with start_row.
        """
        try:
            data = _download(url)
            filename = urlparse(url).path.rsplit("/", 1)[-1]
            # Detect ZIP archives even when the URL has no .zip extension.
            # XLSX files are also ZIP-based, so distinguish them.
            is_archive = zipfile.is_zipfile(io.BytesIO(data))

            if is_archive:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    is_archive = "[Content_Types].xml" not in archive.namelist()

            if is_archive:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    names = [name for name in archive.namelist()
                             if name.lower().endswith((".xlsx", ".xls", ".csv"))
                             and not name.startswith("__MACOSX/")]
                    if not file:
                        return {"files_in_zip": names, "instruction": "Select file."}
                    if file not in names:
                        return {"error": "File not in ZIP.", "files_in_zip": names}
                    if archive.getinfo(file).file_size > 25 * 1024 * 1024:
                        return {"error": "Uncompressed file exceeds 25 MB."}
                    data, filename = archive.read(file), file
            start = max(1, int(start_row))

            csv_lines = data.splitlines()

            is_csv = filename.lower().endswith(".csv") or (
                not zipfile.is_zipfile(io.BytesIO(data))
                and not data.startswith(b"\xd0\xcf\x11\xe0")
                and bool(csv_lines)
                and any(sep in csv_lines[0] for sep in (b",", b";", b"\t"))
            )

            if is_csv:
                sheets, selected_sheet = ["CSV"], "CSV"
                df = pd.read_csv(
                    io.BytesIO(data),
                    sep=None,
                    engine="python",
                    header=None,
                    dtype=str,
                    skiprows=start - 1,
                    nrows=2500
                ).fillna("")

            else:
                workbook = pd.ExcelFile(io.BytesIO(data))
                sheets = workbook.sheet_names
                selected_sheet = sheet or sheets[0]
                if selected_sheet not in sheets:
                    return {"error": "Unknown sheet.", "sheets": sheets}
                df = pd.read_excel(workbook, sheet_name=selected_sheet, header=None,
                                   dtype=str, skiprows=start - 1, nrows=2500).fillna("")
            indices = list(range(len(df)))
            if keyword:
                found = [i for i in indices if any(keyword.casefold() in str(v).casefold()
                                                   for v in df.iloc[i])]
                indices = sorted({j for i in found[:20]
                                  for j in range(max(0, i - 2), min(len(df), i + 2))})
            rows = []
            for i in indices[:40]:
                cells = {_excel_column(j): str(v)[:200]
                         for j, v in enumerate(df.iloc[i]) if str(v).strip()}
                rows.append({"row": start + i, "cells": cells})
            return {"url": url, "file": filename, "sheets": sheets,
                    "sheet": selected_sheet, "rows": rows,
                    "scanned_rows": f"{start}-{start + len(df) - 1}"}
        except Exception as error:
            return {"error": str(error), "url": url}

    @tool
    def get_comtrade(commodity: str, year: int, flow: str) -> dict:
        """Get annual UN Comtrade quantity for a selected Import or Export element.
        HS2017 codes and reporter M49 are supplied by Python before this run.
        Return per-code quantities and estimation flags; add totals only for
        complete, unambiguous and convertible results.
        """
        direction = {"m": "M", "import": "M", "imports": "M",
                     "x": "X", "export": "X", "exports": "X"}.get(flow.lower().strip())
        if not direction:
            return {"error": "flow must be Import/M or Export/X."}
        word = "import" if direction == "M" else "export"
        selected = [(key, value) for key, value in scope.items()
                    if key[1] == commodity and key[2] == int(year)
                    and word in key[3].lower() and "value" not in key[3].lower()]
        if len(selected) != 1:
            return {"error": "No unique matching trade quantity in selected SUA scope."}
        codes = hs_codes.get(commodity, [])
        if not codes or any(not re.fullmatch(r"\d{6}", code) for code in codes):
            return {"error": "No valid precomputed HS2017 mapping."}
        secret = os.getenv("UN_COMTRADE_API_KEY", "").strip()
        if not secret:
            return {"error": "UN_COMTRADE_API_KEY is not configured."}

        key, original = selected[0]
        endpoint = "https://comtradeapi.un.org/data/v1/get/C/A/HS"
        params = {
            "reporterCode": reporter,
            "period": str(int(year)),
            "cmdCode": ",".join(codes),
            "flowCode": direction,
            "partnerCode": "0",
            "partner2Code": "0",
            "customsCode": "C00",
            "motCode": "0",
            "includeDesc": "true",
            "breakdownMode": "classic",
            "maxRecords": 250000,
            "subscription-key": secret,
        }
        try:
            for attempt in range(3):
                response = requests.get(
                    endpoint, params=params, timeout=45
                )
                if response.status_code not in (
                    429, 500, 502, 503, 504
                ):
                    break
                if attempt < 2:
                    time.sleep(4 * (attempt + 1))
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data")
            if not isinstance(data, list):
                return {"error": "Comtrade did not return a data list."}
            if len(data) >= 250000:
                return {"error": "Comtrade response may be truncated."}
            components = []
            for code in codes:
                records = [r for r in data
                           if str(r.get("cmdCode", "")).zfill(6) == code
                           and str(r.get("partnerCode", "0")) == "0"
                           and str(r.get("motCode", "0")) == "0"
                           and str(r.get("customsCode", "C00")) == "C00"
                           and str(r.get("partner2Code", "0")) == "0"]
                if len(records) != 1:
                    components.append({"hs2017": code, "status": "missing_or_ambiguous",
                                       "record_count": len(records)})
                    continue
                r = records[0]
                quantity, unit, estimated = r.get("netWgt"), "kg", r.get("isNetWgtEstimated")
                if quantity is None:
                    quantity, unit, estimated = r.get("qty"), r.get("qtyUnitAbbr", ""), r.get("isQtyEstimated")
                converted = convert_mass(quantity, unit, original["unit"])
                components.append({
                    "hs2017": code,
                    "reported_classification": r.get("classificationCode"),
                    "description": r.get("cmdDesc"),
                    "is_original_classification": r.get("isOriginalClassification"),
                    "is_aggregate": r.get("isAggregate"),
                    "status": (
                        "ok" if converted is not None
                        else "incompatible_quantity"
                    ),
                    "source_quantity": quantity,
                    "source_unit": unit,
                    "fao_quantity": converted,
                    "fao_unit": original["unit"],
                    "estimated": estimated,
                    "is_reported": r.get("isReported"),
                })
            complete = all(item["status"] == "ok" for item in components)
            return {"publisher": "UN Comtrade", "commodity": commodity, "year": int(year),
                    "element": key[3], "reporter_m49": reporter, "hs2017": codes,
                    "total_fao_value": sum(item["fao_quantity"] for item in components) if complete else None,
                    "unit": original["unit"], "complete": complete, "components": components,
                    "url": endpoint + "?" + urlencode({k: v for k, v in params.items()
                                                      if k != "subscription-key"}),
                    "note": "Check mapping coverage, estimated quantities, and trade definitions."}
        except requests.RequestException as error:
            # Raw request errors can contain the secret subscription-key in their URL.
            status = getattr(getattr(error, "response", None), "status_code", "unknown")
            return {"error": f"Comtrade network/HTTP error ({status})."}
        except Exception as error:
            return {"error": f"Comtrade response error ({type(error).__name__})."}

    @tool
    def check_existing_observation(country: str, commodity: str, year: int, 
                                   element: str, value: float, unit: str, period: str = "") -> dict:
        """After establishing comparability, check selected SUA and saved findings.
        Return same_as_sua, already_known, new_candidate, outside_scope,
        or incompatible_unit. Never write to the database.
        """
        first_year = marketing_year_start(period)
        effective_year = first_year if first_year is not None else int(year)
        current = scope.get((country, commodity, effective_year, element))
        if current is None:
            return {"status": "outside_scope"}
        converted = convert_mass(value, unit, current["unit"])
        if converted is None:
            return {"status": "incompatible_unit", "fao_unit": current["unit"]}

        original = convert_mass(
            current["value"], current["unit"], current["unit"]
        )

        # A marketing-year observation remains eligible for expert review
        # even when its numeric value matches calendar-year SUA.
        if (
            first_year is None
            and original is not None
            and _same(converted, original, current["unit"])
        ):
            return {"status": "same_as_sua", "sua_value": original}


        if db_path.exists():
            with sqlite3.connect(db_path) as db:
                table = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                                   "AND name='alternatives'").fetchone()
                if table:
                    # Check whether the database has a review_status column
                    columns = {
                        col[1]
                        for col in db.execute("PRAGMA table_info(alternatives)")
                    }

                    # Ignore rejected observations when checking known alternatives
                    status_filter = (
                        " AND (review_status IS NULL OR "
                        "review_status IN ('accepted', 'legacy'))"
                        if "review_status" in columns else ""
                    )

                    # Compare saved findings using the effective SUA year and the original statistical period
                    params = [country, commodity, effective_year, element]

                    period_filter = ""
                    if "period" in columns:
                        period_filter = (
                            " AND LOWER(TRIM(COALESCE(period, ''))) "
                            "= LOWER(TRIM(?))"
                        )
                        params.append(period)

                    rows = db.execute(
                        "SELECT alternative_value, unit FROM alternatives "
                        "WHERE country=? AND commodity=? AND year=? AND element=?"
                        + period_filter
                        + status_filter,
                        params,
                    ).fetchall()

                    for stored_value, stored_unit in rows:
                        known = convert_mass(stored_value, stored_unit, current["unit"])
                        if known is not None and _same(converted, known, current["unit"]):
                            return {"status": "already_known"}
        return {"status": "new_candidate", "fao_value": converted,
                "fao_unit": current["unit"], "sua_value": original}

    return [search_web, read_webpage, read_pdf, read_excel,
            get_comtrade, check_existing_observation]
