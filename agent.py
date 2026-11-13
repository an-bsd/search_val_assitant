import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    wrap_model_call,
    wrap_tool_call)
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field
from typing import Literal
from agent_tools_search import (
    make_search_tools,
    convert_mass,
    marketing_year_start
)

load_dotenv()

MAX_MINUTES = 2
MAX_LLM_TOKENS = 250_000
MEMORY_FILE = Path("data/guidelines_experience.json")
MAPPING_DIR = Path("guidelines/search")


class BudgetReached(Exception):
    """Investigation reached its configured time or token limit"""


class Observation(BaseModel):
    """New, comparable, source-supported SUA alternative."""
    country: str
    commodity: str
    year: int
    element: str
    source_value: float = Field(allow_inf_nan=False)
    fao_value: float = Field(allow_inf_nan=False)
    source_unit: str
    publisher: str
    source_type: Literal["national_official", "semi_official"]
    url: str
    period: str = Field(
        description=(
            "Original statistical period. Explicitly label marketing years, "
            "for example 'Marketing year 2020/2021'."
        )
    )
    evidence: str = Field(description="Original table, PDF page or spreadsheet cell and its heading")
    comparability: str = Field(
        description=(
            "Details of commodity, element, units, coverage, reference period, "
            "and any marketing-year first-year mapping or limitation."
        )
    )


class SearchOutput(BaseModel):
    observations: list[Observation] = Field(default_factory=list)
    search_summary: str = Field(description="Brief coverage, results and unresolved limitations")
    updated_experience: str = Field(max_length=1800,
        description="Only NEW reusable search lessons not already in memory. Return an empty string if nothing new.")


SEARCH_PROMPT = """You are ONE autonomous FAOSTAT SUA alternative-data research agent.

OBJECTIVE: Find as many NEW, independently published, comparable official and
reliable semi-official numerical observations as possible for the EXACT supplied
SUA country, commodities, years and elements. Do not investigate other SUA rows.
You receive the selected SUA data and PRECOMPUTED HS2017 mappings from Python.
Choose your own web queries and tools; no fixed query templates or extra agents.

SOURCE POLICY:
1. Use original national official sources of the selected country:
   national statistical offices, ministries, customs authorities,
   government agricultural agencies, and other official national
   institutions publishing their own statistics.
2. Official government publications, statistical databases,
   yearbooks, PDF reports, Excel/CSV datasets and statistical APIs
   are acceptable sources.
3. ALSO USE reliable semi-official institutions (recognized sector associations,
   international organizations such as USDA, Eurostat, FAO/GIEWS, ICAC).
4. NEVER use newspapers, press/news sites, blogs, commercial news aggregators,
   marketing sites, unattributed reprints or unverified publishers as evidence.
   Look up the ORIGINAL institutional publisher, not a copied figure.
5. Verify the original publisher and the official website.
   Do not assume a website is official merely because its
   domain name looks governmental.
6. Search results and snippets are discovery leads only.
   Every candidate figure must be checked in its original
   official publication or statistical dataset.


SEARCH STRATEGY:

1. UNDERSTAND THE INVESTIGATION
- Investigate only the selected country, commodities, years
  and SUA elements.
- The selected SUA observations and HS mappings have already
  been prepared by Python.
- Do not request unnecessary data or investigate elements
  outside the selection.

2. DISCOVER OFFICIAL NATIONAL DATA SOURCES
- First identify the country's official statistical agencies
  and relevant agricultural statistical publications.
- Search for national commodity balances, supply-utilization
  accounts, food balance sheets, agricultural yearbooks,
  crop statistics and official statistical databases.
- Search using both English and the country's national language.
- Prefer sources covering multiple requested elements and years
  rather than searching separately for every observation.
- Search for national datasets using local-language equivalents
  of crop balances, food balances, feed use, seed use, losses,
  opening stocks, industrial use and production.
- Prefer official numerical datasets over general annual reports.
- Treat wheat grain and wheat flour as different commodities.
  Do not use grain production as evidence for flour production.

3. SEARCH EFFICIENTLY
- Use search_web with deep=False for quick discovery.
- Use deep=True for difficult or specific publication searches.
- Restrict searches to verified official domains when known.
- Vary unsuccessful queries using national-language terminology,
  publication titles, commodity synonyms and statistical concepts.
- Do not repeat searches that have already failed.
- After finding promising official search results, open and
  examine the original publication before issuing more searches.
- Do not repeatedly search for the same commodity and element
  using minor variations of the same query.
- Prefer extracting numerical tables from a promising source
  over collecting additional search snippets.
- When a statistical dataset is available, use read_excel
  or read_pdf to extract relevant figures.
- If two searches produce no useful sources, change the
  search strategy rather than repeating similar queries.

4. EXAMINE ORIGINAL PUBLICATIONS
- Open ordinary webpages with read_webpage.
- If JavaScript or interactive controls are required,
  retry with use_browser=True.
- Inspect the available document and download links.
- Prefer original Excel, CSV and ZIP statistical datasets
  when they contain the requested information.
- Use read_excel for spreadsheet data and read_pdf for PDFs.
- For long webpages, use keyword and start_char to inspect
  relevant sections.
- Extract all relevant observations from a useful publication
  before moving to another source.

5. RESEARCH MISSING ELEMENTS
- After examining comprehensive statistical publications,
  identify which selected SUA elements remain unresolved.
- Search for these missing elements individually using
  appropriate official national sources.
- Production: national crop and livestock production statistics.
- Imports and exports: national customs and trade statistics.
- Stocks: national commodity balances and inventory statistics.
- Food, feed, seed, losses and processing: national supply
  balances, agricultural utilization reports and surveys.
- Do not assume every element is directly published.

6. USE COMTRADE SELECTIVELY
- Use get_comtrade only when import or export quantity
  is among the selected SUA elements and supplementary
  comparison would be useful.
- Do not prioritize Comtrade over original national statistics.
- Check HS classification, coverage, estimation flags and units.
- Do not submit Comtrade values as national official observations.

7. VALIDATE AND CONTINUE
- Verify each candidate against the selected SUA observation.
- Check commodity definition, statistical period, unit,
  coverage and independence from the SUA source.
- Compare values at their published precision.
- Call check_existing_observation before proposing a candidate.
- If a CALENDAR-YEAR figure matches SUA, it is not a new candidate.
- If a MARKETING-YEAR figure matches SUA, it may still be a
  candidate because its reference period is different.
- If an observation is already saved, treat it as previously known.
- Always pass the original period to check_existing_observation.
- Never invent values, citations, conversions or missing data.

8. REPORT COVERAGE
- Distinguish new alternative observations, verified SUA matches,
  inaccessible sources and figures not found.
- Identify useful publications and original URLs.
- Explain important comparability limitations.
- Return the structured SearchOutput within the available budget.

EVIDENCE AND COMPARABILITY (REQUIRED FOR EVERY CANDIDATE):
- Verify country, exact FAO commodity and product definition, SUA element,
  statistical year/period, measurement unit, coverage and source independence.
- Do NOT conflate wheat grain with flour, crop production with harvested area.
- Record the reported source value and unit, normalized FAO value, original
  publisher/URL and an identifiable table/page/cell with heading or footnote.
- Use verified numerical conversions; if a unit or methodology is incompatible,
  explain the issue in search_summary, do not invent an alternative.
- AFTER verifying comparability, call check_existing_observation.
  Output only figures whose status is new_candidate.
- A calendar-year figure matching SUA is not a new candidate.
  A marketing-year figure may be a new candidate even when
  numerically equal to SUA, because its reporting period differs.
- A figure already recorded for the same source period and
  numerical value is previously known.

GLOBAL REFERENCE-PERIOD RULE:

- Research statistical publications in any country and language.
- Understand the original source's reporting-period terminology.
- Identify whether an observation refers to a calendar year,
  marketing year, crop year or another statistical period.
- When the original source explicitly uses a marketing year,
  normalize the period field to English:
  "Marketing year 2020/2021"
  "Marketing year 2021/2022"
  "Marketing year 2022/2023"
- This normalization applies worldwide, regardless of the
  original language or country.
- Assign each marketing-year observation to the FIRST year:
  2020/2021 -> 2020
  2021/2022 -> 2021
  2022/2023 -> 2022
- Do not reject an otherwise valid candidate solely because
  its source reports a marketing year rather than a calendar year.
- In the comparability (Details) field, preserve the original
  source period and explain the year mapping:
  "Original reporting period: [original source wording].
  Marketing year 2020/2021 assigned to SUA year 2020
  using the first-year convention. The source period
  differs from a calendar year."
- Do not treat an arbitrary two-year date range as a
  marketing year. Confirm the period type from the source.
- Do not silently reclassify fiscal years or other
  reporting periods as marketing years.
- Preserve differences in commodity definitions, coverage
  and methodology for expert review.

RESEARCH COMPLETENESS:
- Before finishing, check coverage of every selected
  commodity, year and element.
- Do not stop simply because one publication matches SUA
  or previously saved alternatives.
- For unresolved observations, search at least one additional
  promising official dataset or publication when feasible.
- When an official dataset portal is found, inspect its
  View Data, Archive and Download options before concluding
  that figures are unavailable.
- Distinguish clearly between figures verified as matching
  SUA, previously known figures, incompatible figures,
  and figures not found.
- In search_summary, provide a short coverage assessment
  for every selected commodity and element, including
  official source URLs and unresolved years.
- If a relevant official source exists but could not be
  extracted or its reference period is incompatible,
  explain that specifically rather than saying not found.
- Continue researching useful unresolved observations
  until coverage is adequate or the budget requires
  a final response.
- Do not finish merely because initial searches produced no
  new alternatives.
- Continue investigating unresolved selected observations
  using other promising official national sources.
- Finish early only if all selected observations have been
  adequately investigated and no promising official sources
  remain, or if the research budget requires completion.

FINAL STRUCTURED RESPONSE:
- observations: only new, evidenced, comparable figures in the
  chosen SUA scope. These are CANDIDATES awaiting expert review.
  Do not claim they have been accepted or rejected.
- search_summary: Write a clear, structured research report
  using Markdown formatting.
  Use short paragraphs or bullet points.
  Include original source URLs and table/page references.
  Omit empty sections.
  Do not describe candidate observations as approved or rejected.
- updated_experience: suggest up to TWO specific, reusable
  lessons learned during this investigation.
  Consider successful and unsuccessful research:
  - Official statistical portals or datasets worth searching again.
  - Effective national-language search terminology.
  - Document download or extraction methods that worked.
  - Specific access or extraction failures and useful workarounds.
  - Important commodity or statistical-period interpretation pitfalls.
  A lesson does NOT require finding a new alternative observation.
  However, every lesson must be supported by something actually
  encountered during this investigation.
  Compare suggestions with APPROVED SEARCH EXPERIENCE.
  Do not repeat existing instructions or previously approved lessons.
  Avoid generic advice such as "search official sources".
  If no genuinely new, actionable lesson exists, return an empty string.
  Never save suggestions automatically.
  Expert approval remains mandatory.
Continue autonomously; produce the final structured response before budget ends.
"""


def load_search_experience(path=MEMORY_FILE):
    """Read expert-approved memory once, outside the tool loop."""
    path = Path(path)
    if not path.exists():
        return ""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return str(data.get("search_experience") or "")


def save_approved_search_experience(text, path=MEMORY_FILE):
    """Call only from the expert-confirmation button on the report page."""
    text = text.strip()
    if len(text) > 1800:
        raise ValueError("Search Experience must be 1,800 characters or fewer.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    settings["search_experience"] = text
    path.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")


def _digits(value):
    """Normalize a FAOSTAT code, retaining significant leading zeros."""
    text = str(value).strip().lstrip("'")
    if re.fullmatch(r"\d+(?:\.0)?", text):
        return text.split(".", 1)[0]
    return ""


def prepare_search_inputs(selected_sua_data, mapping_dir=MAPPING_DIR):
    """Prepare the user selection and HS codes.
    Streamlit page passes the already-filtered SUA dataframe. FAOSTAT Area
    Code (M49) is preferred. Optional pycountry resolves it when absent.
    HS mapping is by exact FCL code or FAO commodity name, then exact CPC code.
    If no reliable mapping is found, that commodity simply has no Comtrade tool
    availability. The agent can still investigate other official sources.
    """
    df = pd.DataFrame(selected_sua_data).copy()
    required = {"Area", "Item", "Year", "Element", "Value", "Unit"}
    if df.empty or not required.issubset(df.columns):
        raise ValueError(f"Selected SUA records must contain {sorted(required)}")
    if df["Area"].nunique() != 1:
        raise ValueError("Select one country per investigation.")
    rows = [{key: (None if pd.isna(value) else value) for key, value in record.items()}
            for record in df.to_dict("records")]

    reporter = ""
    for column in ("Area Code (M49)", "Area Code (M49) "):
        if column in df.columns:
            values = {_digits(x) for x in df[column] if _digits(x)}
            if len(values) == 1:
                reporter = values.pop().zfill(3)
            break
    if not reporter:
        try:
            import pycountry
            reporter = pycountry.countries.lookup(str(df["Area"].iloc[0])).numeric
        except (ImportError, LookupError, AttributeError) as exc:
            raise ValueError("M49 code missing: keep the FAOSTAT 'Area Code (M49)' column.") from exc

    workbooks = sorted(Path(mapping_dir).glob("*CPC*FCL*HS*.xlsx"))
    mapping = {}
    if workbooks:
        reference = pd.read_excel(workbooks[0], sheet_name="Mapping master file_FAO Agric",
                                  header=3, dtype=str).fillna("")
        names = reference["FCL Description"].str.strip().str.casefold()
        fcl_codes = reference["106"].map(_digits).str.zfill(4)
        hs_values = reference["HS 2017"].map(_digits)
        for commodity in df["Item"].astype(str).unique():
            selected = df[df["Item"].astype(str) == commodity]
            candidates = pd.Series(False, index=reference.index)
            if "Item Code" in selected.columns:
                fcl = {_digits(x).zfill(4) for x in selected["Item Code"] if _digits(x)}
                if len(fcl) == 1:
                    candidates = fcl_codes.isin(fcl)
            if not candidates.any():
                candidates = names.eq(commodity.strip().casefold())
            if not candidates.any() and "Item Code (CPC)" in selected.columns:
                cpc = {_digits(x) for x in selected["Item Code (CPC)"] if _digits(x)}
                if len(cpc) == 1:
                    code = next(iter(cpc))
                    if len(code) >= 4:
                        candidates = reference[["4-digit", "5-digit", "7-digit"]].apply(
                            lambda col: col.astype(str).str.startswith(code)
                        ).any(axis=1)
            codes = hs_values[candidates & hs_values.str.fullmatch(r"\d{6}")]
            if not codes.empty:
                mapping[commodity] = list(dict.fromkeys(codes.tolist()))

    return rows, reporter, mapping


def make_agent(sua_rows, reporter_m49, hs_by_commodity,
               max_tokens=MAX_LLM_TOKENS,
               max_minutes=MAX_MINUTES,
               app_db="data/app.sqlite", experience=""):

    tools = make_search_tools(
        sua_rows, reporter_m49, hs_by_commodity, app_db
    )

    stats = {
        "model_calls": 0,
        "tool_calls": 0,
        "by_tool": {tool.name: 0 for tool in tools},
        "finish_trigger": "Not triggered",
        "budget_minutes": max_minutes,
        "budget_tokens": max_tokens,
    }

    started = time.monotonic()
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    calls = 0

    @wrap_model_call
    def guard_model(request, handler):
        nonlocal calls
        calls += 1
        stats["model_calls"] = calls
        elapsed = time.monotonic() - started

        # Calculate if the budget is nearly exhausted
        finish_now = (
            usage["total_tokens"] + 8_000 >= max_tokens
            or elapsed >= max_minutes * 60 * 0.9
        )

        # Record the reason
        if finish_now and stats["finish_trigger"] == "Not triggered":
            if usage["total_tokens"] + 8_000 >= max_tokens:
                stats["finish_trigger"] = "Token budget"
            else:
                stats["finish_trigger"] = "Time budget"


        if finish_now:
            current = request.system_message.content

            request = request.override(
                # Research tools are removed.
                # LangChain still supplies the SearchOutput response tool.
                tools=[],
                system_message=SystemMessage(content=(
                    str(current)
                    + "\nResearch is finished. Return SearchOutput now, "
                    "using only evidence already obtained. Include "
                    "verified new observations, explain unfinished "
                    "coverage in search_summary, and do not invent "
                    "missing values."
                )),
            )

        result = handler(request)

        for message in result.result:
            metadata = getattr(message, "usage_metadata", None) or {}

            for key in usage:
                usage[key] += int(metadata.get(key, 0) or 0)

        return result


    @wrap_tool_call
    def count_tool_calls(request, handler):
        name = request.tool_call["name"]

        if name in stats["by_tool"]:
            stats["tool_calls"] += 1
            stats["by_tool"][name] += 1

        return handler(request)


    model = ChatOpenAI(
        model="gpt-6-luna",
        use_responses_api=True,
        max_completion_tokens=5000,
        timeout=90,
        max_retries=0,
    )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=(
            SEARCH_PROMPT
            + "\nAPPROVED SEARCH EXPERIENCE:\n"
            + (experience or "None.")
        ),
        response_format=ToolStrategy(SearchOutput),
        middleware=[
            guard_model,
            count_tool_calls,
            ModelCallLimitMiddleware(run_limit=100),
            ToolCallLimitMiddleware(run_limit=200),
        ],
    )

    return agent, usage, tools, stats


def run_search(selected_sua_data, mapping_dir=MAPPING_DIR, app_db="data/app.sqlite",
               max_tokens=MAX_LLM_TOKENS, max_minutes=MAX_MINUTES):
    """One call for Streamlit: prepare inputs, investigate, validate output.
    Never modifies the SUA database, alternatives database, or experience file.
    The report page will handle candidate and memory expert approval.
    """
    rows, m49, mapping = prepare_search_inputs(selected_sua_data, mapping_dir)
    experience = load_search_experience()
    agent, usage, tools, stats = make_agent(rows, m49, mapping, max_tokens, max_minutes, app_db, experience)
    selected = pd.DataFrame(rows)
    columns = ["Item", "Year", "Element", "Value", "Unit"]
    if "Flag" in selected.columns:
        columns.append("Flag")
    question = (
        f"Country: {rows[0]['Area']}\n\n"
        "Selected SUA observations (CSV):\n"
        + selected[columns].to_csv(index=False)
        + "\nPrecomputed HS2017 mappings:\n"
        + json.dumps(mapping, ensure_ascii=False)
        + "\nResearch the supplied observations where feasible. "
          "Return structured new alternatives."
    )
    started = time.monotonic()
    try:
        state = agent.invoke({"messages": [{"role": "user", "content": question}]},
                             config={"recursion_limit": 200})
        output = state.get("structured_response")
        if output is None:
            raise RuntimeError("Agent returned no structured_response.")
        # Run deterministic duplicate/unit/scope checks AFTER the agent finishes.
        checker = next(tool for tool in tools if tool.name == "check_existing_observation")
        scope = {(r["Area"], r["Item"], int(r["Year"]), r["Element"]): r for r in rows}
        new_rows, seen = [], set()

        for obs in output.observations:

            first_year = marketing_year_start(obs.period)

            if first_year is not None:
                details = obs.comparability.strip()

                if "first-year convention" not in details.casefold():
                    details += (
                        f" Source period: {obs.period}. "
                        f"Assigned to SUA year {first_year} using the "
                        "first-year convention; the reference period "
                        "differs from the calendar year and requires "
                        "expert review."
                    )

                obs = obs.model_copy(update={
                    "year": first_year,
                    "comparability": details.strip()
                })

            key = (obs.country, obs.commodity, obs.year, obs.element)

            target = scope.get(key)
            if not target or not obs.evidence.strip() or not obs.publisher.strip():
                continue
            if not obs.period.strip() or not obs.comparability.strip():
                continue
            try:
                url = urlparse(obs.url)
            except ValueError:
                continue
            if url.scheme not in ("http", "https") or not url.hostname:
                continue
            converted = convert_mass(obs.source_value, obs.source_unit, target["Unit"])
            if converted is None or abs(converted - obs.fao_value) > max(1e-7, abs(converted)*1e-8):
                continue
            status = checker.invoke({
                "country": obs.country,
                "commodity": obs.commodity,
                "year": obs.year,
                "element": obs.element,
                "value": obs.source_value,
                "unit": obs.source_unit,
                "period": obs.period
            })
            dedup_key = (
                *key,
                round(converted, 8),
                obs.period.strip().casefold()
            )
            if status.get("status") == "new_candidate" and dedup_key not in seen:
                seen.add(dedup_key)
                new_rows.append(obs.model_dump())
        excluded = len(output.observations) - len(new_rows)
        summary = output.search_summary.strip() + f"\nNew candidate rows after checks: {len(new_rows)}."
        if excluded:
            summary += f" {excluded} proposed rows excluded as duplicate, out of scope, or insufficiently comparable."
        return {"observations": new_rows, "search_summary": summary,
                "updated_experience": output.updated_experience,
                "usage": usage, "seconds": round(time.monotonic() - started, 1),
                "stats": stats, "status": "complete",}
    except (BudgetReached, GraphRecursionError) as exc:
        return {"observations": [], "search_summary": f"Search stopped: {exc}. No finalized findings available.",
                "updated_experience": "", "usage": usage,
                "seconds": round(time.monotonic() - started, 1), "stats": stats, "status": "incomplete"}
