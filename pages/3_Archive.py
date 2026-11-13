import json
import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st
from rag import search_guidelines
from langchain_openai import ChatOpenAI

st.title("Archive")

database = Path("data/app.sqlite")

if not database.exists():
    st.info("No archived reports yet.")
    st.stop()

with sqlite3.connect(database) as conn:
    table = conn.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type='table' AND name='archived_reports'"
    ).fetchone()

    if not table:
        st.info("No archived reports yet.")
        st.stop()

    rows = conn.execute(
        "SELECT id, created_at, report_json "
        "FROM archived_reports ORDER BY id DESC"
    ).fetchall()

if not rows:
    st.info("No archived reports yet.")
    st.stop()

# Choose a previous investigation
labels = []

for report_id, created_at, report_json in rows:
    report = json.loads(report_json)

    labels.append(
        f"#{report_id} | {created_at} UTC | "
        f"{report['country']} | "
        f"{', '.join(report['commodities'])} | "
        f"{report['years'][0]}–{report['years'][1]}"
    )

selected = st.selectbox(
    "Select archived investigation",
    range(len(rows)),
    format_func=lambda i: labels[i]
)

report = json.loads(rows[selected][2])
result = report["result"]
stats = result.get("stats", {})

st.subheader("Investigation details")

st.write("Country:", report["country"])
st.write("Commodities:", ", ".join(report["commodities"]))
st.write("Elements:", ", ".join(report["elements"]))
st.write(
    "Years:",
    f"{report['years'][0]}–{report['years'][1]}"
)

st.subheader("Summary")
st.markdown(result["search_summary"])

st.subheader("Agent Statistics")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Time", f"{result['seconds']:.1f} s")
c2.metric("LLM tokens", f"{result['usage']['total_tokens']:,}")
c3.metric("Model calls", stats.get("model_calls", "N/A"))
c4.metric("Tool calls", stats.get("tool_calls", "N/A"))

st.write(
    "Finalization trigger:",
    stats.get("finish_trigger", "Not recorded")
)

if stats.get("by_tool"):
    with st.expander("Research tool usage"):
        st.dataframe(
            pd.DataFrame([
                {"Tool": name, "Calls": count}
                for name, count in stats["by_tool"].items()
            ]),
            hide_index=True
        )

st.subheader("Findings and expert decisions")

if report["findings"]:
    st.dataframe(
        pd.DataFrame(report["findings"]),
        hide_index=True,
        width="stretch",
        column_config={
            "url": st.column_config.LinkColumn("Source URL"),
            "period": st.column_config.TextColumn("Original period"),
            "comparability": st.column_config.TextColumn(
                "Details", width="large"
            )
        }
    )
else:
    st.info("No new alternative observations were recorded")



# FAO Guidelines

st.divider()
st.subheader("FAO Guidelines Review")

st.caption(
    "Methodological assessment using the FAO FBS/SUA Handbook 2025 "
)

# Use the actual archived report ID so different reports
# have separate assessments and question histories.
report_id = rows[selected][0]

country = report["country"]
commodities = report["commodities"]
elements = report["elements"]

context = (
    f"Country: {country}\n"
    f"Commodities: {', '.join(commodities)}\n"
    f"Elements: {', '.join(elements)}\n"
    f"Years: {report['years']}\n\n"
    f"Investigation summary:\n"
    f"{result.get('search_summary', '')[:4500]}\n\n"
    f"Findings and expert decisions:\n"
    f"{json.dumps(report.get('findings', [])[:6], ensure_ascii=False)[:3500]}"
)


def archived_guideline_answer(question):
    passages = []

    # Search both documents independently.
    sources = [
        (
            "FBS handbook.pdf",
            question + " " + " ".join(elements)
        ),
        (
            "COMMODITY TREES & Conversion factors.pdf",
            country + " " + " ".join(commodities) + " " + question
        ),
    ]

    for source, query in sources:
        hits = search_guidelines(
            query,
            category="validation",
            source=source,
            n_results=3
        )

        for excerpt in (hits.get("documents") or [[]])[0]:
            passages.append((source, excerpt[:1400]))

    if not passages:
        return "No relevant guideline passages were retrieved.", []

    evidence = "\n\n".join(
        f"[{i}] Source: {source}\n{excerpt}"
        for i, (source, excerpt) in enumerate(passages, start=1)
    )

    model = ChatOpenAI(
        model="gpt-6-luna",
        use_responses_api=True,
        max_completion_tokens=1800,
        timeout=90,
        max_retries=1
    )

    response = model.invoke([
        (
            "system",
            "You are an FAO SUA/FBS methodological assistant. "
            "Use only the investigation information and the retrieved "
            "FAO passages provided below. "
            "Cite supporting passages using [1], [2], etc. "
            "Do not invent PDF page numbers or numerical factors. "
            "Ignore irrelevant passages. "
            "Distinguish historical conversion factors from "
            "current statistical observations. "
            "Explain comparability concerns without automatically "
            "accepting or rejecting findings. "
            "If evidence is insufficient, say so."
        ),
        (
            "human",
            f"ARCHIVED INVESTIGATION:\n{context}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"FAO GUIDELINE PASSAGES:\n{evidence}"
        )
    ])

    content = response.content

    if isinstance(content, str):
        answer = content
    else:
        answer = "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )

    return answer or "No textual answer was returned.", passages


def show_archived_guideline_sources(passages):
    for i, (source, excerpt) in enumerate(passages, start=1):
        st.markdown(f"**[{i}] {source}**")
        st.caption(excerpt)


# A. Methodological assessment
analysis_key = f"archive_fao_analysis_{report_id}"

if st.button(
    "Analyze archived report using FAO guidelines",
    key=f"archive_fao_analyze_{report_id}"
):
    try:
        with st.spinner("Analyzing FAO methodological guidance..."):
            question = (
                "Provide up to three concise methodological "
                "considerations relevant to this investigation. "
                "Focus on the selected SUA elements, source comparability, "
                "reporting periods and relevant commodity conversion factors. "
                "Only discuss conversion factors when relevant. "
                "Identify limitations and issues for expert attention."
            )

            st.session_state[analysis_key] = (
                archived_guideline_answer(question)
            )

    except Exception as exc:
        st.error(f"FAO assessment failed: {exc}")

if analysis_key in st.session_state:
    answer, passages = st.session_state[analysis_key]

    st.markdown(answer)

    with st.expander("Sources supporting the assessment"):
        show_archived_guideline_sources(passages)


# B. Questions and answers
st.markdown("#### Ask the FAO Guidelines")

chat_key = f"archive_fao_chat_{report_id}"

with st.form(key=f"archive_fao_form_{report_id}"):
    question = st.text_input(
        "Ask about this investigation or FAO methodology"
    )

    ask = st.form_submit_button("Ask question")

if ask and question.strip():
    try:
        with st.spinner("Searching the FAO guidelines..."):
            answer, passages = archived_guideline_answer(
                question.strip()
            )

        st.session_state.setdefault(chat_key, []).append(
            (question.strip(), answer, passages)
        )

    except Exception as exc:
        st.error(f"Could not answer the question: {exc}")

for i, (asked, answer, passages) in enumerate(
    st.session_state.get(chat_key, []),
    start=1
):
    st.markdown(f"**Question {i}: {asked}**")
    st.markdown(answer)

    with st.expander(f"Sources for question {i}"):
        show_archived_guideline_sources(passages)
