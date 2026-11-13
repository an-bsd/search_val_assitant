import sqlite3
import pandas as pd
import streamlit as st
from agent import load_search_experience, save_approved_search_experience
from pathlib import Path
import re
from difflib import SequenceMatcher
import json
from rag import search_guidelines
from langchain_openai import ChatOpenAI

def save_reviewed_findings(rows):
    """Save findings, evidence and expert review decisions."""
    column_types = {
        "country": "TEXT",
        "commodity": "TEXT",
        "year": "INTEGER",
        "element": "TEXT",
        "sua_value": "REAL",
        "alternative_value": "REAL",
        "unit": "TEXT",
        "publisher": "TEXT",
        "url": "TEXT",
        "source_value": "REAL",
        "source_unit": "TEXT",
        "source_type": "TEXT",
        "period": "TEXT",
        "evidence": "TEXT",
        "comparability": "TEXT",
        "review_status": "TEXT DEFAULT 'legacy'",
        "review_note": "TEXT",
    }

    with sqlite3.connect("data/app.sqlite") as conn:
        definitions = ", ".join(
            f"{name} {data_type}"
            for name, data_type in column_types.items()
        )

        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS alternatives (
                {definitions},
                    UNIQUE (
                        country, commodity, year, element,
                        alternative_value, unit, period
                    )
            )
        """)

        # Extend the older table without deleting existing records.
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(alternatives)")
        }

        for name, data_type in column_types.items():
            if name not in existing:
                conn.execute(
                    f"ALTER TABLE alternatives ADD COLUMN {name} {data_type}"
                )

        columns = list(column_types)
        placeholders = ", ".join(["?"] * len(columns))


        saved = 0

        for row in rows.to_dict("records"):

            # Check whether this observation is already stored
            existing = conn.execute(
                """
                SELECT rowid FROM alternatives
                WHERE country = ?
                  AND commodity = ?
                  AND year = ?
                  AND element = ?
                  AND alternative_value = ?
                  AND unit = ?
                  AND COALESCE(period, '') = COALESCE(?, '')
                LIMIT 1
                """,
                (
                    row["country"],
                    row["commodity"],
                    row["year"],
                    row["element"],
                    row["alternative_value"],
                    row["unit"],
                    row["period"]
                )
            ).fetchone()

            if existing:
                # Update the expert's decision
                conn.execute(
                    """
                    UPDATE alternatives
                    SET review_status = ?, review_note = ?
                    WHERE rowid = ?
                    """,
                    (
                        row["review_status"],
                        row["review_note"],
                        existing[0]
                    )
                )

            else:
                # Save a new observation
                conn.execute(
                    f"INSERT INTO alternatives ({', '.join(columns)}) "
                    f"VALUES ({placeholders})",
                    tuple(row[column] for column in columns)
                )

            saved += 1

        return saved

    


st.title("Investigation report")

result = st.session_state.get("search_result")

if result is None:
    st.info("Run an investigation from the Start page first.")
    st.stop()

run_id = st.session_state["investigation_id"]
review_key = f"saved_review_{run_id}"

if result["status"] != "complete":
    st.warning(
        "The investigation stopped before producing a finalized result."
    )

st.subheader("Previously saved findings")

history_key = f"previous_findings_{run_id}"

if history_key not in st.session_state:
    previous = pd.DataFrame()
    database_path = Path("data/app.sqlite")

    if database_path.exists():
        with sqlite3.connect(database_path) as conn:
            table_exists = conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='alternatives'"
            ).fetchone()

            if table_exists:
                selection = st.session_state["investigation_data"]

                commodities = selection["Item"].unique().tolist()
                elements = selection["Element"].unique().tolist()
                years = selection["Year"].astype(int)

                commodity_placeholders = ",".join(["?"] * len(commodities))
                element_placeholders = ",".join(["?"] * len(elements))

                previous = pd.read_sql_query(
                    f"""
                    SELECT * FROM alternatives
                    WHERE country = ?
                      AND commodity IN ({commodity_placeholders})
                      AND element IN ({element_placeholders})
                      AND year BETWEEN ? AND ?
                    ORDER BY commodity, year, element
                    """,
                    conn,
                    params=(
                        selection["Area"].iloc[0],
                        *commodities,
                        *elements,
                        int(years.min()),
                        int(years.max())
                    )
                )


    # Older records may not have a review-status column.
    if not previous.empty:
        if "review_status" not in previous.columns:
            previous["review_status"] = "legacy"
        else:
            previous["review_status"] = (
                previous["review_status"].fillna("legacy")
            )

    # Preserve the history shown before saving this run.
    st.session_state[history_key] = previous

previous = st.session_state[history_key]

if previous.empty:
    st.info("No previously saved findings for this selection.")
else:
    st.dataframe(
        previous.drop(columns="country"),
        hide_index=True,
        width="stretch",
        column_config={
            "url": st.column_config.LinkColumn("Source URL"),
            "period": st.column_config.TextColumn(
                "Original period"
            ),
            "comparability": st.column_config.TextColumn(
                "Details",
                width="large"
            )
        }
    )

st.subheader("Findings from current run")

if result["observations"]:

    # Match observations with the original SUA data
    keys = ["country", "commodity", "year", "element"]

    sua = st.session_state["investigation_data"].rename(columns={
        "Area": "country",
        "Item": "commodity",
        "Year": "year",
        "Element": "element",
        "Value": "sua_value",
        "Unit": "unit"
    })

    sua["year"] = sua["year"].astype(int)

    sua = sua[keys + ["sua_value", "unit"]].drop_duplicates(
        subset=keys
    )

    # Convert agent observations into a DataFrame
    df = pd.DataFrame(result["observations"]).rename(
        columns={"fao_value": "alternative_value"}
    )

    df["year"] = df["year"].astype(int)

    # Add the corresponding SUA figures
    df = df.merge(
        sua,
        on=keys,
        how="left",
        validate="many_to_one"
    )

    # Calculate difference from SUA
    df["difference"] = (
        df["alternative_value"] - df["sua_value"]
    )

    # Keep individual review decisions in session state
    if review_key not in st.session_state:
        st.session_state[review_key] = {}

    reviewed = st.session_state[review_key]

    # Keep the overview table
    st.dataframe(
        df.drop(columns="country"),
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

    st.subheader("Review individual observations")

    for index, row in df.iterrows():

        title = (
            f"{row['commodity']} / {row['year']} / "
            f"{row['element']} / {row['source_value']} "
            f"{row['source_unit']}"
        )

        with st.expander(title):

            st.write("Publisher:", row["publisher"])
            st.write("Source:", row["url"])
            st.write("Evidence:", row["evidence"])
            st.write("Original period:", row["period"])
            st.write("Details:", row["comparability"])

            # This observation has already been reviewed

            if index in reviewed:
                decision = reviewed[index]
                st.success(
                    f"Decision recorded: {decision['status']}"
                )
                if decision["comment"]:
                    st.write("Reason:", decision["comment"])
                if st.button(
                    "Change decision",
                    key=f"change_{run_id}_{index}"
                ):
                    reviewed.pop(index)

                    # Remove any memory proposal based on the old decision
                    st.session_state.pop(
                        f"memory_proposal_{run_id}", None
                    )
                    st.session_state.pop(
                        f"memory_editor_{run_id}", None
                    )
                    st.rerun()
                continue


            # Rejection comment
            reason = st.text_input(
                "Reason for rejection (required only if rejecting)",
                key=f"rejection_{run_id}_{index}"
            )

            col1, col2 = st.columns(2)

            with col1:
                approve = st.button(
                    "Approve observation",
                    key=f"approve_{run_id}_{index}"
                )

            with col2:
                reject = st.button(
                    "Reject observation",
                    key=f"reject_{run_id}_{index}"
                )

            if approve or reject:

                if reject and not reason.strip():
                    st.error("Please explain why this observation is rejected.")

                else:
                    status = "rejected" if reject else "accepted"
                    comment = reason.strip() if reject else ""

                    # Save only this observation
                    observation = df.loc[[index]].copy()
                    observation["review_status"] = status
                    observation["review_note"] = comment

                    try:
                        count = save_reviewed_findings(observation)

                    except sqlite3.Error as error:
                        st.error(f"Could not save observation: {error}")

                    else:
                        if count > 0:
                            reviewed[index] = {
                                "status": status,
                                "comment": comment
                            }
                            # Refresh saved findings displayed on the report page
                            st.session_state.pop(history_key, None)

                            # Remove memory suggestions based on previous decisions
                            st.session_state.pop(
                                f"memory_proposal_{run_id}", None
                            )
                            st.session_state.pop(
                                f"memory_editor_{run_id}", None
                            )

                            st.rerun()

                        else:
                            st.warning(
                                "This observation already exists in the database."
                            )

else:
    if result["status"] == "complete":
        st.info("No new alternative observations passed the checks.")
    else:
        st.info("No finalized observations are available for review.")


st.subheader("Summary")

st.markdown(result["search_summary"])


st.subheader("Agent Statistics")

stats = result.get("stats", {})
usage = result["usage"]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Investigation time",
    f"{result['seconds']:.1f} s"
)

col2.metric(
    "LLM tokens",
    f"{usage['total_tokens']:,}"
)

col3.metric(
    "Model calls",
    stats.get("model_calls", "N/A")
)

col4.metric(
    "Research tool calls",
    stats.get("tool_calls", "N/A")
)

st.caption(
    f"Input tokens: {usage['input_tokens']:,} | "
    f"Output tokens: {usage['output_tokens']:,}"
)

st.write(
    "Finalization trigger:",
    stats.get("finish_trigger", "Not recorded")
)

# Detailed usage by tool
if stats.get("by_tool"):
    tool_data = pd.DataFrame([
        {"Tool": name, "Calls": count}
        for name, count in stats["by_tool"].items()
    ])

    with st.expander("Research tool usage"):
        st.dataframe(
            tool_data,
            hide_index=True,
            width="stretch"
        )




# Search Experience
st.subheader("Search Experience")

# Memory can be proposed only after all observations are reviewed
total = len(result["observations"])
reviewed = st.session_state.get(review_key, {})

if result["status"] != "complete":
    st.info("No memory suggestion is available for an incomplete investigation.")

elif len(reviewed) < total:
    st.info(
        f"Review all observations first ({len(reviewed)}/{total} reviewed). "
        "Then you can generate a Search Experience suggestion."
    )

else:
    proposal_key = f"memory_proposal_{run_id}"
    editor_key = f"memory_editor_{run_id}"

    if st.button("Generate Search Experience suggestion"):

        existing_memory = load_search_experience()

        # Lessons identified by the research agent
        lessons = [
            line.strip(" -•\t")
            for line in result["updated_experience"].splitlines()
            if line.strip()
        ]

        # Add expert rejection explanations
        for index, decision in reviewed.items():
            if decision["status"] == "rejected":
                row = df.loc[index]

                lessons.append(
                    f"Expert rejected {row['publisher']} for "
                    f"{row['country']} / {row['commodity']} / "
                    f"{row['year']} / {row['element']}: "
                    f"{decision['comment']}"
                )


        # Generate lessons from the investigation summary if the research agent and expert feedback proposed nothing.
        if not lessons:
            try:
                model = ChatOpenAI(
                    model="gpt-6-luna",
                    use_responses_api=True,
                    max_completion_tokens=350,
                    timeout=35,
                    max_retries=0
                )

                response = model.invoke([
                    (
                        "system",
                        "You analyze completed statistical research investigations. "
                        "Suggest at most TWO genuinely new, specific, actionable "
                        "Search Experience lessons that would improve future searches. "
                        "Use only documented discoveries, failures and limitations. "
                        "Do not invent unsuccessful tool calls, source behavior or "
                        "successful extraction methods. "
                        "Do not repeat approved memory or generic search advice. "
                        "Return one lesson per line, without headings. "
                        "If nothing useful was learned, return exactly NONE."
                    ),
                    (
                        "human",
                        f"APPROVED MEMORY:\n{existing_memory}\n\n"
                        f"INVESTIGATION SUMMARY:\n"
                        f"{result['search_summary'][:6000]}"
                    )
                ])

                text = response.content

                if isinstance(text, str) and text.strip().upper() != "NONE":
                    lessons.extend(
                        line.strip(" -•\t")
                        for line in text.splitlines()
                        if line.strip(" -•\t")
                    )

            except Exception as exc:
                st.warning(f"Could not generate memory lessons: {exc}")


        # Split existing memory into entries for comparison
        existing_entries = [
        line.strip()
        for line in existing_memory.splitlines()
        if line.strip()
        ]

        new_lessons = []

        for lesson in lessons:

            # Normalize capitalization and punctuation
            normalized = re.sub(r"\W+", " ", lesson.casefold()).strip()

            if not normalized:
                continue

            # Compare with existing memory and other new lessons
            previous = existing_entries + new_lessons

            duplicate = any(
                SequenceMatcher(
                    None,
                    normalized,
                    re.sub(r"\W+", " ", old.casefold()).strip()
                ).ratio() >= 0.85
                for old in previous
            )

            if not duplicate:
                new_lessons.append(lesson)

        if new_lessons:
            # Preserve the existing memory and add only new lessons
            proposal = "\n".join(
                [existing_memory.strip()] + new_lessons
            ).strip()

            st.session_state[proposal_key] = proposal
            st.session_state.pop(editor_key, None)

        else:
            st.session_state.pop(proposal_key, None)
            st.session_state.pop(editor_key, None)
            st.info("No new Search Experience lessons were identified.")

    # Show the proposal only after it has been generated
    if proposal_key in st.session_state:

        st.caption(
            "Review the complete proposed Search Experience below. "
            "Existing memory is preserved; new lessons are added. "
            "You may edit or remove any suggested lesson."
        )

        experience = st.text_area(
            "Proposed Search Experience",
            value=st.session_state[proposal_key],
            height=220,
            key=editor_key
        )

        if len(experience) > 1800:
            st.warning(
                "Search Experience exceeds 1,800 characters. "
                "Shorten the text before saving."
            )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Approve and save Search Experience"):
                if len(experience) > 1800:
                    st.error("Memory is too long.")
                else:
                    try:
                        save_approved_search_experience(experience)
                    except (OSError, ValueError) as error:
                        st.error(f"Could not save memory: {error}")
                    else:
                        st.success("Search Experience approved and saved.")

        with col2:
            if st.button("Discard suggestion"):
                st.session_state.pop(proposal_key, None)
                st.session_state.pop(editor_key, None)
                st.rerun()


# Archive the completed report
st.subheader("Archive report")

if result["status"] != "complete":
    st.info("Only completed investigations can be archived.")

elif len(reviewed) < len(result["observations"]):
    st.info("Review all observations before archiving the report.")

else:
    if st.button("Save report to Archive"):

        selection = st.session_state["investigation_data"]

        # Include expert decisions with the findings
        findings = []

        if result["observations"]:
            archived_df = df.copy()

            archived_df["review_status"] = [
                reviewed[i]["status"] for i in df.index
            ]
            archived_df["review_note"] = [
                reviewed[i]["comment"] for i in df.index
            ]

            findings = json.loads(
                archived_df.to_json(orient="records")
            )

        report = {
            "country": selection["Area"].iloc[0],
            "commodities": selection["Item"].unique().tolist(),
            "elements": selection["Element"].unique().tolist(),
            "years": [
                int(selection["Year"].min()),
                int(selection["Year"].max())
            ],
            "result": result,
            "findings": findings
        }

        with sqlite3.connect("data/app.sqlite") as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS archived_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    report_json TEXT NOT NULL
                )
            """)

            conn.execute(
                "INSERT INTO archived_reports (report_json) VALUES (?)",
                (json.dumps(report, ensure_ascii=False),)
            )

        st.success("Investigation report saved")



# FAO Guidelines Review
st.subheader("Methodology Review")
st.caption(
    "FAO methodological guidance"
)

selection = st.session_state["investigation_data"]
country = str(selection["Area"].iloc[0])
commodities = selection["Item"].drop_duplicates().astype(str).tolist()
elements = selection["Element"].drop_duplicates().astype(str).tolist()


def fao_guideline_answer(question, search_query):
    passages = []

    # Retrieve from BOTH FAO documents
    for source, query_text in [
        ("FBS handbook.pdf", search_query),
        (
            "COMMODITY TREES & Conversion factors.pdf",
            f"{country} {', '.join(commodities)} {search_query}"
        )
    ]:
        hits = search_guidelines(
            query_text,
            source=source,
            n_results=3
        )

        for excerpt in (hits.get("documents") or [[]])[0]:
            passages.append((source, excerpt[:1400]))

    if not passages:
        return "No guideline passages were retrieved.", []

    evidence = "\n\n".join(
        f"[{i}] {source}\n{excerpt}"
        for i, (source, excerpt) in enumerate(passages, 1)
    )

    model = ChatOpenAI(
        model="gpt-6-luna",
        use_responses_api=True,
        max_completion_tokens=1800,
        timeout=45
    )

    response = model.invoke([
        (
            "system",
            "You are an FAO methodological assistant. "
            "Answer using only the provided guideline excerpts "
            "and investigation context. Cite supporting passages "
            "using [1], [2], etc. Do not invent PDF page numbers. "
            "Ignore irrelevant passages. State when evidence is "
            "insufficient. Historical conversion factors are not "
            "contemporary statistical observations. "
            "Do not make expert approval decisions."
        ),
        (
            "human",
            f"Investigation: {country}\n"
            f"Commodities: {', '.join(commodities)}\n"
            f"Elements: {', '.join(elements)}\n"
            f"Report summary:\n{result['search_summary'][:4000]}\n\n"
            f"Request: {question}\n\n"
            f"GUIDELINE EXCERPTS:\n{evidence}"
        )
    ])

    content = response.content

    answer = (
        content if isinstance(content, str)
        else "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )
    )

    return answer or "No textual answer was returned.", passages


def display_guideline_sources(passages):
    for i, (source, excerpt) in enumerate(passages, 1):
        st.markdown(f"**[{i}] {source}**")
        st.caption(excerpt)


# A. Automatic methodological assessment
analysis_key = f"fao_analysis_{run_id}"

if st.button(
    "Analyze investigation using FAO guidelines",
    key=f"fao_analyze_{run_id}"
):
    try:
        with st.spinner("Consulting FAO guidelines..."):
            st.session_state[analysis_key] = fao_guideline_answer(
                "Write up to three concise methodological considerations "
                "relevant to this investigation. "
                "Explain limitations, not verdicts.",
                "FAO SUA methodology comparability source quality "
                "reference periods "
                + " ".join(elements + commodities)
            )
    except Exception as exc:
        st.error(f"Guidelines review failed: {exc}")

if analysis_key in st.session_state:
    answer, passages = st.session_state[analysis_key]

    st.markdown(answer)

    with st.expander("Guideline passages used"):
        display_guideline_sources(passages)


# B. Ask questions about the investigation and FAO guidelines
st.markdown("**Ask the FAO guidelines**")

with st.form(key=f"fao_question_form_{run_id}"):
    question = st.text_input(
        "Question about this investigation or FAO methodology"
    )
    ask = st.form_submit_button("Ask")

chat_key = f"fao_questions_{run_id}"

if ask and question.strip():
    try:
        with st.spinner("Searching the FAO guidelines..."):
            answer, passages = fao_guideline_answer(
                question.strip(),
                question.strip()
            )

        st.session_state.setdefault(chat_key, []).append(
            (question.strip(), answer, passages)
        )

    except Exception as exc:
        st.error(f"Could not answer question: {exc}")

for i, (asked, answer, passages) in enumerate(
    st.session_state.get(chat_key, []), 1
):
    st.markdown(f"**Question {i}:** {asked}")
    st.markdown(answer)

    with st.expander(f"Sources for question {i}"):
        display_guideline_sources(passages)
