import streamlit as st
import pandas as pd
import sqlite3
from agent import run_search


st.set_page_config(page_title="SUA Validation Assistant", layout="wide")

st.title("SUA Validation Assistant")

# Architecture diagram
with st.expander("Application Architecture", expanded=False):

    # st.caption(
    #     "FAOSTAT SUA dataset as source, autonomous research, user review, learning from feedback, methodological guidance"
    # )

    st.graphviz_chart(r"""
    digraph SUA {
        graph [
            rankdir=TB,
            bgcolor="transparent",
            pad="0.24",
            nodesep="0.42",
            ranksep="0.60",
            splines=polyline
        ]

        node [
            shape=box,
            style="rounded,filled",
            fillcolor="#F4F7FB",
            color="#97A6B5",
            fontcolor="#1A2B3B",
            fontname="Arial",
            fontsize=12,
            margin="0.18,0.12"
        ]

        edge [
            color="#73859B",
            fontcolor="#FFFFFF",
            arrowsize=0.70,
            fontname="Arial",
            fontsize=10
        ]

        bulk [
            label="FAOSTAT bulk download\nImported SUA data"
        ]

        sua [
            label="Local FAOSTAT SUA\nSQLite database"
        ]

        selection [
            label="USER INPUT\nCountry, commodities,\nelements and years",
            fillcolor="#DFF0FD"
        ]

        agent [
            label="AUTONOMOUS AI AGENT\nLangChain\nApproved search experience",
            fillcolor="#E6E6FC"
        ]

        tools [
            label="FREE CHOICE OF SIX TOOLS\nTavily search | Playwright | PDF\nExcel/CSV | UN Comtrade API | SQLite DB  | Local checks",
            fillcolor="#E6E6FC"
        ]

        structured [
            label="STRUCTURED RESPONSE\nNumerical candidate observations, source URLs,\n summary, suggestions",
            fillcolor="#DFF0FD"
        ]

        report [
            label="INVESTIGATION REPORT\nSource, comparison,\n expert review",
            fillcolor="#DFF0FD"
        ]

        alternatives [
            label="SAVED ALTERNATIVES\nSQLite, expert decisions",
            fillcolor="#E5F3E9"
        ]

        archive [
            label="REPORT ARCHIVE\nSaved investigation reports",
            fillcolor="#E5F3E9"
        ]

        memory [
            label="APPROVED SEARCH EXPERIENCE\nUpdateable JSON memory",
            fillcolor="#FEF2D6"
        ]

        documents [
            label="METHODOLOGICAL GUIDELINES\n2025 FAO Handbook + Conversion Factors"
        ]

        chroma [
            label="CHROMADB\nLangChain RecursiveCharacterTextSplitter\nEmbedding with default 'all-MiniLM-L6-v2' transformer\nSemantic similarity retrieval",
            fillcolor="#FEF2D6"
        ]

        guidance [
            label="FAO GUIDELINES REVIEW\nMethodological highlights\n Questions & answers",
            fillcolor="#FEF2D6"
        ]

        bulk -> sua -> selection -> agent
        agent -> structured -> report

        agent -> tools [
            dir=both,
            label="iterative tool calling",
            constraint=false
        ]

        report -> alternatives [label="expert review"]
        report -> archive [label="save snapshot"]
        report -> memory [label="approved memory"]

        memory -> agent [
            style=dashed,
            label="next investigation",
            constraint=false
        ]

        documents -> chroma -> guidance

        report -> guidance [
            style=dashed,
            label="report context"
        ]
    }
    """, width="stretch")



st.header("Choose a country, commodities, and year range to start investigation")

# Database
conn = sqlite3.connect("data/faostat_sua.sqlite")

# Index to speed up selections
conn.execute('CREATE INDEX IF NOT EXISTS idx_selection ON sua ("Area", "Item", "Year")')
conn.close()

# Cache database queries
@st.cache_data
def query(sql, params=()):
    with sqlite3.connect("data/faostat_sua.sqlite") as conn:
        return pd.read_sql(sql, conn, params=params)

# Country - single selection
countries = query('SELECT DISTINCT "Area" FROM sua ORDER BY "Area"')["Area"].tolist()

country = st.selectbox("Country", countries)

# Commodities - multiple selection
commodities = query('SELECT DISTINCT "Item" FROM sua WHERE "Area" = ? ORDER BY "Item"', (country,))["Item"].tolist()

selected_commodities = st.multiselect("Commodities", commodities)


# Elements - independent of selected commodities
elements = query(
    '''SELECT DISTINCT "Element" FROM sua
       WHERE "Area" = ?
       ORDER BY "Element"''',
    (country,)
)["Element"].tolist()

selected_elements = st.multiselect("Elements", elements)


# Year range
year_data = query('SELECT MIN("Year") AS start_year, MAX("Year") AS end_year FROM sua WHERE "Area" = ?',(country,))
start_year = int(year_data["start_year"][0])
end_year = int(year_data["end_year"][0])

# Resources
selected_years = st.slider(
    "Year range",
    min_value=start_year,
    max_value=end_year,
    value=(start_year, end_year)
)

# Show selected SUA data
st.subheader("Selected FAOSTAT SUA data")

if selected_commodities and selected_elements:

    commodity_placeholders = ",".join(["?"] * len(selected_commodities))
    element_placeholders = ",".join(["?"] * len(selected_elements))

    selected_data = query(
        f'''SELECT * FROM sua
            WHERE "Area" = ?
            AND "Item" IN ({commodity_placeholders})
            AND "Element" IN ({element_placeholders})
            AND CAST("Year" AS INTEGER) BETWEEN ? AND ?
            ORDER BY "Item", "Year", "Element"''',
        (
            country,
            *selected_commodities,
            *selected_elements,
            *selected_years
        )
    )

    st.dataframe(selected_data, height=400, width="stretch")

else:
    st.info("Select at least one commodity and one element to display SUA data.")


# Run investigation
if st.button("Run investigation"):
    if not selected_commodities:
        st.warning("Please select at least one commodity.")
    elif not selected_elements:
        st.warning("Please select at least one element.")
    elif selected_data.empty:
        st.warning("No SUA observations match this selection.")
    else:
        # Clear the previous result before starting another investigation.
        st.session_state.pop("search_result", None)

        try:
            with st.spinner("Investigating..."):
                result = run_search(selected_data)
        except Exception as error:
            st.error(f"Investigation failed: {error}")
            st.stop()

        st.session_state["search_result"] = result
        st.session_state["investigation_data"] = selected_data.copy()

        # Give each investigation its own review form.
        st.session_state["investigation_id"] = (
            st.session_state.get("investigation_id", 0) + 1
        )

        st.switch_page("pages/1_Investigation_report.py")