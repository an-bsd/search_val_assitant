import streamlit as st
from pathlib import Path
import json

st.title("Guidelines and Experience")

# List Search Guidelines
st.subheader("Search Guidelines")

for file in sorted(Path("guidelines/search").iterdir()):
    if file.is_file():
        st.write(file.name)

# List Validation Guidelines
st.subheader("Validation Guidelines")

for file in sorted(Path("guidelines/validation").iterdir()):
    if file.is_file():
        st.write(file.name)

# Load experience memories
settings_file = Path("data/guidelines_experience.json")
settings = json.loads(settings_file.read_text(encoding="utf-8"))

# Search Experience
st.subheader("Search Experience")

search_experience = st.text_area(
    "Search Experience Memory",
    value=settings.get("search_experience", ""),
    height=250
)

# Validation Experience
st.subheader("Validation Experience")

validation_experience = st.text_area(
    "Validation Experience Memory",
    value=settings.get("validation_experience", ""),
    height=250
)

# Save memories
if st.button("Save Experience"):

    settings["search_experience"] = search_experience
    settings["validation_experience"] = validation_experience

    settings_file.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )