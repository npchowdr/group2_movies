from pathlib import Path

import streamlit as st

from vasundara_apputil import (
    load_data,
    ROI_VARIABLES,
    roi_relationship_figure
)


st.set_page_config(
    page_title="ROI Relationship Analysis",
    layout="wide"
)


# Load movie dataset
data_path = Path(__file__).parent / "movie_roi.csv"
df = load_data(data_path)


st.title("Movie ROI Relationship Analysis")

st.write(
    "Explore how different movie characteristics relate to ROI. "
    "For each characteristic, movies are divided into four approximately "
    "equal groups: Low, Medium, High, and Very High."
)


selected_name = st.selectbox(
    "Choose a movie characteristic",
    list(ROI_VARIABLES.keys())
)

selected_variable = ROI_VARIABLES[selected_name]


fig, correlation, summary = roi_relationship_figure(
    df,
    selected_variable,
    selected_name
)


st.plotly_chart(
    fig,
    width="stretch"
)


st.metric(
    "Pearson correlation (r)",
    f"{correlation:.3f}"
)


st.caption(
    "Hover over each point to see the median ROI, mean ROI, "
    "and number of movies in that group."
)