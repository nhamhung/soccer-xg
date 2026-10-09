"""Multi-page Streamlit app entry point.

Run locally:
    streamlit run app/streamlit_app.py
"""

import streamlit as st

from pages_src import about, finishing, model_page, shot_maps, simulator

st.set_page_config(page_title="Soccer xG Lab", page_icon="⚽", layout="wide")

pages = [
    # Every page module exports `render`, so give each an explicit url_path.
    st.Page(simulator.render, title="Shot Simulator", icon="⚽", url_path="simulator", default=True),
    st.Page(finishing.render, title="Finishing: Skill or Luck?", icon="🎯", url_path="finishing"),
    st.Page(shot_maps.render, title="Shot Maps & Match xG", icon="🗺️", url_path="shot-maps"),
    st.Page(model_page.render, title="Model vs StatsBomb", icon="📏", url_path="model"),
    st.Page(about.render, title="About", icon="ℹ️", url_path="about"),
]

st.navigation(pages).run()
