import os
import streamlit as st
import pandas as pd

from agent import HeatEquityAgent, OUTPUT_DIR
from generateSampleData import generate

st.set_page_config(page_title="Urban Heat Equity — Analysis Agent", layout="wide")

st.title("Urban Heat Equity — Data Analysis Agent")
st.caption(
    "An agent that decides for itself what's worth investigating in a census-tract "
    "heat/equity dataset, writes and runs its own Python, and stops when it thinks "
    "it has covered the key relationships — rather than following a fixed script."
)

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Get one at console.anthropic.com. Not stored anywhere.",
    )

    st.divider()
    st.subheader("Dataset")
    use_sample = st.checkbox("Use synthetic Urban Heat sample data", value=True)
    uploaded = None
    if not use_sample:
        uploaded = st.file_uploader("Upload your own CSV (tract-level)", type=["csv"])

    st.divider()
    run_clicked = st.button("Run Agent", type="primary", use_container_width=True)

# resolve dataset path
csv_path = None
if use_sample:
    csv_path = "urban_heat_sample.csv"
    if not os.path.exists(csv_path):
        generate(out_path=csv_path)
elif uploaded is not None:
    csv_path = "user_uploaded.csv"
    with open(csv_path, "wb") as f:
        f.write(uploaded.getbuffer())

if csv_path and os.path.exists(csv_path):
    with st.expander("Preview dataset", expanded=not run_clicked):
        st.dataframe(pd.read_csv(csv_path).head(10), use_container_width=True)

if run_clicked:
    if not api_key:
        st.error("Add your Anthropic API key in the sidebar first.")
        st.stop()
    if not csv_path:
        st.error("Upload a CSV or use the sample dataset first.")
        st.stop()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    st.header("Agent reasoning log")
    log_container = st.container(height=480, border=True)

    agent = HeatEquityAgent(api_key=api_key, csv_path=csv_path)

    with log_container:
        for event in agent.run():
            if event.kind == "thought":
                st.markdown(f"🧠 {event.content}")
            elif event.kind == "code":
                with st.expander("▸ ran code", expanded=False):
                    st.code(event.content, language="python")
            elif event.kind == "code_result":
                with st.expander("▸ output", expanded=False):
                    st.text(event.content[:2000])
            elif event.kind == "finding":
                st.success(f"**{event.extra.get('title','Finding')}** — {event.content}")
            elif event.kind == "done":
                st.info(f"**Wrapped up:** {event.content}")
            elif event.kind == "error":
                st.warning(event.content)

    st.divider()
    st.header("Final Report")

    if agent.summary:
        st.markdown(f"**Synthesis:** {agent.summary}")

    if not agent.findings:
        st.write("No findings were recorded.")
    else:
        for f in agent.findings:
            st.subheader(f.get("title", "Finding"))
            st.write(f.get("insight", ""))
            chart = f.get("chart_file")
            if chart:
                chart_path = os.path.join(OUTPUT_DIR, chart)
                if os.path.exists(chart_path):
                    st.image(chart_path, use_container_width=True)
                else:
                    st.caption(f"(referenced chart `{chart}` not found)")

    # export report as markdown
    if st.button("Prepare markdown report for download"):
        lines = [f"# Urban Heat Equity — Agent Analysis Report\n"]
        if agent.summary:
            lines.append(f"**Synthesis:** {agent.summary}\n")
        for f in agent.findings:
            lines.append(f"## {f.get('title','Finding')}\n\n{f.get('insight','')}\n")
        report_md = "\n".join(lines)
        st.download_button("Download report.md", report_md, file_name="urban_heat_report.md")
else:
    st.info("Set your API key, pick a dataset, and click **Run Agent** to start.")