"""
Data-Analysis Agent for Urban Heat Equity data.

Design idea: instead of hard-coding "compute correlation, make a scatterplot,
done", the agent gets a Python execution tool and has to DECIDE what's worth
investigating, run the code itself, look at the output, and choose whether to
keep digging or wrap up. That decision loop is the actual project — the
plumbing (API calls, exec sandboxing) is scaffolding.
"""
import io
import contextlib
import traceback
import os
import uuid
from dataclasses import dataclass, field

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from anthropic import Anthropic

MODEL = "claude-sonnet-4-6"
MAX_STEPS = 14
OUTPUT_DIR = "outputs"

SYSTEM_PROMPT = """You are a data analysis agent investigating an Urban Heat Equity \
dataset at the census-tract level. The overall question you're pursuing: \
"Is urban heat exposure distributed unequally by income, race, or access to \
green space, and if so, how strongly and through what pathway?"

You have a `df` (pandas DataFrame) already loaded in your Python session, plus \
pandas, numpy, matplotlib.pyplot (as plt), and seaborn (as sns) imported.

Ground rules:
- Decide for yourself what to check next based on what you've already found. \
Don't run a fixed checklist of stats for their own sake.
- Prefer analyses that address the equity question specifically (e.g. does the \
income->heat relationship route through tree canopy, or is it more about \
impervious surface? are minority-population tracts hotter even controlling \
for income?) over generic summary statistics.
- When you make a plot, save it with plt.savefig('outputs/<short_name>.png', \
dpi=110, bbox_inches='tight') then plt.close() — do not use plt.show().
- Call record_finding each time you land on something worth reporting, with \
a short evidence-backed claim (cite the actual numbers you computed).
- Aim for 4-7 solid findings, not 15 shallow ones. Depth over coverage.
- When you believe you've covered the key equity relationships in the data, \
call finish_analysis with a short synthesis.
- Think out loud briefly before each tool call so a reader can follow your \
reasoning, but keep it to 1-3 sentences — you're not writing the final report yet.
"""

TOOLS = [
    {
        "name": "execute_python",
        "description": (
            "Execute Python code in a persistent, stateful session. `df` is "
            "already loaded. Use for exploration, stats, and plots. Save any "
            "plot to outputs/<name>.png with plt.savefig(...) then plt.close(). "
            "Returns stdout and any error traceback."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code to run."}},
            "required": ["code"],
        },
    },
    {
        "name": "record_finding",
        "description": "Record one completed, evidence-backed insight for the final report.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "insight": {"type": "string", "description": "1-3 sentences, must reference concrete numbers you computed."},
                "chart_file": {"type": "string", "description": "Filename in outputs/ that supports this finding, if any. Omit if none."},
            },
            "required": ["title", "insight"],
        },
    },
    {
        "name": "finish_analysis",
        "description": "Call once the key equity relationships in the dataset have been covered (or step budget is nearly exhausted).",
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string", "description": "2-4 sentence synthesis tying the findings together."}},
            "required": ["summary"],
        },
    },
]


@dataclass
class AgentEvent:
    kind: str            # "thought" | "code" | "code_result" | "finding" | "done" | "error"
    content: str = ""
    extra: dict = field(default_factory=dict)


class HeatEquityAgent:
    def __init__(self, api_key: str, csv_path: str):
        self.client = Anthropic(api_key=api_key)
        self.df_path = csv_path
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # persistent exec namespace shared across tool calls
        self.namespace = {
            "pd": pd, "df": pd.read_csv(csv_path),
            "plt": plt,
        }
        try:
            import seaborn as sns
            self.namespace["sns"] = sns
        except ImportError:
            pass
        self.findings = []
        self.summary = None

    def _run_code(self, code: str) -> str:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, self.namespace)
        except Exception:
            return buf.getvalue() + "\n" + traceback.format_exc()
        out = buf.getvalue().strip()
        return out if out else "(no stdout — if you expected output, use print())"

    def run(self):
        """Generator that yields AgentEvents as the agent works, for live UI streaming."""
        schema_preview = self.namespace["df"].head(3).to_string()
        columns = list(self.namespace["df"].columns)
        user_msg = (
            f"Dataset columns: {columns}\n\nFirst rows:\n{schema_preview}\n\n"
            "Begin your investigation."
        )
        messages = [{"role": "user", "content": user_msg}]

        for step in range(MAX_STEPS):
            resp = self.client.messages.create(
                model=MODEL,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": resp.content})

            tool_results = []
            done = False

            for block in resp.content:
                if block.type == "text" and block.text.strip():
                    yield AgentEvent("thought", block.text.strip())

                elif block.type == "tool_use":
                    if block.name == "execute_python":
                        code = block.input["code"]
                        yield AgentEvent("code", code)
                        result = self._run_code(code)
                        yield AgentEvent("code_result", result)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result[:4000],
                        })

                    elif block.name == "record_finding":
                        f = block.input
                        self.findings.append(f)
                        yield AgentEvent("finding", f.get("insight", ""), extra=f)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Recorded.",
                        })

                    elif block.name == "finish_analysis":
                        self.summary = block.input.get("summary", "")
                        yield AgentEvent("done", self.summary)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Analysis complete.",
                        })
                        done = True

            if done:
                return

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                # no tool call at all — nudge it to either act or finish
                messages.append({"role": "user", "content": "Continue: call a tool (execute_python, record_finding, or finish_analysis)."})

        yield AgentEvent("error", "Hit step budget before the agent called finish_analysis. Showing findings gathered so far.")