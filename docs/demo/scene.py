"""Demo scene. Serve first:  python -m http.server 8770 -d site
Then: python docs/demo/scene.py && python ~/.claude/skills/demo-gif/scripts/compose.py docs/demo/build --out docs/demo.gif
"""
import os, sys; sys.path.insert(0, os.path.expanduser("~/.claude/skills/demo-gif/scripts"))
from pathlib import Path
from director import Director, run

async def scene(d: Director):
    p = d.page
    await p.goto("http://localhost:8770/", wait_until="networkidle")  # warm the cache
    await p.locator("#daily.js-plotly-plot").wait_for()
    await d.goto("http://localhost:8770/", settle=0.5)
    await p.locator("#kpis .kpi").first.wait_for()
    await p.locator("#daily.js-plotly-plot").wait_for()
    await d.caption("Fleet health at a glance", step=1)
    await d.hold(1.2)
    await d.caption("Open the top alert", step=2)
    await d.click(p.locator("#top-alerts tr.click").first)
    await p.locator("#cell-chart.js-plotly-plot").wait_for()
    await d.hold(1.0)
    await d.caption("Actual vs. expected on the cell", step=3)
    await d.focus(p.locator("#cell-chart"), zoom=1.3)
    await d.arrow(p.locator("#cell-chart .shapelayer path").first, "Flagged drop", side="auto")
    await d.hold(2.0)
    await d.clear_arrow(); await d.unfocus()
    await d.caption("Spot capacity risk early", step=4)
    await d.click(p.get_by_role("button", name="Capacity"))
    await p.locator("#cap-chart.js-plotly-plot").wait_for()
    await d.hold(1.8)

run(scene, out=str(Path(__file__).with_name("build")), title="Network Traffic Forecasting",
    subtitle="Per-cell forecasts, anomaly alerts and capacity flags", url_label="traffic-ops.local")
