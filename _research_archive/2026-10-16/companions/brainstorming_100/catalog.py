"""Build the browsable proposal catalog and a progress checkpoint."""
import json,re
from pathlib import Path
base=Path(__file__).parent
state=json.loads((base/"RUN_STATE.json").read_text(encoding="utf-8-sig"))
notes=sorted(base.glob("idea_[0-9][0-9][0-9].md"))
rows=[]
for note in notes:
    body=note.read_text(encoding="utf-8-sig")
    title=next((s.lstrip("# ").strip() for s in body.splitlines() if s.startswith("# ")) ,note.stem)
    title=re.sub(r"^\d{3}\s*[—–:-]\s*","",title)
    match=re.search(r"(?i)verdict\s*[:—]\s*(.+)",body.replace("**",""))
    verdict=(match.group(1).replace("**","").split(".")[0].strip() if match else "See note")
    if len(verdict)>100:verdict=verdict[:97]+"..."
    rows.append(f"| {note.stem[5:]} | [{title}]({note.name}) | {verdict.replace('|','/')} |")
target=state['requested']
text=f"# {target} Astra Ultra research investigations\n\n"
text+=f"Progress: **{len(notes)}/{target} proposal notes saved**; **{len(state['dispatched'])}/{target} investigations assigned**. The user reduced the original 100-idea scope to 40. Original proposal IDs are retained for traceability.\n\n"
text+="All workers were requested with model **gpt-6-astra** and reasoning effort **ultra**. The runtime rejected further creation after **21 distinct numbered agents** (plus three initial landscape readers). Existing workers were reused for the remaining assignments. The investigation count is not a distinct-agent count. Notes include proposals, checked results, and failed routes; the final report identifies the strongest theorem and merges equivalent methods.\n\n"
text+="[Primary audit](ROOT_AUDIT.md) · [Assignment brief](BRIEF.md) · [Original 100-seed plan; only selected 40 completed](SEEDS.md) · [Run checkpoint](RUN_STATE.json)\n\n"
text+="| ID | Proposal | Agent verdict |\n|---|---|---|\n"+"\n".join(rows)+"\n"
(base/"INDEX.md").write_text(text,encoding="utf-8")
(base/"STATUS.md").write_text(f"# Status\n\nUser reduced scope from100 to{target}. Runtime permitted21 distinct numbered Astra Ultra agents; remaining assignments reuse them. Three initial landscape readers are additional.\n\nAssigned: {len(state['dispatched'])}/{target}. Proposal notes saved: {len(notes)}/{target}.\n\nSee RUN_STATE.json for exact completed, running, and cancelled assignments. INDEX.md links every saved proposal.\n",encoding="utf-8")
print(json.dumps({"dispatched":len(state["dispatched"]),"notes_saved":len(notes),"ids":[int(p.stem[5:]) for p in notes]}))
