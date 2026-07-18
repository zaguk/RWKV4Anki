# ruff: noqa: E501
from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, Literal

from .behavior_lab import (
    BehaviorLabExperiment,
    BehaviorLabResult,
    cohort_delta_summary,
    observation_deltas,
)
from .modal_html import (
    FieldOption,
    ModalButton,
    ModalField,
    render_button,
    render_card,
    render_close_footer,
    render_field,
    render_modal_shell,
    render_notice,
    render_prompt_overlay_template,
)

_GRAPH_WIDTH = 920
_GRAPH_HEIGHT = 330
_GRAPH_LEFT = 58
_GRAPH_RIGHT = 20
_GRAPH_TOP = 18
_GRAPH_BOTTOM = 42


def render_behavior_lab_editor(
    experiment: BehaviorLabExperiment,
    *,
    saved_experiments: Mapping[str, BehaviorLabExperiment] | None = None,
    message: str = "",
    generation: int = 1,
    is_dark: bool = False,
) -> str:
    saved = {str(name): value.to_dict() for name, value in (saved_experiments or {}).items()}
    initial_json = _script_json(experiment.to_dict())
    saved_json = _script_json(saved)
    message_html = render_notice(message, tone="success") if message else ""
    baseline_fields = "".join(
        (
            _field(
                ModalField("experiment-name", "Experiment name"),
            ),
            _field(
                ModalField(
                    "focal-card",
                    "Focal card ID",
                    kind="number",
                    minimum=1,
                )
            ),
            _field(
                ModalField(
                    "baseline-time",
                    "Baseline time",
                    kind="datetime-local",
                    step=1,
                )
            ),
            _field(
                ModalField(
                    "tracked-search",
                    "Tracked Anki search",
                    placeholder="optional, e.g. deck:Biology",
                )
            ),
            _field(
                ModalField(
                    "tracked-cards",
                    "Additional tracked card IDs",
                    placeholder="comma-separated",
                ),
                "rwkv-behavior-field--wide",
            ),
            _field(
                ModalField(
                    "selection-cards",
                    "Browser selection / context IDs",
                    placeholder="comma-separated",
                ),
                "rwkv-behavior-field--wide",
            ),
            _field(
                ModalField(
                    "track-siblings",
                    "Track siblings",
                    kind="checkbox",
                )
            ),
            _field(
                ModalField(
                    "track-collection",
                    "Track whole collection",
                    kind="checkbox",
                )
            ),
        )
    )
    template_fields = "".join(
        (
            _field(
                ModalField(
                    "template-name",
                    "Template",
                    kind="select",
                    value="rating_comparison",
                    options=(
                        FieldOption("rating_comparison", "Rating comparison"),
                        FieldOption("review_context", "Review versus filtered"),
                        FieldOption("sibling_spillover", "Sibling spillover"),
                        FieldOption("intervening_reviews", "Intervening reviews"),
                        FieldOption("good_streak", "Good-review streak"),
                        FieldOption("custom", "Custom"),
                    ),
                )
            ),
            _field(
                ModalField(
                    "template-delay",
                    "Initial delay (days)",
                    kind="number",
                    value=7,
                    minimum=0,
                    step=0.01,
                )
            ),
            _field(
                ModalField(
                    "template-duration",
                    "Answer time (seconds)",
                    kind="number",
                    value=5,
                    minimum=0,
                    maximum=60,
                    step=0.1,
                )
            ),
            _field(
                ModalField(
                    "template-context-count",
                    "Context reviews",
                    kind="number",
                    value=100,
                    minimum=0,
                    maximum=60_000,
                )
            ),
        )
    )
    sweep_fields = "".join(
        (
            render_field(
                ModalField(
                    "sweep-field",
                    "Field to vary",
                    kind="select",
                    value="rating",
                    options=(
                        FieldOption("rating", "Rating"),
                        FieldOption("review_context", "Review context"),
                        FieldOption("delay", "Review delay (seconds)"),
                        FieldOption("duration", "Duration (seconds)"),
                        FieldOption("context_count", "Context count"),
                    ),
                )
            ),
            render_field(
                ModalField(
                    "sweep-values",
                    "Values",
                    placeholder="e.g. 1, 2, 3, 4",
                )
            ),
        )
    )
    saved_field = render_field(
        ModalField(
            "saved-name",
            "Saved experiment",
            kind="select",
            options=(FieldOption("", "Choose saved experiment…"),),
        )
    )
    json_field = render_field(
        ModalField(
            "json-exchange",
            "Experiment JSON",
            kind="textarea",
            rows=10,
        )
    )
    baseline_card = render_card(
        f'<div class="rwkv-behavior-grid rwkv-behavior-baseline-grid">{baseline_fields}</div>',
        title="Baseline and tracked cards",
        extra_classes=("rwkv-behavior-card", "rwkv-behavior-baseline"),
    )
    template_card = render_card(
        f'<div class="rwkv-behavior-template-row">{template_fields}'
        f'{_local_button("Apply template", "apply-template")}</div>',
        title="Start from a template",
        extra_classes=("rwkv-behavior-card", "rwkv-behavior-template"),
    )
    scenarios_card = render_card(
        '<div class="rwkv-behavior-section-title"><h2 class="rwkv-card-title">Scenarios</h2>'
        '<button class="rwkv-button rwkv-button--secondary rwkv-behavior-icon-button" '
        'id="add-scenario" type="button" aria-label="Add scenario">+</button></div>'
        '<div id="scenario-list"></div>'
        '<div class="rwkv-behavior-toolbar rwkv-behavior-toolbar--compact">'
        f'{_local_button("Duplicate", "duplicate-scenario")}'
        f'{_local_button("Delete", "delete-scenario", variant="destructive")}</div>'
        '<div class="rwkv-behavior-sweep"><h3>Sweep selected scenario</h3>'
        f'{sweep_fields}{_local_button("Create sweep", "apply-sweep")}</div>',
        extra_classes=("rwkv-behavior-card", "rwkv-behavior-scenarios"),
        aria_label="Scenario editor",
    )
    timeline_card = render_card(
        '<div class="rwkv-behavior-section-title">'
        '<h2 class="rwkv-card-title" id="timeline-title">Timeline</h2>'
        '<div class="rwkv-behavior-toolbar rwkv-behavior-toolbar--compact">'
        '<button class="rwkv-button rwkv-button--secondary" type="button" data-add-event="review">+ Review</button>'
        '<button class="rwkv-button rwkv-button--secondary" type="button" data-add-event="wait">+ Wait</button>'
        '<button class="rwkv-button rwkv-button--secondary" type="button" data-add-event="context">+ Context</button>'
        '<button class="rwkv-button rwkv-button--secondary" type="button" data-add-event="observe">+ Observe</button>'
        '</div></div><div id="event-list"></div>',
        extra_classes=("rwkv-behavior-card", "rwkv-behavior-timeline"),
        aria_label="Scenario timeline editor",
    )
    storage_card = render_card(
        '<div class="rwkv-behavior-storage-row">'
        f'{saved_field}<div class="rwkv-behavior-toolbar">'
        f'{_local_button("Load", "load-saved")}'
        f'{_local_button("Save current", "save-current")}'
        f'{_local_button("Delete saved", "delete-saved", variant="destructive")}'
        '</div></div>'
        '<details><summary>Import or export experiment JSON</summary>'
        f'{json_field}<div class="rwkv-behavior-toolbar rwkv-behavior-toolbar--compact">'
        f'{_local_button("Show current JSON", "export-json")}'
        f'{_local_button("Import JSON", "import-json")}</div></details>',
        title="Saved experiments",
        extra_classes=("rwkv-behavior-card", "rwkv-behavior-storage"),
    )
    page_html = f"""
 <main class="rwkv-modal-page rwkv-modal-page--wide" id="lab-page">
  <header class="rwkv-page-header rwkv-behavior-header">
    <div>
      <h1 class="rwkv-page-title" id="behavior-lab-title">RWKV Behavior Lab</h1>
      <p class="rwkv-page-intro">Run disposable counterfactual timelines from the current checkpoint.</p>
    </div>
    {_local_button("Run all scenarios", "run", variant="primary")}
  </header>
  {message_html}
  {render_notice(
      "Synthetic events never write Anki reviews or enter the durable checkpoint. "
      "Normal checkpoint loading may first incorporate real reviews already present in Anki. "
      "The first scenario is the matched control used for probability-point deltas.",
      tone="warning",
  )}
  {baseline_card}
  {template_card}
  <div class="rwkv-behavior-workspace">{scenarios_card}{timeline_card}</div>
  {storage_card}
 </main>
""".strip()
    custom_script = f"""
<script>
(() => {{
  const colors = ["#607d8b", "#1976d2", "#ef6c00", "#2e7d32", "#c62828", "#6a1b9a", "#00838f", "#ad1457"];
  let state = {initial_json};
  let saved = {saved_json};
  let selectedScenario = 0;
  const $ = (id) => document.getElementById(id)
    || document.getElementById(`rwkv-field-${{id}}`);
  const esc = (value) => String(value ?? "").replace(/[&<>\"']/g,
    c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'\"':"&quot;","'":"&#39;"}}[c]));
  const number = (value, fallback=0) => Number.isFinite(Number(value)) ? Number(value) : fallback;
  const ids = (text) => String(text || "").split(/[\\s,]+/).filter(Boolean)
    .map(Number).filter(n => Number.isInteger(n) && n > 0);
  const send = (action, payload={{}}) => window.RWKVModal.send(action, payload).catch(() => {{}});
  const defaultEvent = (kind) => ({{
    kind, label: kind === "observe" ? "Observation" : "", after_seconds: 0,
    card_id: null, rating: 3, review_context: "review", duration_seconds: 5,
    capture_curve: true, context_scope: "unrelated", context_count: 100,
    context_spacing_seconds: 1, context_rating_mode: "collection", context_seed: 5489,
    context_card_ids: []
  }});
  function syncBaseline() {{
    state.name = $("experiment-name").value.trim() || "Untitled experiment";
    state.focal_card_id = Math.trunc(number($("focal-card").value));
    state.tracked_card_ids = ids($("tracked-cards").value);
    state.selection_card_ids = ids($("selection-cards").value);
    state.track_siblings = $("track-siblings").checked;
    state.track_collection = $("track-collection").checked;
    state.tracked_search = $("tracked-search").value.trim();
    const raw = $("baseline-time").value;
    state.baseline_timestamp_seconds = raw ? new Date(raw).getTime() / 1000 : null;
  }}
  function populateBaseline() {{
    $("experiment-name").value = state.name || "";
    $("focal-card").value = state.focal_card_id || "";
    $("tracked-cards").value = (state.tracked_card_ids || []).join(", ");
    $("selection-cards").value = (state.selection_card_ids || []).join(", ");
    $("track-siblings").checked = state.track_siblings !== false;
    $("track-collection").checked = !!state.track_collection;
    $("tracked-search").value = state.tracked_search || "";
    if (state.baseline_timestamp_seconds != null) {{
      const date = new Date(state.baseline_timestamp_seconds * 1000);
      const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
      $("baseline-time").value = local.toISOString().slice(0, 19);
    }} else {{ $("baseline-time").value = ""; }}
    window.RWKVModal?.syncPopupControl($("baseline-time"));
  }}
  function renderSaved() {{
    const selected = $("saved-name").value;
    $("saved-name").innerHTML = '<option value="">Choose saved experiment…</option>' +
      Object.keys(saved).sort((a,b) => a.localeCompare(b)).map(name =>
        `<option value="${{esc(name)}}">${{esc(name)}}</option>`).join("");
    if (saved[selected]) $("saved-name").value = selected;
    window.RWKVModal?.initializePopupControls($("saved-name").parentElement);
    window.RWKVModal?.syncPopupControl($("saved-name"));
  }}
  function renderScenarios() {{
    if (!state.scenarios?.length) state.scenarios = [{{name:"Control", color:colors[0], events:[]}}];
    selectedScenario = Math.max(0, Math.min(selectedScenario, state.scenarios.length - 1));
    $("scenario-list").innerHTML = state.scenarios.map((scenario, index) => `
      <button class="rwkv-button rwkv-button--quiet rwkv-behavior-scenario ${{index === selectedScenario ? "selected" : ""}}" type="button" data-scenario="${{index}}">
        <span class="rwkv-behavior-swatch" style="background:${{esc(scenario.color || colors[index % colors.length])}}"></span>
        <span>${{esc(scenario.name || `Scenario ${{index + 1}}`)}}</span>
        ${{index === 0 ? '<small>control</small>' : ''}}
      </button>`).join("");
    document.querySelectorAll("[data-scenario]").forEach(button => button.onclick = () => {{
      selectedScenario = Number(button.dataset.scenario); renderScenarios(); renderEvents();
    }});
  }}
  function updateEvent(index, field, value) {{
    const event = state.scenarios[selectedScenario].events[index];
    const numeric = new Set(["after_seconds","rating","duration_seconds","context_count",
      "context_spacing_seconds","context_seed"]);
    if (field === "card_id") event[field] = value === "" ? null : Math.trunc(number(value));
    else if (field === "context_card_ids") event[field] = ids(value);
    else if (field === "capture_curve") event[field] = !!value;
    else if (numeric.has(field)) event[field] = number(value);
    else event[field] = value;
  }}
  function field(label, body, wide=false) {{
    return `<label class="rwkv-field rwkv-behavior-event-field ${{wide ? "rwkv-behavior-field--wide" : ""}}"><span class="rwkv-field__label">${{esc(label)}}</span>${{body}}</label>`;
  }}
  function input(index, fieldName, value, type="text", attrs="") {{
    return `<input class="rwkv-field__control" data-event-field="${{fieldName}}" data-event-index="${{index}}" type="${{type}}" value="${{esc(value ?? "")}}" ${{attrs}}>`;
  }}
  function select(index, fieldName, value, options) {{
    return `<select class="rwkv-field__control" data-event-field="${{fieldName}}" data-event-index="${{index}}">` +
      options.map(([key,label]) => `<option value="${{key}}" ${{String(key) === String(value) ? "selected" : ""}}>${{label}}</option>`).join("") + `</select>`;
  }}
  function eventFields(event, index) {{
    const common = field("Label", input(index,"label",event.label), true);
    if (event.kind === "wait") return common + field("Wait seconds", input(index,"after_seconds",event.after_seconds,"number",'min="0" step="0.1"'));
    const delay = field("Advance before event (seconds)", input(index,"after_seconds",event.after_seconds,"number",'min="0" step="0.1"'));
    if (event.kind === "observe") return common + delay;
    const ratings = [[1,"Again (1)"],[2,"Hard (2)"],[3,"Good (3)"],[4,"Easy (4)"]];
    const contexts = [["new","New / learning start"],["learning","Learning"],["review","Review"],["relearning","Relearning"],["filtered","Filtered"]];
    if (event.kind === "review") return common + delay +
      field("Card ID (blank = focal)", input(index,"card_id",event.card_id,"number",'min="1"')) +
      field("Rating", select(index,"rating",event.rating,ratings)) +
      field("Review context", select(index,"review_context",event.review_context,contexts)) +
      field("Answer time (seconds)", input(index,"duration_seconds",event.duration_seconds,"number",'min="0" max="60" step="0.1"')) +
      `<div class="rwkv-field rwkv-field--choice"><label class="rwkv-checkbox"><input data-event-field="capture_curve" data-event-index="${{index}}" type="checkbox" ${{event.capture_curve !== false ? "checked" : ""}}><span class="rwkv-checkbox__label">Capture post-answer curve</span></label></div>`;
    const scopes = [["unrelated","Unrelated (exclude focal note)"],["siblings","Siblings"],
      ["same_deck","Same deck"],["same_preset","Same preset"],["selection","Browser selection"],["collection","Whole collection"]];
    const ratingModes = [["collection","Reuse sampled collection outcomes"],["fixed","Use fixed outcome"]];
    return common + delay + field("Card scope", select(index,"context_scope",event.context_scope,scopes)) +
      field("Review count", input(index,"context_count",event.context_count,"number",'min="0" max="60000"')) +
      field("Seconds between reviews", input(index,"context_spacing_seconds",event.context_spacing_seconds,"number",'min="0" step="0.1"')) +
      field("Outcome source", select(index,"context_rating_mode",event.context_rating_mode,ratingModes), true) +
      field("Fixed rating", select(index,"rating",event.rating,ratings)) +
      field("Fixed context", select(index,"review_context",event.review_context,contexts)) +
      field("Fixed answer time", input(index,"duration_seconds",event.duration_seconds,"number",'min="0" max="60" step="0.1"')) +
      field("Random seed", input(index,"context_seed",event.context_seed,"number")) +
      field("Explicit card IDs (optional)", input(index,"context_card_ids",(event.context_card_ids || []).join(", ")), true);
  }}
  function renderEvents() {{
    const scenario = state.scenarios[selectedScenario];
    $("timeline-title").textContent = scenario.name || "Timeline";
    $("event-list").innerHTML = (scenario.events || []).map((event,index) => `
      <article class="rwkv-behavior-event">
        <div class="rwkv-behavior-event-head"><strong>${{index + 1}} · ${{esc(event.kind)}}</strong>
          <div class="rwkv-behavior-toolbar rwkv-behavior-toolbar--compact"><button class="rwkv-button rwkv-button--secondary" type="button" aria-label="Move event up" data-move-up="${{index}}">↑</button><button class="rwkv-button rwkv-button--secondary" type="button" aria-label="Move event down" data-move-down="${{index}}">↓</button><button class="rwkv-button rwkv-button--destructive" type="button" data-remove-event="${{index}}">Remove</button></div></div>
        <div class="rwkv-behavior-grid rwkv-behavior-event-grid">${{eventFields(event,index)}}</div>
      </article>`).join("") || '<div class="rwkv-notice">Add an event to this scenario.</div>';
    window.RWKVModal?.initializePopupControls($("event-list"));
    document.querySelectorAll("[data-event-field]").forEach(control => {{
      const handler = () => updateEvent(Number(control.dataset.eventIndex), control.dataset.eventField,
        control.type === "checkbox" ? control.checked : control.value);
      control.onchange = handler; control.oninput = handler;
    }});
    document.querySelectorAll("[data-remove-event]").forEach(button => button.onclick = () => {{
      scenario.events.splice(Number(button.dataset.removeEvent),1); renderEvents();
    }});
    document.querySelectorAll("[data-move-up]").forEach(button => button.onclick = () => moveEvent(Number(button.dataset.moveUp),-1));
    document.querySelectorAll("[data-move-down]").forEach(button => button.onclick = () => moveEvent(Number(button.dataset.moveDown),1));
  }}
  function moveEvent(index, delta) {{
    const events = state.scenarios[selectedScenario].events; const target = index + delta;
    if (target < 0 || target >= events.length) return;
    [events[index],events[target]] = [events[target],events[index]]; renderEvents();
  }}
  function render() {{ populateBaseline(); renderSaved(); renderScenarios(); renderEvents(); }}
  ["experiment-name","focal-card","tracked-cards","selection-cards","track-siblings",
   "track-collection","tracked-search","baseline-time"].forEach(id => $(id).onchange = syncBaseline);
  $("run").onclick = () => {{ syncBaseline(); send("run",{{experiment:state}}); }};
  $("apply-template").onclick = () => {{ syncBaseline(); send("template",{{
    template:$("template-name").value, focal_card_id:state.focal_card_id,
    selection_card_ids:state.selection_card_ids,
    delay_seconds:number($("template-delay").value)*86400,
    duration_seconds:number($("template-duration").value),
    context_count:Math.trunc(number($("template-context-count").value))
  }}); }};
  document.querySelectorAll("[data-add-event]").forEach(button => button.onclick = () => {{
    state.scenarios[selectedScenario].events.push(defaultEvent(button.dataset.addEvent)); renderEvents();
  }});
  $("add-scenario").onclick = () => {{ state.scenarios.push({{name:`Scenario ${{state.scenarios.length+1}}`, color:colors[state.scenarios.length%colors.length], events:[]}}); selectedScenario=state.scenarios.length-1; renderScenarios(); renderEvents(); }};
  $("duplicate-scenario").onclick = () => {{ const copy=structuredClone(state.scenarios[selectedScenario]); copy.name += " copy"; copy.color=colors[state.scenarios.length%colors.length]; state.scenarios.push(copy); selectedScenario=state.scenarios.length-1; renderScenarios(); renderEvents(); }};
  $("delete-scenario").onclick = () => {{ if(state.scenarios.length<=1) return; state.scenarios.splice(selectedScenario,1); selectedScenario=Math.max(0,selectedScenario-1); renderScenarios(); renderEvents(); }};
  $("apply-sweep").onclick = () => {{
    const fieldName=$("sweep-field").value, values=$("sweep-values").value.split(",").map(v=>v.trim()).filter(Boolean);
    if(!values.length) return; const source=state.scenarios[selectedScenario], kind=fieldName==="context_count"?"context":"review";
    const eventIndex=source.events.findIndex(e=>e.kind===kind); if(eventIndex<0) return;
    state.scenarios=values.map((value,index)=>{{ const copy=structuredClone(source), event=copy.events[eventIndex];
      if(fieldName==="review_context") event.review_context=value.toLowerCase();
      else if(fieldName==="delay") event.after_seconds=number(value);
      else if(fieldName==="duration") event.duration_seconds=number(value);
      else if(fieldName==="context_count") event.context_count=Math.trunc(number(value));
      else event.rating=Math.trunc(number(value));
      copy.name=`${{source.name}} · ${{value}}`; copy.color=colors[index%colors.length]; return copy; }});
    selectedScenario=0; renderScenarios(); renderEvents();
  }};
  $("save-current").onclick = async () => {{
    syncBaseline();
    const response = await window.RWKVPrompt.show({{
      title:"Save experiment", message:"Choose a name for this reusable experiment.",
      label:"Experiment name", value:state.name || "", confirmLabel:"Save experiment",
      cancelLabel:"Cancel", required:true, maxLength:160,
      requiredMessage:"Enter a name before saving."
    }});
    if (!response.accepted) return;
    const name=String(response.value || "").trim();
    send("save",{{name,experiment:state}});
  }};
  $("load-saved").onclick = () => {{ const name=$("saved-name").value; if(saved[name]){{ state=structuredClone(saved[name]); selectedScenario=0; render(); }} }};
  $("delete-saved").onclick = () => {{
    const name=$("saved-name").value;
    if(!name) return;
    send("delete",{{name}});
  }};
  $("export-json").onclick = () => {{ syncBaseline(); $("json-exchange").value=JSON.stringify(state,null,2); }};
  $("import-json").onclick = async () => {{
    const response = await window.RWKVPrompt.show({{
      title:"Import experiment JSON",
      message:"Paste or edit a complete Behavior Lab experiment.",
      label:"Experiment JSON",
      value:$("json-exchange").value,
      multiline:true,
      required:true,
      requiredMessage:"Enter experiment JSON before importing.",
      confirmLabel:"Import experiment",
      validate:value => {{
        try {{ JSON.parse(value); return ""; }}
        catch(error) {{ return `The JSON is not valid: ${{String(error?.message || error)}}`; }}
      }}
    }});
    if (!response.accepted) return;
    $("json-exchange").value=response.value;
    const imported=JSON.parse(response.value);
    send("import",{{experiment:imported}});
  }};
  $("json-exchange").spellcheck = false;
  render();
}})();
</script>
""".strip()
    document = render_modal_shell(
        page_html=page_html,
        generation=generation,
        title_id="behavior-lab-title",
        footer_html=render_close_footer(),
        overlay_html=render_prompt_overlay_template(),
        is_dark=is_dark,
        root_extra_classes="rwkv-behavior-lab rwkv-behavior-lab--editor",
    )
    return f"{document}\n{custom_script}"


def render_behavior_lab_results(
    result: BehaviorLabResult,
    *,
    generation: int = 1,
    is_dark: bool = False,
) -> str:
    graph = _result_graph(result)
    focal_rows = _focal_observation_rows(result)
    curve_rows = _curve_rows(result)
    impact_rows = _impact_rows(result)
    cohort_rows = _cohort_rows(result)
    event_rows = _event_rows(result)
    report_json = html.escape(
        json.dumps(behavior_lab_result_to_dict(result), indent=2, sort_keys=True),
        quote=False,
    )
    fingerprint = (
        f" · checkpoint {html.escape(result.checkpoint_fingerprint[:16])}"
        if result.checkpoint_fingerprint
        else ""
    )
    result_cards = "".join(
        (
            render_card(
                graph,
                title=f"Focal card {result.focal_card_id}",
                extra_classes=("rwkv-behavior-card",),
            ),
            render_card(
                _table(
                    ("Scenario", "Observation", "Simulation time", "Retrievability", "Δ control"),
                    focal_rows,
                ),
                title="Focal observations",
                extra_classes=("rwkv-behavior-card",),
            ),
            render_card(
                _table(
                    (
                        "Scenario",
                        "Event",
                        "At answer",
                        "1 min",
                        "1 hour",
                        "1 day",
                        "7 days",
                        "30 days",
                    ),
                    curve_rows,
                ),
                title="Post-answer curves",
                extra_classes=("rwkv-behavior-card",),
            ),
            render_card(
                '<p class="rwkv-subtle rwkv-behavior-subtle">Final current-state observation '
                "in each scenario, compared with the control observation at the same ordinal.</p>"
                + _table(
                    ("Card", "Relationship", *(scenario.name for scenario in result.scenarios)),
                    impact_rows,
                ),
                title="Card and sibling impact",
                extra_classes=("rwkv-behavior-card",),
            ),
            render_card(
                _table(
                    (
                        "Scenario",
                        "Observation",
                        "Cards",
                        "Mean Δ",
                        "Median Δ",
                        "Minimum Δ",
                        "Maximum Δ",
                    ),
                    cohort_rows,
                ),
                title="Cohort changes",
                extra_classes=("rwkv-behavior-card",),
            ),
            render_card(
                _table(
                    (
                        "Scenario",
                        "Event",
                        "Card",
                        "Rating",
                        "Context",
                        "Answer time",
                        "Elapsed",
                        "P before answer",
                    ),
                    event_rows,
                ),
                title="Processed-event ledger",
                extra_classes=("rwkv-behavior-card",),
            ),
        )
    )
    page_html = f"""
 <main class="rwkv-modal-page rwkv-modal-page--wide">
  <header class="rwkv-page-header rwkv-behavior-header"><div><h1 class="rwkv-page-title" id="behavior-lab-title">{
        html.escape(result.experiment_name)
    }</h1>
    <p class="rwkv-page-intro">Model {html.escape(result.model_id)}{fingerprint}</p></div>
    <div class="rwkv-behavior-toolbar">{_local_button("Edit experiment", "edit")}{_local_button("Run again", "rerun", variant="primary")}</div>
  </header>
  {render_notice(
      "Solid lines are current-state observations. Dashed lines are frozen post-answer curves; "
      "later reviews can change shared recurrent state without changing an older curve.",
      tone="warning",
  )}
  {result_cards}
  <details class="rwkv-card rwkv-behavior-card rwkv-behavior-result-json"><summary>Reproducible result JSON</summary><textarea class="rwkv-field__control" aria-label="Reproducible result JSON" readonly>{
        report_json
    }</textarea></details>
 </main>
""".strip()
    custom_script = """
<script>
document.getElementById("edit").onclick = () => window.RWKVModal.send("edit", {}).catch(() => {});
document.getElementById("rerun").onclick = () => window.RWKVModal.send("rerun", {}).catch(() => {});
</script>
""".strip()
    document = render_modal_shell(
        page_html=page_html,
        generation=generation,
        title_id="behavior-lab-title",
        footer_html=render_close_footer(),
        is_dark=is_dark,
        root_extra_classes="rwkv-behavior-lab rwkv-behavior-lab--results",
    )
    return f"{document}\n{custom_script}"


def behavior_lab_result_to_dict(result: BehaviorLabResult) -> dict[str, Any]:
    return asdict(result)


def _result_graph(result: BehaviorLabResult) -> str:
    plot_width = _GRAPH_WIDTH - _GRAPH_LEFT - _GRAPH_RIGHT
    plot_height = _GRAPH_HEIGHT - _GRAPH_TOP - _GRAPH_BOTTOM
    series: list[tuple[str, str, bool, list[tuple[float, float]]]] = []
    maximum_elapsed = 1.0
    for scenario in result.scenarios:
        color = scenario.color or "#1976d2"
        observation_points: list[tuple[float, float]] = []
        for observation in scenario.observations:
            prediction = observation.prediction_for(result.focal_card_id)
            if prediction is None or not math.isfinite(prediction):
                continue
            elapsed = max(0.0, observation.timestamp_seconds - result.baseline_timestamp_seconds)
            maximum_elapsed = max(maximum_elapsed, elapsed)
            observation_points.append((elapsed, prediction))
        series.append((scenario.name, color, False, observation_points))
        for review in scenario.reviews:
            if review.card_id != result.focal_card_id or not review.curve_points:
                continue
            offset = review.timestamp_seconds - result.baseline_timestamp_seconds
            points = [
                (max(0.0, offset + point.elapsed_seconds), point.probability)
                for point in review.curve_points
            ]
            maximum_elapsed = max(maximum_elapsed, *(value[0] for value in points))
            series.append((f"{scenario.name} curve", color, True, points))

    x_max = math.log1p(maximum_elapsed)

    def x_scale(elapsed: float) -> float:
        return _GRAPH_LEFT + plot_width * math.log1p(max(0.0, elapsed)) / x_max

    def y_scale(probability: float) -> float:
        return _GRAPH_TOP + plot_height * (1.0 - min(1.0, max(0.0, probability)))

    lines: list[str] = []
    legend: list[str] = []
    seen_legend: set[tuple[str, bool]] = set()
    for name, color, dashed, points in series:
        if not points:
            continue
        coords = " ".join(f"{x_scale(x):.2f},{y_scale(y):.2f}" for x, y in points)
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        if len(points) == 1:
            x, y = points[0]
            lines.append(
                f'<circle cx="{x_scale(x):.2f}" cy="{y_scale(y):.2f}" r="4" fill="{html.escape(color)}" />'
            )
        else:
            lines.append(
                f'<polyline points="{coords}" fill="none" stroke="{html.escape(color)}" '
                f'stroke-width="2.2"{dash} />'
            )
        key = (name, dashed)
        if key not in seen_legend:
            seen_legend.add(key)
            legend.append(
                f'<span><i style="background:{html.escape(color)}"></i>{html.escape(name)}</span>'
            )
    grid: list[str] = []
    for percent in (0, 25, 50, 75, 100):
        y = y_scale(percent / 100)
        grid.append(
            f'<line x1="{_GRAPH_LEFT}" y1="{y:.2f}" x2="{_GRAPH_WIDTH - _GRAPH_RIGHT}" y2="{y:.2f}" class="grid" />'
            f'<text x="{_GRAPH_LEFT - 8}" y="{y + 4:.2f}" text-anchor="end">{percent}%</text>'
        )
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        transformed = x_max * fraction
        elapsed = math.expm1(transformed)
        x = _GRAPH_LEFT + plot_width * fraction
        grid.append(
            f'<line x1="{x:.2f}" y1="{_GRAPH_TOP}" x2="{x:.2f}" y2="{_GRAPH_HEIGHT - _GRAPH_BOTTOM}" class="grid" />'
            f'<text x="{x:.2f}" y="{_GRAPH_HEIGHT - 17}" text-anchor="middle">{html.escape(_format_duration(elapsed))}</text>'
        )
    return f"""
<svg class="timeline-chart" viewBox="0 0 {_GRAPH_WIDTH} {_GRAPH_HEIGHT}" role="img"
 aria-label="Retrievability over simulated time"><g class="grid-lines">{"".join(grid)}</g>
 <g class="series">{"".join(lines)}</g></svg><div class="legend">{"".join(legend)}</div>
"""


def _focal_observation_rows(result: BehaviorLabResult) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for scenario in result.scenarios:
        for observation in scenario.observations:
            prediction = observation.prediction_for(result.focal_card_id)
            if prediction is None:
                continue
            deltas = dict(observation_deltas(result, scenario, observation))
            rows.append(
                (
                    html.escape(scenario.name),
                    html.escape(observation.label),
                    _format_duration(
                        observation.timestamp_seconds - result.baseline_timestamp_seconds
                    ),
                    _format_probability(prediction),
                    _delta_cell(deltas.get(result.focal_card_id)),
                )
            )
    return rows


def _curve_rows(result: BehaviorLabResult) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    targets = (60.0, 3600.0, 86_400.0, 604_800.0, 2_592_000.0)
    for scenario in result.scenarios:
        for review in scenario.reviews:
            points = {
                round(point.elapsed_seconds): point.probability for point in review.curve_points
            }
            values = [_nearest_curve_probability(points, target) for target in targets]
            rows.append(
                (
                    html.escape(scenario.name),
                    html.escape(review.label),
                    _format_probability(review.prediction_before_answer),
                    *(_format_probability(value) if value is not None else "—" for value in values),
                )
            )
    return rows


def _impact_rows(result: BehaviorLabResult) -> list[tuple[str, ...]]:
    cards = {card.card_id: card for card in result.cards}
    final_observations = [
        scenario.observations[-1] if scenario.observations else None
        for scenario in result.scenarios
    ]
    rows: list[tuple[str, ...]] = []
    for card_id in sorted(cards)[:200]:
        cells: list[str] = []
        for scenario, observation in zip(result.scenarios, final_observations, strict=True):
            if observation is None:
                cells.append("—")
                continue
            delta = dict(observation_deltas(result, scenario, observation)).get(card_id)
            cells.append(_delta_cell(delta, heat=True))
        rows.append(
            (
                str(card_id),
                html.escape(cards[card_id].relation),
                *cells,
            )
        )
    if len(cards) > 200:
        rows.append((f"… {len(cards) - 200:,} more", "", *("" for _ in result.scenarios)))
    return rows


def _cohort_rows(result: BehaviorLabResult) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for scenario in result.scenarios:
        for observation in scenario.observations:
            summary = cohort_delta_summary(result, scenario, observation)
            if summary is None:
                continue
            rows.append(
                (
                    html.escape(scenario.name),
                    html.escape(observation.label),
                    f"{int(summary['count']):,}",
                    _format_delta(float(summary["mean"])),
                    _format_delta(float(summary["median"])),
                    _format_delta(float(summary["minimum"])),
                    _format_delta(float(summary["maximum"])),
                )
            )
    return rows


def _event_rows(result: BehaviorLabResult) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for scenario in result.scenarios:
        for review in scenario.reviews:
            rows.append(
                (
                    html.escape(scenario.name),
                    html.escape(review.label),
                    str(review.card_id),
                    str(review.rating),
                    html.escape(review.review_context),
                    f"{review.duration_seconds:g}s",
                    _format_duration(review.elapsed_seconds),
                    _format_probability(review.prediction_before_answer),
                )
            )
    return rows


def _table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    if not rows:
        return render_notice("No matching results.")
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="rwkv-table-wrap rwkv-behavior-table-wrap"><table class="rwkv-data-table rwkv-behavior-table"><thead><tr>{header_html}</tr></thead><tbody>{body}</tbody></table></div>'


def _nearest_curve_probability(
    points: Mapping[int, float],
    target: float,
) -> float | None:
    if not points:
        return None
    key = min(points, key=lambda value: abs(value - target))
    return points[key] if abs(key - target) <= max(1.0, target * 0.05) else None


def _format_probability(value: float) -> str:
    return "—" if not math.isfinite(value) else f"{value * 100:.2f}%"


def _format_delta(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def _delta_cell(value: float | None, *, heat: bool = False) -> str:
    if value is None:
        return "—"
    rendered = _format_delta(value)
    if not heat:
        return rendered
    strength = min(1.0, abs(value) / 0.10)
    if value >= 0:
        color = f"rgba(33, 150, 243, {0.08 + strength * 0.34:.3f})"
    else:
        color = f"rgba(255, 152, 0, {0.08 + strength * 0.34:.3f})"
    return f'<span class="heat" style="background:{color}">{rendered}</span>'


def _format_duration(seconds: float) -> str:
    value = max(0.0, float(seconds))
    if value < 60:
        return f"{value:.0f}s"
    if value < 3600:
        return f"{value / 60:.1f}m"
    if value < 86_400:
        return f"{value / 3600:.1f}h"
    if value < 31_536_000:
        return f"{value / 86_400:.1f}d"
    return f"{value / 31_536_000:.1f}y"


def _script_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _field(field: ModalField, *wrapper_classes: str) -> str:
    classes = " ".join(("rwkv-behavior-field", *wrapper_classes))
    return f'<div class="{html.escape(classes, quote=True)}">{render_field(field)}</div>'


def _local_button(
    label: str,
    button_id: str,
    *,
    variant: Literal["primary", "secondary", "quiet", "destructive"] = "secondary",
) -> str:
    return render_button(
        ModalButton(
            label,
            None,
            variant=variant,
            button_id=button_id,
        )
    )
