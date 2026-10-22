"""Pure, serializable Settings presentation model for the Svelte frontend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .config_options import ConfigOption
from .glossary import GlossaryRenderer


def settings_pages(
    config: Mapping[str, Any],
    *,
    options: Sequence[ConfigOption],
    sections: Sequence[str],
    choices: Mapping[tuple[str, ...], Sequence[str]],
    page_intros: Mapping[str, str],
    group_intros: Mapping[tuple[str, str], str],
    subsection_intros: Mapping[tuple[str, str, str], str],
    capability_note: str,
    choice_label,
    datetime_parts,
    model_key: str,
) -> list[dict[str, Any]]:
    """Keep grouping and first-use glossary annotations identical to Settings."""

    def get_value(path):
        value = config
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
        return value

    pages = []
    for section in sections:
        glossary = GlossaryRenderer(section)
        page = {
            "name": section,
            "intro": glossary.render(page_intros[section]),
            "capability": glossary.render(capability_note) if section == "General" else "",
            "groups": [],
        }
        page_options = [option for option in options if option.section == section]
        for group_name in dict.fromkeys(option.group for option in page_options):
            group_options = [option for option in page_options if option.group == group_name]
            group = {
                "name": group_name,
                "heading": glossary.render(group_name),
                "intro": glossary.render(group_intros.get((section, group_name), "")),
                "cards": [],
            }
            for subsection in dict.fromkeys(option.subsection for option in group_options):
                intro = subsection_intros.get((section, group_name, subsection), "")
                card = {
                    "key": subsection or group_name,
                    "heading": glossary.render(subsection) if subsection else "",
                    "intro": glossary.render(intro) if intro else "",
                    "curveCard": subsection == "RWKV Forgetting Curve",
                    "options": [],
                }
                # Preserve glossary first-use ordering in the old curve help.
                if card["curveCard"]:
                    glossary.render(
                        "Enable Calculate Forgetting Curves under Performance "
                        "to use these features."
                    )
                for option in group_options:
                    if option.subsection != subsection:
                        continue
                    value = get_value(option.key_path)
                    value = not bool(value) if option.inverted else value
                    if option.value_type == "int":
                        value = int(value or 0)
                    path = "/".join(option.key_path)
                    card["options"].append(
                        {
                            "path": path,
                            "id": "config-" + path.replace("/", "-"),
                            "label": option.label,
                            "labelHtml": glossary.render(option.label),
                            "description": glossary.render(option.tooltip),
                            "kind": option.value_type,
                            "value": value,
                            "choices": [
                                {
                                    "value": str(item),
                                    "label": str(item)
                                    if path == model_key
                                    else choice_label(str(item)),
                                }
                                for item in choices.get(option.key_path, ())
                            ],
                            "dateTimeParts": datetime_parts(value)
                            if option.value_type == "datetime"
                            else None,
                            "minimum": option.minimum,
                            "maximum": option.maximum,
                            "requiresCurves": option.requires_curves,
                            "requiresV1": option.requires_v1_generation,
                            "v5UnavailableReason": option.v5_unavailable_reason,
                            "parentPath": "/".join(option.parent_key_path)
                            if option.parent_key_path
                            else None,
                            "visiblePath": "/".join(option.visible_when_key_path)
                            if option.visible_when_key_path
                            else None,
                            "restartRequired": option.restart_required,
                            "restartBadge": option.restart_badge,
                            "restartTooltip": option.restart_tooltip,
                            "rebuildRequired": option.checkpoint_rebuild_required,
                            "speedTest": option.speed_test,
                            "comparison": option.comparison,
                        }
                    )
                group["cards"].append(card)
            page["groups"].append(group)
        pages.append(page)
    return pages
