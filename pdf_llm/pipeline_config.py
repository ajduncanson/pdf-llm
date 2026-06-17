"""
Loads a pipeline config YAML and builds a system prompt from it.

The config drives three concerns:
  - feature_priorities: ordered list of features and their importance
  - comparison_rules:   how to handle optional tiers, missing features, etc.
  - output:             desired format and structure of the response
"""

from pathlib import Path
from typing import Optional


def load_system_prompt(config_path: str) -> str:
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml is required for pipeline config: pip install pyyaml")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    return _build_system_prompt(config)


def _build_system_prompt(config: dict) -> str:
    sections = []

    # --- Feature priorities ---
    features = config.get("feature_priorities", [])
    if features:
        lines = ["## Feature Priority Order"]
        lines.append(
            "When comparing products, treat features in the following priority order "
            "(highest priority first). Give more prominence to high-priority features "
            "in your analysis and output."
        )
        for f in features:
            name = f.get("name", "")
            priority = f.get("priority", "medium")
            note = f.get("note", "")
            line = f"- {name} [{priority}]"
            if note:
                line += f": {note}"
            lines.append(line)
        sections.append("\n".join(lines))

    # --- Comparison rules ---
    rules = config.get("comparison_rules", {})
    if rules:
        lines = ["## Comparison Rules"]

        tier_handling = rules.get("tier_handling")
        if tier_handling == "expand":
            lines.append(
                "- Where a product offers optional tiers or add-ons for a feature, "
                "expand these into separate rows rather than collapsing them into one."
            )
        elif tier_handling == "summarise":
            lines.append(
                "- Where a product offers optional tiers or add-ons, summarise them "
                "in a single row with a brief note about the options available."
            )

        missing = rules.get("missing_feature")
        if missing == "mark_na":
            lines.append("- If a feature is not mentioned in a product's document, mark it as N/A.")
        elif missing == "omit":
            lines.append("- Omit rows where a feature is absent from all products being compared.")
        elif missing == "flag_for_review":
            lines.append(
                "- If a feature is absent or ambiguous, flag it explicitly for human review "
                "rather than making an assumption."
            )

        extra_rules = rules.get("extra", [])
        for rule in extra_rules:
            lines.append(f"- {rule}")

        sections.append("\n".join(lines))

    # --- Output format ---
    output = config.get("output", {})
    if output:
        lines = ["## Output Format"]

        fmt = output.get("format", "markdown")
        if fmt == "markdown":
            lines.append("- Present the comparison as a Markdown table.")
        elif fmt == "csv":
            lines.append("- Present the comparison as CSV, with a header row.")
        elif fmt == "json":
            lines.append(
                "- Present the comparison as a JSON array of objects, "
                "one object per feature with product names as keys."
            )
        elif fmt == "html":
            lines.append("- Present the comparison as an HTML table.")

        if output.get("group_by_priority"):
            lines.append("- Group rows by priority: high-priority features first, then medium, then low.")

        if output.get("highlight_best"):
            lines.append(
                "- For each feature row, identify which product offers the best outcome "
                "and indicate it clearly (e.g. bold, asterisk, or a 'Best' column)."
            )

        if output.get("include_summary"):
            lines.append(
                "- After the table, include a short summary paragraph identifying the "
                "strongest overall product and any notable trade-offs."
            )

        sections.append("\n".join(lines))

    return "\n\n".join(sections)
