# synthetic_twin_agents.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from edsl import Agent, AgentList
from .mappings import MAPPINGS


def create_synthetic_twins(
    df: pd.DataFrame,
    logger: Optional[Callable[[str], None]] = None,
) -> Tuple[AgentList, List[str]]:
    """
    Build synthetic twin agents from a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input rows (Prolific participants).
    logger : Optional[Callable[[str], None]]
        Optional logging function for errors.

    Returns
    -------
    Tuple[AgentList, List[str]]
        agents : AgentList of created Agent objects
        errors : error messages for failed rows
    """
    agents_py: List[Agent] = []
    errors: List[str] = []

    for idx in range(len(df)):
        try:
            traits, name, template, instruction = _build_agent_payload(
                df, idx
            )
            agent = Agent(
                traits=traits,
                name=name,
                traits_presentation_template=template,
                instruction=instruction,
            )
            agents_py.append(agent)
        except Exception as exc:  # noqa: BLE001
            msg = (
                f"row {idx} "
                f"(PROLIFIC_PID={_safe_str(df.iloc[idx].get('PROLIFIC_PID'))}): {exc}"
            )
            errors.append(msg)
            if logger:
                logger(msg)

    return AgentList(agents_py), errors


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------

def _build_agent_payload(
    df: pd.DataFrame,
    index: int,
) -> Tuple[Dict[str, Any], str, str, str]:
    """Return traits, name, template, instruction for one participant row."""
    row = df.iloc[index]

    def get_map(col: str, raw: Any) -> str:
        """Map raw value using mappings; fallback 'Unknown'."""
        if pd.isna(raw):
            return f"Unknown {col}"
        value = _coerce_int_if_numeric(raw)
        return MAPPINGS.get(col, {}).get(value, f"Unknown {col}")

    def lower_first(text: Optional[str]) -> Optional[str]:
        """Lowercase the first character of a string."""
        if not text or not isinstance(text, str):
            return text
        return text[0].lower() + text[1:]

    def list_from_prefix(prefix: str, end_inclusive: int) -> List[str]:
        """Map all columns with a common prefix (e.g., races_1..7)."""
        values: List[str] = []
        for i in range(1, end_inclusive + 1):
            key = f"{prefix}{i}"
            if key in row and pd.notna(row[key]):
                mapped = get_map(key, row[key])
                if isinstance(mapped, str) and mapped.lower() != "unknown":
                    values.append(lower_first(mapped) or mapped)
        return values

    # Handle dataset typo.
    if "political_orientaton" in row.index:
        pol_orient_key = "political_orientaton"
    elif "political_orientation" in row.index:
        pol_orient_key = "political_orientation"
    else:
        pol_orient_key = "political_orientaton"

    # Core traits.
    age = _coerce_int(row.get("age"))
    children_n = _coerce_int(row.get("children") or 0)
    state_val = get_map("state", row.get("state"))
    country_val = (
        "U.S."
        if pd.notna(row.get("state"))
        else get_map("country", row.get("country"))
    )

    ethnicity = ", ".join(list_from_prefix("races_", 7)) or "not specified"
    devices_used = (
        ", ".join(list_from_prefix("device_used_to_buy_", 4))
        or "unspecified devices"
    )
    brand_prefs = (
        ", ".join(list_from_prefix("brands_type_pref_", 6))
        or "no specific preferences"
    )

    # Party strength.
    rep_strength = get_map("republican_strength", row.get("republican_strength"))
    dem_strength = get_map("democrat_strength", row.get("democrat_strength"))
    pol_strength_key, pol_strength_val = _first_known(
        [("republican_strength", rep_strength), ("democrat_strength", dem_strength)]
    )

    # Voting.
    voted = _coerce_int(row.get("voted"))
    vote_for = _coerce_int(row.get("vote_for"))

    traits: Dict[str, Any] = {
        "prolific_pid": row.get("PROLIFIC_PID"),
        "age": age,
        "gender": lower_first(get_map("gender", row.get("gender"))),
        "state": state_val,
        "country": country_val,
        "ethnicity": ethnicity,
        "marital_status": (
            get_map("marital_status", row.get("marital_status")) or ""
        ).lower(),
        "children": (
            "no children"
            if children_n == 0
            else f"{children_n} {'child' if children_n == 1 else 'children'}"
        ),
        "employment_status": lower_first(
            get_map("employment_status", row.get("employment_status"))
        ),
        "education_level": lower_first(
            get_map("education_level", row.get("education_level"))
        ),
        "household_income": (
            get_map("household_income", row.get("household_income")) or ""
        ).lower(),
        "political_orientation": get_map(pol_orient_key, row.get(pol_orient_key)),
        "political_ideology": get_map(
            "political_Ideology_1", row.get("political_Ideology_1")
        ),
        "shopping_frequency": (
            get_map("shopping_freq", row.get("shopping_freq")) or ""
        ).lower(),
        "monthly_spend": (
            get_map("monthly_spend", row.get("monthly_spend")) or ""
        ).lower(),
        "devices_used": devices_used,
        "brand_preferences": brand_prefs,
        "social_media_influence": (
            get_map("social_m_influence", row.get("social_m_influence")) or ""
        ).lower(),
        "eco_friendly_importance": (
            get_map("eco_friendly_imp", row.get("eco_friendly_imp")) or ""
        ).lower(),
        # Big Five raw scores (pass-through).
        "extraversion_score": row.get("extraversion_score"),
        "agreeableness_score": row.get("agreeableness_score"),
        "conscientiousness_score": row.get("conscientiousness_score"),
        "neuroticism_score": row.get("neuroticism_score"),
        "openness_score": row.get("openness_score"),
    }

    if pol_strength_key and not traits.get(
        pol_strength_key, ""
    ).lower().startswith("unknown"):
        traits[pol_strength_key] = pol_strength_val

    # Voting text fields.
    if voted == 1:
        traits["vote"] = get_map("voted", voted)
        traits["vote_for"] = get_map("vote_for", vote_for)
    elif voted == 2:
        traits["vote"] = get_map("voted", voted)
        traits["vote_for"] = "No one"

    # Prompt fields.
    pol_orient_num = _coerce_int(row.get(pol_orient_key))
    pol_ideology_num = _coerce_int(row.get("political_Ideology_1"))

    traits.update(
        {
            "political_orientation_prompt": _fmt_political_orientation(
                pol_orient_num
            ),
            "political_ideology_prompt": _fmt_political_ideology(pol_ideology_num),
            "political_strength_prompt": _fmt_political_strength(
                row, pol_orient_num
            ),
            "voting_behavior_prompt": _fmt_voting_behavior(voted, vote_for),
            "social_media_influence_prompt": _fmt_social_media_influence(
                _coerce_int(row.get("social_m_influence"))
            ),
            "eco_friendly_importance_prompt": _fmt_eco_friendly_importance(
                _coerce_int(row.get("eco_friendly_imp"))
            ),
        }
    )

    # Presentation text.
    template = (
        f"You are {traits['age']} years old, identifying as "
        f"{traits['gender']}, living in {traits['state']}, "
        f"{traits['country']}. Your ethnicity is {traits['ethnicity']}, "
        f"and you are {traits['marital_status']}, with "
        f"{traits['children']}. You are {traits['employment_status']} "
        f"and have attained {traits['education_level']}. Your household "
        f"income is {traits['household_income']}. "
        f"{traits['political_orientation_prompt']} "
        f"{traits['political_ideology_prompt']} "
        f"{traits['political_strength_prompt']} "
        f"{traits.get('voting_behavior_prompt', '')} "
        f"As an online shopper, you shop {traits['shopping_frequency']} "
        f"and spend {traits['monthly_spend']} per month. You primarily "
        f"use {traits['devices_used']} for purchases, favoring "
        f"{traits['brand_preferences']} brands. "
        f"{traits['social_media_influence_prompt']} "
        f"{traits['eco_friendly_importance_prompt']}\n\n"
        f"Your personality profile is characterized by:\n"
        f" - Extraversion: {traits['extraversion_score']}\n"
        f" - Agreeableness: {traits['agreeableness_score']}\n"
        f" - Conscientiousness: {traits['conscientiousness_score']}\n"
        f" - Neuroticism: {traits['neuroticism_score']}\n"
        f" - Openness: {traits['openness_score']}\n\n"
    )

    instruction = _instruction_block()
    name = _safe_str(row.get("PROLIFIC_PID"))

    return traits, name, template, instruction


# -------------------------------------------------------------------
# Formatting helpers
# -------------------------------------------------------------------

def _instruction_block() -> str:
    """Return the fixed experiment instruction block."""
    return (
        "***IMPORTANT INSTRUCTIONS:***\n"
        "You are a synthetic twin participating in a simulated online "
        "shopping experiment hosted on the Prolific platform. Participants "
        "on Prolific are incentivized by monetary payment, motivating them "
        "to balance speed and quality while completing tasks.\n\n"
        "Each ad includes three images, a title, and a textual description. "
        "Your task is to evaluate these ads based on your assigned "
        "personality traits, preferences, and values.\n\n"
        "**Key Behavioral Guidelines**:\n"
        "- You are incentivized to complete the task efficiently, but you "
        "must provide thoughtful evaluations that reflect your assigned "
        "traits.\n"
        "- You may rely on noticeable elements, like prominent images or key "
        "phrases, especially when ads feel repetitive.\n"
        "- Your attention might fluctuate as you progress, leading to less "
        "detailed evaluations for later ads.\n"
        "- Your responses must reflect a balance between following "
        "instructions carefully and completing the task in a timely "
        "manner.\n\n"
        "**Specific Instructions for Personality Traits**:\n"
        "You must use your assigned personality trait levels to guide your "
        "evaluation. Specifically:\n\n"
        "1. Extraversion:\n"
        "   - If you have high Extraversion, prioritize elements that "
        "emphasize social engagement, fun, or excitement in the ad.\n"
        "   - If you have low Extraversion, focus on practicality and avoid "
        "overvaluing overly social or flashy aspects.\n"
        "2. Agreeableness:\n"
        "   - If you have high Agreeableness, look for signals of warmth, "
        "empathy, and positivity in the ad.\n"
        "   - If you have low Agreeableness, evaluate the ad critically, "
        "without being influenced by attempts to appeal emotionally.\n"
        "3. Conscientiousness:\n"
        "   - If you have high Conscientiousness, assess how detailed, "
        "accurate, and organized the ad is. Look for well-structured "
        "information.\n"
        "   - If you have low Conscientiousness, focus on overall "
        "impressions without getting too caught up in fine details or "
        "structure.\n"
        "4. Neuroticism:\n"
        "   - If you have high Neuroticism, consider whether the ad reduces "
        "uncertainty or worry. Look for reassuring or calming elements.\n"
        "   - If you have low Neuroticism, focus on practical features "
        "without being overly concerned about potential risks.\n"
        "5. Openness:\n"
        "   - If you have high Openness, evaluate the ad’s creativity, "
        "originality, and appeal to curiosity. Look for innovative or "
        "unique features.\n"
        "   - If you have low Openness, prioritize straightforward, familiar, "
        "and functional aspects of the ad.\n\n"
        "**Evaluation Criteria**:\n"
        "For each ad, you will answer six 5-point Likert-scale questions. "
        "Your scores should reflect:\n"
        "1. How well the ad aligns with your assigned personality traits.\n"
        "2. A balanced assessment if traits conflict (e.g., high Openness "
        "encouraging creativity vs. high Conscientiousness valuing "
        "structure).\n"
        "3. Efficiency and quality, consistent with typical Prolific "
        "participants incentivized by payment.\n\n"
        "**Final Reminder**:\n"
        "You are a synthetic twin designed to reflect realistic participant "
        "behavior. Be consistent with your assigned personality traits while "
        "balancing thoughtful evaluation with timely completion."
    )


def _fmt_political_orientation(value: Optional[int]) -> str:
    """Return political orientation text."""
    prompts = {
        1: (
            "You identify as a Republican, reflecting alignment with "
            "conservative political ideologies."
        ),
        2: (
            "You identify as a Democrat, reflecting alignment with "
            "progressive political ideologies."
        ),
        3: (
            "You identify as an Independent, indicating a preference for "
            "policies that may transcend traditional party lines."
        ),
        5: (
            "You do not identify with any specific political orientation, "
            "indicating no particular preference for major political "
            "ideologies."
        ),
    }
    return prompts.get(
        value,
        "You do not identify with any specific political orientation, "
        "indicating no particular preference for major political ideologies.",
    )


def _fmt_political_ideology(value: Optional[int]) -> str:
    """Return political ideology text."""
    prompts = {
        1: (
            "Your political views are extremely liberal, prioritizing "
            "progressive and transformative social policies."
        ),
        2: (
            "Your political views are liberal, favoring policies that "
            "emphasize equality, inclusivity, and progress."
        ),
        3: (
            "Your political views are moderately liberal, indicating a "
            "balanced approach toward progressive ideals."
        ),
        4: "You identify as centrist, reflecting a pragmatic, neutral stance.",
        5: (
            "Your political views are moderately conservative, indicating a "
            "preference for traditional values with some openness to change."
        ),
        6: (
            "Your political views are conservative, favoring traditional "
            "values and limited government intervention."
        ),
        7: (
            "Your political views are extremely conservative, emphasizing "
            "deeply traditional and preservationist principles."
        ),
        0: (
            "You decline to specify your political ideology, leaving your "
            "views undefined."
        ),
    }
    return prompts.get(value, "You have an undefined political ideology.")


def _fmt_political_strength(
    row: pd.Series,
    pol_orientation_value: Optional[int],
) -> str:
    """Return party-strength text."""
    if pol_orientation_value == 1:
        level = _coerce_int(row.get("republican_strength"))
        prompts = {
            1: (
                "You consider yourself a strong Republican, deeply aligned "
                "with the party's principles and policies."
            ),
            2: (
                "You consider yourself a not very strong Republican, showing "
                "some alignment with the party's principles but with nuanced "
                "perspectives."
            ),
        }
        return prompts.get(level, "")

    if pol_orientation_value == 2:
        level = _coerce_int(row.get("democrat_strength"))
        prompts = {
            1: (
                "You consider yourself a strong Democrat, firmly aligned with "
                "the party's progressive values and policies."
            ),
            2: (
                "You consider yourself a not very strong Democrat, supporting "
                "the party's values with some reservations or alternative "
                "perspectives."
            ),
        }
        return prompts.get(level, "")

    if pol_orientation_value == 3:
        return (
            "You identify as Independent, reflecting an independent "
            "political view."
        )

    return "Your political strength and orientation are undefined."


def _fmt_voting_behavior(
    voted: Optional[int],
    vote_for: Optional[int],
) -> str:
    """Return voting behavior text."""
    if voted == 1:
        return {
            1: (
                "You voted in the 2024 presidential election for Donald Trump, "
                "reflecting alignment with Republican values."
            ),
            2: (
                "You voted in the 2024 presidential election for Kamala "
                "Harris, reflecting alignment with Democratic values."
            ),
        }.get(vote_for, "You voted in the 2024 presidential election.")

    if voted == 2:
        return "You did not vote in the 2024 presidential elections."

    return "Your voting behavior is undefined."


def _fmt_social_media_influence(value: Optional[int]) -> str:
    """Return social-media influence text."""
    prompts = {
        1: (
            "Social media has no influence on your decision-making process "
            "when you buy online."
        ),
        2: (
            "Social media influences your decision-making process a little "
            "when you buy online."
        ),
        3: (
            "Social media somewhat influences your decision-making process "
            "when you buy online."
        ),
        4: (
            "Social media has quite a bit of influence on your decision-"
            "making process when you buy online."
        ),
        5: (
            "Social media has a strong influence on your decision-making "
            "process when you buy online."
        ),
    }
    return prompts.get(value, "Unknown influence of social media.")


def _fmt_eco_friendly_importance(value: Optional[int]) -> str:
    """Return eco-friendliness importance text."""
    prompts = {
        1: (
            "Eco-friendliness plays no role in your decision-making process "
            "when choosing products."
        ),
        2: (
            "Eco-friendliness has a minor influence on your decision-making "
            "process when choosing products."
        ),
        3: (
            "Eco-friendliness moderately influences your decision-making "
            "process when choosing products."
        ),
        4: (
            "Eco-friendliness is an important factor in your decision-making "
            "process when choosing products."
        ),
        5: (
            "Eco-friendliness is a key consideration in your decision-making "
            "process when choosing products."
        ),
    }
    return prompts.get(value, "Unknown eco-friendliness importance.")


# -------------------------------------------------------------------
# Tiny primitives
# -------------------------------------------------------------------

def _coerce_int(val: Any) -> Optional[int]:
    """Try to turn val into int; return None if impossible."""
    if pd.isna(val):
        return None
    try:
        if isinstance(val, float) and val.is_integer():
            return int(val)
        return int(val)
    except (ValueError, TypeError):
        return None


def _coerce_int_if_numeric(val: Any) -> Any:
    """If val is numeric-like, cast to int; else leave as-is."""
    coerced = _coerce_int(val)
    return coerced if coerced is not None else val


def _first_known(
    pairs: Iterable[Tuple[str, str]],
) -> Tuple[Optional[str], Optional[str]]:
    """Return first (key, value) where value is not 'Unknown'."""
    for key, value in pairs:
        if isinstance(value, str) and not value.lower().startswith("unknown"):
            return key, value
    return None, None


def _safe_str(value: Any) -> str:
    """Safe string cast that handles NaN/None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


if __name__ == "__main__":  # pragma: no cover
    # Example usage (commented):
    # df = pd.read_csv("participants.csv")
    # agents, errs = create_synthetic_twins(df)
    # print(f"Created: {len(agents)}, Errors: {len(errs)}")
    pass