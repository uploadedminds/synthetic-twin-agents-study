# scenario_list.py

"""Build an EDSL ScenarioList for product ad evaluations."""

import os
import sys

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Also add the utilities directory itself to the path
if current_dir not in sys.path:
    sys.path.append(current_dir)

from typing import Dict, List
from edsl import ScenarioList, Scenario, FileStore, QuestionLinearScale

# Import static data
from mappings import (
    STATEMENTS,
    TRAIT_PRODUCT_DESCRIPTIONS,
    PRODUCT_TITLES,
    IMAGE_UUIDS,
)


def _prefetch_images(
    image_uuids: Dict[int, List[str]],
) -> Dict[int, Dict[int, str]]:
    """Resolve UUIDs to FileStore handles for quick template use."""
    return {
        product: {
            idx: FileStore.pull(uuid) for idx, uuid in enumerate(uuids)
        }
        for product, uuids in image_uuids.items()
    }


def _build_question() -> QuestionLinearScale:
    """Create the Likert-scale question template used by scenarios."""
    return QuestionLinearScale(
        question_name="question",
        question_text=(
            "Please evaluate the effectiveness of this product ad by "
            "indicating the extent to which you agree with the following "
            "statement:\n\n"
            "{{ statement }}.\n\n"
            "The ad includes three images:\n\n"
            "1. {{ image_1 }}\n"
            "2. {{ image_2 }}\n"
            "3. {{ image_3 }}\n\n"
            "A title: {{ title }}, and a description: {{ description }}."
        ),
        question_options=[1, 2, 3, 4, 5],
        option_labels={
            1: "Strongly disagree",
            2: "Disagree",
            3: "Neither agree nor disagree",
            4: "Agree",
            5: "Strongly agree",
        },
    )


def create_scenario_list() -> ScenarioList:
    """
    Build and return a ScenarioList for all products/traits/statements.

    Returns
    -------
    ScenarioList
        Fully populated ScenarioList.
    """
    pre_fetched_images = _prefetch_images(IMAGE_UUIDS)

    scenarios = [
        Scenario(
            {
                "question_name": f"p_{product}_{trait}_item_{i + 1}",
                "image_1": pre_fetched_images[product][0],
                "image_2": pre_fetched_images[product][1],
                "image_3": pre_fetched_images[product][2],
                "title": PRODUCT_TITLES[product],
                "description": TRAIT_PRODUCT_DESCRIPTIONS[product][trait],
                "statement": statement,
            }
        )
        for product in IMAGE_UUIDS
        for trait in TRAIT_PRODUCT_DESCRIPTIONS[product]
        for i, statement in enumerate(STATEMENTS)
    ]

    _ = _build_question()  # Just ensures the template is available

    return ScenarioList(scenarios)
