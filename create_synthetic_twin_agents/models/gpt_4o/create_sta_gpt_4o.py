import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

import pandas as pd

from edsl import ScenarioList, AgentList, QuestionLinearScale, Model, Coop
from dotenv import load_dotenv
load_dotenv()

from utilities.synthetic_twin_agents import create_synthetic_twins
from utilities.scenario_list import create_scenario_list

coop = Coop(api_key=os.getenv("EXPECTED_PARROT_API_KEY"))


# change to FALSE to pull from local files
PULLFROM_SERVER = True
sl = None
agents = None

if PULLFROM_SERVER:
    try:
        sl = ScenarioList.pull("ddfc9685-f065-4f06-b22f-7ed5a3a691bb")
        print(f"Coop: Successfully pulled {len(sl)} scenarios from Coop server.")
    except Exception as e:
        print(f"Coop: Error pulling scenarios: {e}")
    try:
        agents = AgentList.pull("f33e3099-2757-4ac4-a626-03d949adb912")
        print(f"Coop: Successfully pulled {len(agents)} agents from Coop server.")
    except Exception as e:
        print(f"Coop: Error pulling agents: {e}")
else:
    try:
        human_participants_df = pd.read_csv('../../../data/filtered_participants_dataset.csv')
        agents, errors = create_synthetic_twins(human_participants_df)
        print(f"Local: Successfully created {len(agents)} agents.")
    except Exception as e:
        print(f"Local: Successfully created {len(human_participants_df)} scenarios from Coop server.")
    try:
        sl = create_scenario_list()
        print(f"Local: Successfully created {len(sl)} scenarios")
    except Exception as e:
        print(f"Local: Error creating scenarios: {e}")

q = QuestionLinearScale(
    question_name="question",
    question_text="""
    Please evaluate the effectiveness of this product ad by indicating the extent to which you agree with the following statement: 
    {{ statement }}.
    
    The ad includes three images:
    
    1. {{ image_1 }}
    2. {{ image_2 }}
    3. {{ image_3 }}
    
    A a title: {{ title }}, and a description: {{ description }}.
    """,
    question_options=[
        1, 2, 3, 4, 5
    ],
    option_labels={
        1: "Strongly disagree",
        2: "Disagree",
        3: "Neither agree nor disagree",
        4: "Agree",
        5: "Strongly agree"
    }
)

# Create the model object
m = Model("gpt-4o", service_name = "openai", temperature = 1)

batch_1 = agents[:150]
batch_2 = agents[150:300]
batch_3 = agents[300:]


OUT_PATH_BATCH_1 = "../../synthetics_survey_results/gpt_5/results_1_gpt_4o.csv"
OUT_PATH_BATCH_2 = "../../synthetics_survey_results/gpt_5/results_2_gpt_4o.csv"
OUT_PATH_BATCH_3 = "../../synthetics_survey_results/gpt_5/results_3_gpt_4o.csv"


try:
    # First Batch Results 150 agents
    # results = q.by(sl).by(batch_1).by(m).run(disable_remote_inference=True, progress_bar=True, verbose=True)
    # results.to_pandas().to_csv(OUT_PATH_BATCH_3, index=False)

    # results = q.by(sl).by(batch_2).by(m).run(disable_remote_inference=True, progress_bar=True, verbose=True)
    # results.to_pandas().to_csv(OUT_PATH_BATCH_2, index=False)

    results = q.by(sl).by(batch_3).by(m).run(disable_remote_inference=True, progress_bar=True, verbose=True)
    results.to_pandas().to_csv(OUT_PATH_BATCH_3, index=False)
except Exception as e:
    print(f"Error running a job: {e}")