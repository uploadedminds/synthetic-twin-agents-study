import os
import sys

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

import pandas as pd

from edsl import ScenarioList, AgentList, QuestionLinearScale, Model
from dotenv import load_dotenv
load_dotenv()

from utilities.synthetic_twin_agents import create_synthetic_twins
from utilities.scenario_list import create_scenario_list


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
m = Model("gemini-2.0-flash", service_name = "google", temperature = 1)

first_batch = agents[:150]
second_batch = agents[150:300]
thrid_batch = agents[300:]


try:
    # First Batch Results 150 agents
    results = q.by(sl).by(first_batch).by(m).run(disable_remote_inference=True, progress_bar=True, verbose=True)
    results.to_pandas().to_csv("../../synthetics_survey_results/gemini_2.0_flash/results_1.csv", index=False)

    # Second Batch Results 150 agents
    # results = q.by(sl).by(second_batch).by(m).run(disable_remote_inference=True, progress_bar=True, verbose=True)
    # results.to_pandas().to_csv("../../synthetics_survey_results/gemini_2.0_flash/results_2.csv", index=False)

    # Third Batch Results 73 agents
    # results = q.by(sl).by(thrid_batch).by(m).run(disable_remote_inference=True, progress_bar=True, verbose=True)
    # results.to_pandas().to_csv("../../synthetics_survey_results/gemini_2.0_flash/results_3.csv", index=False)
except Exception as e:
    print(f"Error running a job: {e}")