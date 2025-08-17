import os
from edsl import ScenarioList, AgentList, QuestionLinearScale, Model
from dotenv import load_dotenv

load_dotenv()  # By default, loads from .env in the current directory

try:
    sl = ScenarioList.pull("ddfc9685-f065-4f06-b22f-7ed5a3a691bb")
    agents = AgentList.pull("f33e3099-2757-4ac4-a626-03d949adb912")
except Exception as e:
    print(f"Error pulling scenario or agents: {e}")
    sl = ScenarioList.pull("ddfc9685-f065-4f06-b22f-7ed5a3a691bb")
    agents = AgentList.pull("f33e3099-2757-4ac4-a626-03d949adb912")
    print(f"Pulled scenario and agents: {sl} {agents}")

# (Optional) inspect what's inside
print(f"Loaded scenario: {len(sl)}")
print(f"Number of agents: {len(agents)}")



# Create the question object
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

all_responses = q.by(sl).by(agents[150:300]).by(m).run(disable_remote_inference=True, progress_bar=True, verbose=True)

all_responses.to_pandas().to_csv("../../synthetics_survey_results/gemini_2.0_flash/results_2.csv", index=False)