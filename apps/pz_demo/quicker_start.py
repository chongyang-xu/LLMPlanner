# copied from palimpzest

import pandas as pd
import llm_planner.compatible.palimpzest.datamanager.datamanager as pzdm
from llm_planner.compatible.palimpzest.policy import MaxQuality

import llm_planner.compatible.palimpzest as pz

# Dataset registration
dataset_path = "/home/cxu/ws/llm/LLMPlanner/scripts/pz/testdata/enron-tiny"
dataset_name = "enron-tiny"
pzdm.DataDirectory().register_local_directory(dataset_path, dataset_name)

# Dataset loading
dataset = pz.Dataset(dataset_name, schema=pz.TextFile)


# Schema definition for the fields we wish to compute
class Email(pz.Schema):
    """Represents an email, which in practice is usually from a text file"""
    sender = pz.Field(desc="The email address of the sender")
    subject = pz.Field(desc="The subject of the email")
    date = pz.Field(desc="The date the email was sent")


# Lazy construction of computation to filter for emails about holidays sent in July
dataset = dataset.convert(Email, desc="An email from the Enron dataset")
dataset = dataset.filter("The email was sent in July")
dataset = dataset.filter("The email is about holidays")

# Executing the compuation
policy = MaxQuality()
execution_engine = pz.PipelinedParallelSentinelExecution

results, execution_stats = pz.Execute(
    dataset,
    policy=policy,
    nocache=True,
    allow_token_reduction=False,
    allow_code_synth=False,
    execution_engine=execution_engine,
    verbose=False,
)

# Writing output to disk
output_df = pd.DataFrame([r.as_dict() for r in results
                         ])[["date", "sender", "subject"]]
output_df.to_csv("july_holiday_emails.csv")
