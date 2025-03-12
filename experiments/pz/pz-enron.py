# ported from palimpzest/demos/simple-demo.py

import time

import gradio as gr
import pandas as pd
from tabulate import tabulate

import palimpzest as pz
from palimpzest.core.lib.schemas import TextFile
from palimpzest.core.lib.fields import Field
from palimpzest.sets import Dataset
from palimpzest.query.execution.execute import Execute


def print_table(records, cols=None, gradio=False, plan_str=None):
    records = [{
        key: record[key] for key in record.get_field_names()
    } for record in records]
    records_df = pd.DataFrame(records)
    print_cols = records_df.columns if cols is None else cols
    final_df = records_df[print_cols] if not records_df.empty else pd.DataFrame(
        columns=print_cols)

    if not gradio:
        print(tabulate(final_df, headers="keys", tablefmt="psql"))

    else:
        with gr.Blocks() as demo:
            gr.Dataframe(final_df)

            if plan_str is not None:
                gr.Textbox(value=plan_str, info="Query Plan")

        demo.launch()


class Email(TextFile):
    """Represents an email, which in practice is usually from a text file"""
    sender = Field(desc="The email address of the sender")
    subject = Field(desc="The subject of the email")


# config
policy = pz.MaxQuality()

engine, executor = "sentinel", "parallel"
assert engine == "sentinel"
assert executor == "parallel"

# dataset
dataset_id = 'enron-tiny'

emails = Dataset(dataset_id, schema=Email)
root_set = emails.filter("The email was sent in July")

# output config
cols = ["sender", "subject"]
stat_path = "profiling-data/enron-profiling.json"

verbose = False
profile = False

start_time = time.time()

records, execution_stats = Execute(
    root_set,
    policy=policy,
    nocache=True,
    allow_token_reduction=False,
    allow_code_synth=False,
    verbose=verbose,
)
end_time = time.time()
print("Elapsed time:", end_time - start_time)

# print result
print(f"Policy is: {str(policy)}")
print("Executed plan:")
plan_str = list(execution_stats.plan_strs.values())[0]
print(plan_str)

if profile:
    with open(stat_path, "w") as f:
        json.dump(execution_stats.to_json(), f)

print_table(records, cols=cols, gradio=False, plan_str=plan_str)
