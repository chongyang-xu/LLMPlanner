# pz: b2e4d5f5198fbdbfb3b0d58b210b2e3052294528

import os
import json

import time
import numpy as np
import gradio as gr
from PIL import Image
from pathlib import Path

from llm_planner.compatible.palimpzest.sets import Dataset
from llm_planner.compatible.palimpzest.constants import Cardinality, OptimizationStrategy

from llm_planner.compatible.palimpzest.core.data.datasources import UserSource
from llm_planner.compatible.palimpzest.core.lib.schemas import Schema, Table, XLSFile

from llm_planner.compatible.palimpzest.core.lib.fields import BooleanField, Field, ImageFilepathField, ListField, NumericField, StringField
from llm_planner.compatible.palimpzest.core.elements.records import DataRecord

from llm_planner.compatible.palimpzest.policy import MaxQuality
from llm_planner.compatible.palimpzest.execution.execute import Execute
from llm_planner.compatible.palimpzest.execution.sentinel_execution import PipelinedParallelSentinelExecution

from llm_planner.compatible.palimpzest.datamanager.datamanager import DataDirectory

import io
import pandas as pd
from llm_planner.compatible.palimpzest.constants import MAX_ROWS


def xls_to_tables(candidate: dict):
    """Function used to convert a DataRecord instance of XLSFile to a Table DataRecord."""
    xls_bytes = candidate["contents"]
    sheet_names = candidate["sheet_names"]

    records = []
    for sheet_name in sheet_names:
        dataframe = pd.read_excel(io.BytesIO(xls_bytes),
                                  sheet_name=sheet_name,
                                  engine="openpyxl")

        # TODO extend number of rows with dynamic sizing of context length
        # construct data record
        record = {}
        rows = []
        for row in dataframe.values[:100]:
            row_record = [str(x) for x in row]
            rows += [row_record]
        record["rows"] = rows[:MAX_ROWS]
        record["filename"] = candidate["filename"]
        record["header"] = dataframe.columns.values.tolist()
        record["name"] = candidate["filename"].split("/")[-1] + "_" + sheet_name
        records.append(record)

    return records


class CaseData(Schema):
    """An individual row extracted from a table containing medical study data."""

    case_submitter_id = Field(desc="The ID of the case")
    age_at_diagnosis = Field(
        desc="The age of the patient at the time of diagnosis")
    race = Field(
        desc=
        "An arbitrary classification of a taxonomic group that is a division of a species.",
    )
    ethnicity = Field(
        desc=
        "Whether an individual describes themselves as Hispanic or Latino or not.",
    )
    gender = Field(desc="Text designations that identify gender.")
    vital_status = Field(desc="The vital status of the patient")
    ajcc_pathologic_t = Field(desc="The AJCC pathologic T")
    ajcc_pathologic_n = Field(desc="The AJCC pathologic N")
    ajcc_pathologic_stage = Field(desc="The AJCC pathologic stage")
    tumor_grade = Field(desc="The tumor grade")
    tumor_focality = Field(desc="The tumor focality")
    tumor_largest_dimension_diameter = Field(
        desc="The tumor largest dimension diameter")
    primary_diagnosis = Field(desc="The primary diagnosis")
    morphology = Field(desc="The morphology")
    tissue_or_organ_of_origin = Field(desc="The tissue or organ of origin")
    # tumor_code = Field(desc="The tumor code")
    filename = Field(desc="The name of the file the record was extracted from")
    study = Field(
        desc="The last name of the author of the study, from the table name",)


os.makedirs("profiling-data", exist_ok=True)

datasetid = "biofabric-medium"
workload = "medical-schema-matching"
visualize = False
verbose = False
profile = True
policy = MaxQuality()
execution_engine = PipelinedParallelSentinelExecution
"""
    executor = args.executor
    if executor == "sequential":
        execution_engine = NoSentinelSequentialSingleThreadExecution
    elif executor == "pipelined":
        execution_engine = NoSentinelPipelinedSingleThreadExecution
    elif executor == "parallel":
        execution_engine = NoSentinelPipelinedParallelExecution
    else:
        print("Executor not supported for this demo")
"""

workload = "medical-schema-matching"

plan = Dataset(datasetid, schema=XLSFile)
plan = plan.convert(Table,
                    udf=xls_to_tables,
                    cardinality=Cardinality.ONE_TO_MANY)
plan = plan.filter("The rows of the table contain the patient age")
plan = plan.convert(CaseData,
                    desc="The patient data in the table",
                    cardinality=Cardinality.ONE_TO_MANY)

start_time = time.time()
# execute pz plan
records, execution_stats = Execute(
    plan,
    policy=policy,
    nocache=True,
    optimization_strategy=OptimizationStrategy.PARETO,
    execution_engine=execution_engine,
    verbose=verbose,
)
end_time = time.time()
print("Elapsed time:", end_time - start_time)

# 498.58 s

# save statistics
if profile:
    stats_path = f"profiling-data/{workload}-profiling.json"
    execution_stats_dict = execution_stats.to_json()
    with open(stats_path, "w") as f:
        json.dump(execution_stats_dict, f)

# visualize output in Gradio
if visualize:
    from palimpzest.utils.demo_helpers import print_table

plan_str = list(execution_stats.plan_strs.values())[-1]
print(records[0])
