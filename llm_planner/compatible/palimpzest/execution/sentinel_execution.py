# ported from palimpzest

import time

from llm_planner.compatible.palimpzest.constants import PromptStrategy

from llm_planner.compatible.palimpzest.sets import Dataset

from llm_planner.compatible.palimpzest.policy import Policy

from llm_planner.compatible.palimpzest.execution.execution_engine import ExecutionEngine
from llm_planner.compatible.palimpzest.execution.generators import OpenAIGenerator

from llm_planner.message import Message
from llm_planner.templates.template import Template
from llm_planner.templates.template import kv_get, kv_put

from llm_planner.compatible.palimpzest.core.data.datasources import DataSource, TextFileDirectorySource

from llm_planner.compatible.palimpzest.execution.operators import LLMFilterHAL, LLMConvertHAL


class SentinelExecutionEngine(ExecutionEngine):

    def execute(self, dataset: Dataset, policy: Policy):
       pass

class PipelinedParallelPlanExecutor:
    def execute(self, dataset: Dataset, policy: Policy):
       pass


class Record:
    def get_fields(self):
        return ["null"]

    def __getitem__(self, key):
        """Get the value for a given key using square brackets."""
        return "null"

class ExecutionStats:
    plan_strs = {"null": "null"}

    def to_json(self):
        pass


class PipelinedParallelSentinelExecution(SentinelExecutionEngine, PipelinedParallelPlanExecutor):
    """
    This class performs sentinel execution while executing plans in a pipelined, parallel fashion.
    """
    def execute(self, dataset: Dataset, policy: Policy):

        print("NoSentinelSequentialSingleThreadExecution:Execute")

        dataset_nodes = []
        dn = dataset.copy()

        while isinstance(dn, Dataset):
            dataset_nodes.append(dn)
            dn = dn._source
        dataset_nodes.append(dn)
        dataset_nodes = list(reversed(dataset_nodes))


        # remove unnecessary convert if output schema from data source scan matches
        # input schema for the next operator
        if len(dataset_nodes) > 1 and dataset_nodes[0].schema.get_desc() == dataset_nodes[1].schema.get_desc():
            dataset_nodes = [dataset_nodes[0]] + dataset_nodes[2:]
            if len(dataset_nodes) > 1:
                dataset_nodes[1]._source = dataset_nodes[0]

        llm_planner_t, template_input = self.dataset_nodes_to_template(dataset_nodes)
        llm_planner_t.start(template_input)

        return llm_planner_t.results, ExecutionStats()


    def dataset_nodes_to_template(self, dataset_nodes):
        self.filter_generator = OpenAIGenerator(PromptStrategy.COT_BOOL)
        self.convert_generator = OpenAIGenerator(PromptStrategy.COT_QA)

        llm_planner_t = Template().input()
        template_input = None

        for idx, n in enumerate(dataset_nodes):
            opt = dataset_nodes[idx]
            ipt = None if idx < 1 else dataset_nodes[idx-1]

            opt_schema = opt.schema
            ipt_schema = ipt.schema if ipt is not None else None

            # generators.py line: 400
            input_fields = ipt_schema.field_names() if ipt_schema is not None else None
            output_fields = opt_schema.field_names()

            if isinstance(n, DataSource):
                ds_input_fields = input_fields
                ds_output_fields = output_fields

                class TemplateInputFromDataSource:
                    def __init__(self, source):
                        self.source = source  # Assuming items is a list of objects

                    def __len__(self):
                        return len(self.source)

                    def __iter__(self):
                        for idx in range(len(self.source)):
                            candidate = self.source.get_item(idx)
                            yield candidate # : DataRecord

                template_input = TemplateInputFromDataSource(n)

            # elif isinstance(n, TextFileDirectorySource):
            elif n._filter is not None:
                ft_input_fields = input_fields
                ft_output_fields = output_fields

                if n._filter.filter_condition is not None:
                    filter_str = n._filter.filter_condition

                    llm_filter = LLMFilterHAL(filter_str, ft_input_fields, self.filter_generator)
                    llm_planner_t.filter(llm_filter)

                elif n._filter.filter_fn is not None:
                    filter_fn = n._filter.filter_fn
                    raise Exception("Filter type not supported.", type(n._filter))
                else:
                    raise Exception("Filter type not supported.", type(n._filter))

            elif n._group_by is not None:
                assert False
            elif n._agg_func is not None:
                assert False
            elif n._limit is not None:
                assert False
            elif n._project_cols is not None:
                assert False
            elif n._index is not None:
                assert False
            elif opt_schema != ipt_schema:
                cv_ipt_schema = ipt_schema
                cv_opt_schema = opt_schema

                cv_input_fields = input_fields
                cv_output_fields = output_fields

                convert = LLMConvertHAL(cv_input_fields, cv_output_fields, cv_ipt_schema, cv_opt_schema, self.convert_generator)

                llm_planner_t.map(convert)
            else:
                assert False, f"Un-recognized node : {type(n)}"

        llm_planner_t.collect()
        return  llm_planner_t, template_input
