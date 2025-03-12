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

from llm_planner.compatible.palimpzest.execution.operators import LLMFilterHAL, FunctionFilterHAL, LLMConvertHAL


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

        # compute depends_on field for every node
        self._compute_depends_on( dataset_nodes )

        # for idx, n in enumerate( dataset_nodes ):
        #     print(idx, n.schema.field_names())

        # print("Dataset Nodes: ")
        # for idx, n in enumerate( dataset_nodes ):
        #    print(idx, type(n).__name__)

        llm_planner_t, template_input = self.dataset_nodes_to_template(dataset_nodes)
        llm_planner_t.start(template_input)

        return llm_planner_t.results, ExecutionStats()


    def _compute_depends_on(self, all_nodes):
        for idx, n in enumerate( all_nodes ):
            if hasattr(n, '_depends_on'):
                pass
                # print(idx, n._depends_on)
            else:
                assert idx ==0, "assume idx-0 is  DataSource"

    def _extract_input_fields_from_node(self, node):
        """Returns the set of input fields needed to execute a physical operator."""
        if node is None:
            return None

        if not hasattr(node, '_depends_on'):
            depends_on_fields = None
        else:
            depends_on_fields = (
                [field.split(".")[-1] for field in node._depends_on]
                if node._depends_on is not None and len(node._depends_on) > 0
                else None
            )

        input_fields = (
            node.schema.field_names()
            if depends_on_fields is None
            else [field for field in node.schema.field_names() if field in depends_on_fields]
        )

        is_image_conversion = False

        return input_fields

    def dataset_nodes_to_template(self, dataset_nodes):
        self.filter_generator = OpenAIGenerator(PromptStrategy.COT_BOOL)
        self.convert_generator = OpenAIGenerator(PromptStrategy.COT_QA)

        llm_planner_t = Template().input()
        template_input = None

        print('---')
        for idx, n in enumerate(dataset_nodes):
            opt = dataset_nodes[idx]
            ipt = None if idx < 1 else dataset_nodes[idx-1]

            opt_schema = opt.schema
            ipt_schema = ipt.schema if ipt is not None else None

            # generators.py line: 400
            input_fields = self._extract_input_fields_from_node(ipt)
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
                print(idx, f"{type(n).__name__} -> input -> {opt_schema.__name__} ")
            # elif isinstance(n, TextFileDirectorySource):
            elif n._filter is not None:
                ft_input_fields = input_fields
                ft_output_fields = output_fields

                if n._filter.filter_condition is not None:
                    filter_str = n._filter.filter_condition

                    llm_filter = LLMFilterHAL(filter_str, ft_input_fields, self.filter_generator)
                    llm_planner_t.filter(llm_filter)
                    print(idx, f"{ipt_schema.__name__} -> LLMFilterHAL -> {opt_schema.__name__}")

                elif n._filter.filter_fn is not None:
                    llm_filter = FunctionFilterHAL(n._filter.filter_fn)
                    llm_planner_t.filter(llm_filter)
                    print(idx, f"{ipt_schema.__name__} -> FunctionFilterHAL -> {opt_schema.__name__}")

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

                is_image_conversion = False
                for field_n in n.schema.field_names():
                    if field_n.split(".")[-1] in n._depends_on:
                        field = getattr(n.schema, field_n)
                        if field.is_image_field:
                            is_image_conversion = True
                            break

                if is_image_conversion:
                    self.convert_generator = OpenAIGenerator(PromptStrategy.COT_QA_IMAGE)

                cv_ipt_schema = ipt_schema
                cv_opt_schema = opt_schema

                cv_input_fields = input_fields
                cv_output_fields = output_fields

                convert = LLMConvertHAL(cv_input_fields, cv_output_fields, cv_ipt_schema, cv_opt_schema, self.convert_generator)

                llm_planner_t.map(convert)

                print(idx, f"{ipt_schema.__name__} -> LLMConvertHAL -> {opt_schema.__name__}")

            else:
                assert False, f"Un-recognized node : {type(n)}"

        print('---')
        llm_planner_t.collect()
        return  llm_planner_t, template_input
