# ported from palimpzest

import time

from llm_planner.compatible.palimpzest.execution.execution_engine import ExecutionEngine

from llm_planner.compatible.palimpzest.sets import Dataset
from llm_planner.compatible.palimpzest.policy import Policy


class NoSentinelSequentialSingleThreadExecution(ExecutionEngine):

    def execute(self, dataset: Dataset, policy: Policy):
        pass
