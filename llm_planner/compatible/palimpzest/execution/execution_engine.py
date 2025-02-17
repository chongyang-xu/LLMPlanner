# ported from palimpzest

from llm_planner.compatible.palimpzest.core.data.datasources import DataSource, ValidationDataSource
from llm_planner.compatible.palimpzest.datamanager.datamanager import DataDirectory

from llm_planner.compatible.palimpzest.sets import Dataset
from llm_planner.compatible.palimpzest.policy import Policy

from llm_planner.compatible.palimpzest.constants import Model, OptimizationStrategy

from llm_planner.compatible.palimpzest.utils.model_helpers import get_models

class ExecutionEngine:
    def __init__(
        self,
        datasource: DataSource,
        num_samples: int = float("inf"),
        scan_start_idx: int = 0,
        nocache: bool = True,  # NOTE: until we properly implement caching, let's set the default to True
        include_baselines: bool = False,
        min_plans: int | None = None,
        verbose: bool = False,
        available_models: list[Model] | None = None,
        allow_bonded_query: bool = True,
        allow_conventional_query: bool = False,
        allow_model_selection: bool = True,
        allow_code_synth: bool = True,
        allow_token_reduction: bool = False,
        allow_rag_reduction: bool = True,
        allow_mixtures: bool = True,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.PARETO,
        max_workers: int | None = None,
        num_workers_per_plan: int = 1,
        *args,
        **kwargs,
    ) -> None:
        self.num_samples = num_samples
        self.scan_start_idx = scan_start_idx
        self.nocache = nocache
        if not self.nocache:
            raise NotImplementedError("Caching is not yet implemented! Please set nocache=True.")
        self.include_baselines = include_baselines
        self.min_plans = min_plans
        self.verbose = verbose
        self.available_models = available_models
        if self.available_models is None or len(self.available_models) == 0:
            self.available_models = get_models(include_vision=True)
        if self.verbose:
            print("Available models: ", self.available_models)
        self.allow_bonded_query = allow_bonded_query
        self.allow_conventional_query = allow_conventional_query
        self.allow_model_selection = allow_model_selection
        self.allow_code_synth = allow_code_synth
        self.allow_token_reduction = allow_token_reduction
        self.allow_rag_reduction = allow_rag_reduction
        self.allow_mixtures = allow_mixtures
        self.optimization_strategy = optimization_strategy
        self.max_workers = max_workers
        self.num_workers_per_plan = num_workers_per_plan

        self.datadir = DataDirectory()

        # datasource; should be set by execute() with call to get_datasource()
        self.datasource = datasource
        self.using_validation_data = isinstance(self.datasource, ValidationDataSource)


    def execute(self, dataset: Dataset, policy: Policy):
        """
        Execute the workload specified by the given dataset according to the policy provided by the user.
        """
        raise NotImplementedError("Abstract method to be overwritten by sub-classes")
