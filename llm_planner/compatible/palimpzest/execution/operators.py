
from llm_planner.message import Message

from llm_planner.compatible.palimpzest.execution.util import create_data_records_from_field_answers

class LLMFilterHAL:
    def __init__(self, filter_str, input_fields, generator):
        self.filter_str = filter_str  # Store the filter string
        self.input_fields = input_fields
        self.filter_generator = generator

    def __call__(self, msg: Message):
        data_record = msg["content"]
        # Construct kwargs for generation
        gen_kwargs = {"project_cols": self.input_fields, "filter_condition": self.filter_str}

        # Generate output
        field_answers, _, generation_stats = self.filter_generator(data_record, ["passed_operator"], **gen_kwargs)

        # Compute whether the record passed the filter or not
        passed_operator = False
        if isinstance(field_answers["passed_operator"], str):
            passed_operator = "true" in field_answers["passed_operator"].lower()
        elif isinstance(field_answers["passed_operator"], bool):
            passed_operator = field_answers["passed_operator"]

        return passed_operator


class FunctionFilterHAL:
    def __init__(self, filter_fn):
        self.filter_fn = filter_fn
        self.verbose = True

    def __call__(self, msg: Message):

        data_record = msg["content"]

        answer = {}
        try:
            # execute the UDF filter
            passed_operator = self.filter_fn(data_record.as_dict())
            answer = {"passed_operator": passed_operator}

            if self.verbose:
                print(f"FunctionFilterHAL: {self.filter_fn}:\n{passed_operator}")

        except Exception as e:
            print(f"Error invoking user-defined function for filter: {e}")
            raise e

        return answer, None


class LLMConvertHAL:
    def __init__(self, cv_input_fields, cv_output_fields, cv_ipt_schema, cv_opt_schema, generator):
        self.cv_input_fields = cv_input_fields
        self.cv_output_fields = cv_output_fields
        self.cv_ipt_schema = cv_ipt_schema
        self.cv_opt_schema = cv_opt_schema
        self.convert_generator = generator


    def __call__(self, msg: Message):
        print("LLMConvertHAL.__call__")
        data_record = msg["content"]

        candidate = data_record
        # construct kwargs for generation
        gen_kwargs = {"project_cols": self.cv_input_fields, "output_schema": self.cv_opt_schema}
        # generate outputs for all fields in a single query
        field_answers, _, generation_stats = self.convert_generator(candidate, self.cv_output_fields, **gen_kwargs) # TODO: guarantee negative output from generator is None

        print(f"len(field_answers): {len(field_answers)}")
        # if there was an error for any field, execute a conventional query on that field
        for field, answers in field_answers.items():
            print(f"{ field } => { answers }")
            if answers is None:
                single_field_answers, _, single_field_stats = self.convert_generator(candidate, [field], **gen_kwargs)
                field_answers.update(single_field_answers)
                generation_stats += single_field_stats

        drs, ok = create_data_records_from_field_answers(field_answers, candidate, self.cv_ipt_schema, self.cv_opt_schema)
        assert len(drs) == 1 and ok, "only checked for #result=1"
        print('----')
        print(drs[0])
        nmsg = msg.spawn()
        nmsg['content'] = drs[0]
        return nmsg
