from __future__ import annotations

from typing import Any, Callable

from llm_planner.compatible.palimpzest.core.elements.records import DataRecord
from llm_planner.compatible.palimpzest.core.lib.schemas import Schema

FieldName=str

# function used after convert_op
def create_data_records_from_field_answers(
    field_answers: dict[FieldName, list[Any]],
    candidate: DataRecord,
    input_schema: Schema, 
    output_schema: Schema
) -> list[DataRecord]:
    """
    Given a mapping from each field to its (list of) generated value(s), we construct the corresponding
    list of output DataRecords.
    """
    # get the number of records generated; for some convert operations it is possible for fields to
    # have different lengths of generated values, so we take the maximum length of any field's values
    # to be the number of records generated
    n_records = max([len(lst) for lst in field_answers.values()])
    successful_convert = n_records > 0

    drs = []
    for idx in range(max(n_records, 1)):
        # initialize record with the correct output schema, parent record, and cardinality idx
        dr = DataRecord.from_parent(output_schema, parent_record=candidate, cardinality_idx=idx)

        # copy all fields from the input record
        # NOTE: this means that records processed by PZ converts will inherit all pre-computed fields
        #       in an incremental fashion; this is a design choice which may be revisited in the future
        for field in candidate.get_field_names():
            setattr(dr, field, getattr(candidate, field))

        # get input field names and output field names
        input_fields = input_schema.field_names()
        output_fields = output_schema.field_names()

        # parse newly generated fields from the field_answers dictionary for this field; if the list
        # of generated values is shorter than the number of records, we fill in with None
        for field in output_fields:
            if field not in input_fields:
                value = field_answers[field][idx] if idx < len(field_answers[field]) else None
                setattr(dr, field, value)
        
        # append data record to list of output data records
        drs.append(dr)

    return drs, successful_convert
