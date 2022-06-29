import json
import numpy as np
from typing import Optional, List
from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from seq2seq.utils.dataset import DataTrainingArguments, normalize, serialize_schema
from seq2seq.utils.trainer import Seq2SeqTrainer, EvalPrediction


# def cosql_response_get_input(
#     query: str,
#     history: str,
#     serialized_schema: str,
#     prefix: str,
#     normalize_query: bool,
# ) -> str:
#     # "[prefix] [query] [value] || [serialized schema]"
#     _normalize = normalize if normalize_query else (lambda x: x)
#     return prefix + history + " | " + _normalize(query).strip() + " || " + serialized_schema.strip() 
def cosql_response_get_input(
    queries: list,
    questions: list,
    results: list,
    serialized_schema: str,
    prefix: str,
    normalize_query: bool,
    use_question: str,
    use_question_result: bool,
    use_gold_query: bool,
) -> str:
    _normalize = normalize if normalize_query else (lambda x: x)

    if len(questions) < 1:
        final_context = _normalize(queries[0]).strip()
    else:
        query, question, result = queries[-1], questions[-1], results[-1]
        question_blocks = []
        
        if use_question == "None":
            context = ""
            if use_question_result:
                context = _normalize(query).strip() + " | " + result 
            else:
                context = _normalize(query).strip()
            question_blocks.append(context)
        else:   
            if use_question == "current":
                total_number = 1
            elif use_question == "last":
                total_number = 2
            elif use_question == "all":
                total_number = len(queries)
            for i, (qes, qry, res) in enumerate(zip(questions[:total_number], queries[:total_number], results[:total_number])):
                context = qes
                if i==total_number-1:
                    context = context+ " | " + _normalize(query).strip()
                else:
                    if use_gold_query:
                        context = context+ " | " + _normalize(query).strip()
                if use_question_result:
                    context = context+ " | " + res
                question_blocks.append(context)
                
        final_context = " || ".join(question_blocks)
    
    return prefix + final_context + " || " + serialized_schema.strip() 


def cosql_response_get_target(
    utterances: str,
    db_id: str,
    target_with_db_id: bool,
) -> str:
    
    return f"{db_id} | {utterances}" if target_with_db_id else utterances


def cosql_response_add_serialized_schema(
    ex: dict,
    data_training_args: DataTrainingArguments,
) -> dict:
    serialized_schema = serialize_schema(
        question=ex["utterances"],
        db_path=ex["db_path"],
        db_id=ex["db_id"],
        db_column_names=ex["db_column_names"],
        db_table_names=ex["db_table_names"],
        schema_serialization_type=data_training_args.schema_serialization_type,
        schema_serialization_randomized=data_training_args.schema_serialization_randomized,
        schema_serialization_with_db_id=data_training_args.schema_serialization_with_db_id,
        schema_serialization_with_db_content=data_training_args.schema_serialization_with_db_content,
        normalize_query=data_training_args.normalize_query,
    )
    return {"serialized_schema": serialized_schema}


def cosql_response_pre_process_function(
    batch: dict,
    max_source_length: Optional[int],
    max_target_length: Optional[int],
    data_training_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    prefix = data_training_args.source_prefix if data_training_args.source_prefix is not None else ""

    inputs = [
        cosql_response_get_input(
            queries=queries,
            questions=questions,
            results=results,
            serialized_schema = serialized_schema,
            prefix=prefix,
            normalize_query=data_training_args.normalize_query,
            use_question=data_training_args.use_question,
            use_question_result=data_training_args.use_question_result,
            use_gold_query=data_training_args.use_gold_query,
        ) 
        for queries, serialized_schema, questions, results in zip(batch["queries"], batch["serialized_schema"], batch["questions"], batch["results"])
    ]

    model_inputs: dict = tokenizer(
        inputs,
        max_length=max_source_length,
        padding=False,
        truncation=False,
        return_overflowing_tokens=False,
    )

    targets = [
        cosql_response_get_target(
            utterances=utterances,
            db_id=db_id,
            target_with_db_id=data_training_args.target_with_db_id,
        )
        for db_id, utterances in zip(batch["db_id"], batch["utterances"])
    ]

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding=False,
            truncation=True,
            return_overflowing_tokens=False,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


class CoSQLResponseTrainer(Seq2SeqTrainer):
    def _post_process_function(
        self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        inputs = self.tokenizer.batch_decode([f["input_ids"] for f in features], skip_special_tokens=True)
        label_ids = [f["labels"] for f in features]
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            _label_ids = np.where(label_ids != -100, label_ids, self.tokenizer.pad_token_id)
        decoded_label_ids = self.tokenizer.batch_decode(_label_ids, skip_special_tokens=True)
        metas = [
            {
                "query": x["query"],
                "utterances": x["utterances"],
                "questions": x["questions"],
                "queries": x["queries"],
                "results": x["results"],
                "context": context,
                "label": label,
                "db_id": x["db_id"],
                "db_path": x["db_path"],
                "db_table_names": x["db_table_names"],
                "db_column_names": x["db_column_names"],
                "db_foreign_keys": x["db_foreign_keys"],
            }
            for x, context, label in zip(examples, inputs, decoded_label_ids)
        ]
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        assert len(metas) == len(predictions)
        with open(f"{self.args.output_dir}/predictions_{stage}.json", "w") as f:
            json.dump(
                [dict(**{"prediction": prediction}, **meta) for prediction, meta in zip(predictions, metas)],
                f,
                indent=4,
            )
        with open(f"{self.args.output_dir}/predicted_response.txt", "w") as f:
            for p in predictions:
                p = p.split("|", 1)[-1].strip()
                f.write(p)
                f.write("\n")
        return EvalPrediction(predictions=predictions, label_ids=label_ids, metas=metas)

    def _compute_metrics(self, eval_prediction: EvalPrediction) -> dict:
        predictions, label_ids, metas = eval_prediction
        if self.target_with_db_id:
            # Remove database id from all predictions
            predictions = [pred.split("|", 1)[-1].strip() for pred in predictions]
        # TODO: using the decoded reference labels causes a crash in the spider evaluator
        # if self.ignore_pad_token_for_loss:
        #     # Replace -100 in the labels as we can't decode them.
        #     label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        # decoded_references = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        # references = [{**{"query": r}, **m} for r, m in zip(decoded_references, metas)]
        references = metas
        return self.metric.compute(predictions=predictions, references=references)
