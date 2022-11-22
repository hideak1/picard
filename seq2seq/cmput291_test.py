# Set up logging
import sys
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)

from typing import Optional, Dict
from dataclasses import dataclass, field
from pydantic import BaseModel
import os
from contextlib import nullcontext
from transformers.hf_argparser import HfArgumentParser
from transformers.models.auto import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from sqlite3 import Connection, connect, OperationalError
from seq2seq.utils.pipeline import Text2SQLGenerationPipeline, Text2SQLInput, get_schema
from seq2seq.utils.picard_model_wrapper import PicardArguments, PicardLauncher, with_picard
from seq2seq.utils.dataset import DataTrainingArguments
import json
import numpy as np
import sqlite3
import sqlparse
from copy import deepcopy
import torch
@dataclass
class BackendArguments:
    """
    Arguments pertaining to model serving.
    """

    model_path: str = field(
        default="tscholak/cxmefzzi",
        metadata={"help": "Path to pretrained model"},
    )
    cache_dir: Optional[str] = field(
        default="/tmp",
        metadata={"help": "Where to cache pretrained models and data"},
    )
    db_path: str = field(
        default="database",
        metadata={"help": "Where to to find the sqlite files"},
    )
    host: str = field(default="0.0.0.0", metadata={"help": "Bind socket to this host"})
    port: int = field(default=8000, metadata={"help": "Bind socket to this port"})
    device: int = field(
        default=0,
        metadata={
            "help": "Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU. A non-negative value will run the model on the corresponding CUDA device id."
        },
    )

def load_cmput291(path):
    train_questions = []
    train_answers = []
    test_questions = []
    test_answers = []
    with open(path + "/cmput291_database/cmput291.json", "r") as f:
        jd = json.load(f)
        for myjson in jd:
            sentences = myjson['sentences']
            variables = myjson['variables']

            sqls = myjson['sql']
            for sentence in sentences:
                for sql in sqls:
                    question = sentence['text']
                    ans = sql
                    for variable in variables:
                        question = question.replace(variable['name'], sentence['variables'][variable['name']])
                        ans = ans.replace(variable['name'], sentence['variables'][variable['name']])
                    if sentence['question-split'] == 'test':
                        test_questions.append(question)
                        test_answers.append(ans)
                    elif sentence['question-split'] == 'train':
                        train_questions.append(question)
                        train_answers.append(ans)

    return train_questions, train_answers, test_questions, test_answers

def em_accuracy_helper(prediction, label):
    correctness_list = []
    for pred, l in zip(prediction, label):
        a = sqlparse.format(pred, reindent=False, keyword_case='upper')
        b = sqlparse.format(l, reindent=False, keyword_case='upper')
        # pred = pred.split('\n')[0]
        if a == b:
            correctness_list.append(1)
        else:
            correctness_list.append(0)
    return np.mean(correctness_list)

def execute_func(sql):
    def get_connection(db_file: str):
        conn = sqlite3.connect(db_file)
        return conn
    connection = get_connection('database/cmput291_database/cmput291_database.sqlite')
    cursor = connection.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    print("Result: ", results)
    connection.close()
    return results

def execution_accuracy_helper(prediction, label):
    correctness_list = []
    va_list = []
    for pred, l in zip(prediction, label):
        try:
            print(f'pred: {pred} gt: {l}')
            p_result = execute_func(pred)
            gt_result = execute_func(l)
            va_list.append(1)
            if p_result == gt_result:
                correctness_list.append(1)
            else:
                correctness_list.append(0)
        except Exception as e:
            va_list.append(0)
            correctness_list.append(0)
    return np.mean(correctness_list), np.mean(va_list)

def random_sampling(sentences, labels, num):
    """randomly sample subset of the training pairs"""
    assert len(sentences) == len(labels)
    if num > len(labels):
        assert False, f"you tried to randomly sample {num}, which is more than the total size of the pool {len(labels)}"
    idxs = np.random.choice(len(labels), size=num, replace=False)
    selected_sentences = [sentences[i] for i in idxs]
    selected_labels = [labels[i] for i in idxs]
    return deepcopy(selected_sentences), deepcopy(selected_labels)

def prompt_func(train_sentences, train_labels, test_sentence, test_label_option=None):
    table_info = ""
    # with open("data/cmput291/cmput291-fields.txt", "r") as f:
    #     data = f.readlines()
    #     table_info = ''.join(data)
    q_prefix = "QUESTION: "
    a_prefix = "ANSWER: "

    prompt = ""
    # prompt += table_info
    for x, y in zip(train_sentences, train_labels):
        prompt += f"{q_prefix}{x}\n{a_prefix}{y}"
        prompt += "\n\n"

    if test_label_option is None:
        prompt += f"{q_prefix}{test_sentence}\n{a_prefix}"
    else:
        prompt += f"{q_prefix}{test_sentence}\n{a_prefix}" + test_label_option
    return prompt

def main():
    # See all possible arguments by passing the --help flag to this program.
    parser = HfArgumentParser((PicardArguments, BackendArguments, DataTrainingArguments))
    picard_args: PicardArguments
    backend_args: BackendArguments
    data_training_args: DataTrainingArguments
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        picard_args, backend_args, data_training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        picard_args, backend_args, data_training_args = parser.parse_args_into_dataclasses()

    # Initialize config
    config = AutoConfig.from_pretrained(
        backend_args.model_path,
        cache_dir=backend_args.cache_dir,
        max_length=data_training_args.max_target_length,
        num_beams=data_training_args.num_beams,
        num_beam_groups=data_training_args.num_beam_groups,
        diversity_penalty=data_training_args.diversity_penalty,
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        backend_args.model_path,
        cache_dir=backend_args.cache_dir,
        use_fast=True,
    )


    # Initialize Picard if necessary
    with PicardLauncher() if picard_args.launch_picard else nullcontext(None):
        # Get Picard model class wrapper
        if picard_args.use_picard:
            model_cls_wrapper = lambda model_cls: with_picard(
                model_cls=model_cls, picard_args=picard_args, tokenizer=tokenizer
            )
        else:
            model_cls_wrapper = lambda model_cls: model_cls

        # Initialize model
        model = model_cls_wrapper(AutoModelForSeq2SeqLM).from_pretrained(
            backend_args.model_path,
            config=config,
            cache_dir=backend_args.cache_dir,
        )

        # Initalize generation pipeline
        pipe = Text2SQLGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            db_path=backend_args.db_path,
            prefix=data_training_args.source_prefix,
            normalize_query=data_training_args.normalize_query,
            schema_serialization_type=data_training_args.schema_serialization_type,
            schema_serialization_with_db_id=data_training_args.schema_serialization_with_db_id,
            schema_serialization_with_db_content=data_training_args.schema_serialization_with_db_content,
            device=backend_args.device,
        )

        train_questions, train_answers, test_questions, test_answers = load_cmput291(backend_args.db_path)

        ans = []

        for i in range(len(test_questions)):
        # for i in range(20):
            # prompt_questions, prompt_answers = random_sampling(test_questions, test_answers, 1)
            question = test_questions[i]
            # label = test_answers[i]
            # input = prompt_func(prompt_questions, prompt_answers, question)
            outputs = pipe(
                        inputs=Text2SQLInput(utterance=question, db_id="cmput291_database"),
                        num_return_sequences=data_training_args.num_return_sequences
                    )
            ans.append(outputs[0]["generated_text"])
        orig_accuracy = em_accuracy_helper(ans, test_answers)

        org_ex_accuracy, org_va_accuracy = execution_accuracy_helper(ans, test_answers)

        print(f"accuracies {orig_accuracy}")

        print(f"ex accuracies {org_ex_accuracy}")

        print(f"va accuracies {org_va_accuracy}")

        # Initialize REST API
        # app = FastAPI()

        # class AskResponse(BaseModel):
        #     query: str
        #     execution_results: list
        
        # def response(query: str, conn: Connection) -> AskResponse:
        #     try:
        #         return AskResponse(query=query, execution_results=conn.execute(query).fetchall())
        #     except OperationalError as e:
        #         raise HTTPException(
        #             status_code=500, detail=f'while executing "{query}", the following error occurred: {e.args[0]}'
        #         )

        # @app.get("/ask/{db_id}/{question}")
        # def ask(db_id: str, question: str):
        #     try:
        #         outputs = pipe(
        #             inputs=Text2SQLInput(utterance=question, db_id=db_id),
        #             num_return_sequences=data_training_args.num_return_sequences
        #         )
        #     except OperationalError as e:
        #         raise HTTPException(status_code=404, detail=e.args[0])
        #     try:
        #         conn = connect(backend_args.db_path + "/" + db_id + "/" + db_id + ".sqlite")
        #         return [response(query=output["generated_text"], conn=conn) for output in outputs]
        #     finally:
        #         conn.close()

        # # Run app
        # run(app=app, host=backend_args.host, port=backend_args.port)


if __name__ == "__main__":
    main()
