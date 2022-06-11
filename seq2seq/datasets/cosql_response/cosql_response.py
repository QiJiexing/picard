# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CoSQL: A Conversational Text-to-SQL Challenge Towards Cross-Domain Natural Language Interfaces to Databases"""


import json
from third_party.spider.preprocess.get_tables import dump_db_json_schema

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{yu-etal-2019-cosql,
    title = "{C}o{SQL}: A Conversational Text-to-{SQL} Challenge Towards Cross-Domain Natural Language Interfaces to Databases",
    author = "Yu, Tao  and
      Zhang, Rui  and
      Er, Heyang  and
      Li, Suyi  and
      Xue, Eric  and
      Pang, Bo  and
      Lin, Xi Victoria  and
      Tan, Yi Chern  and
      Shi, Tianze  and
      Li, Zihan  and
      Jiang, Youxuan  and
      Yasunaga, Michihiro  and
      Shim, Sungrok  and
      Chen, Tao  and
      Fabbri, Alexander  and
      Li, Zifan  and
      Chen, Luyao  and
      Zhang, Yuwen  and
      Dixit, Shreya  and
      Zhang, Vincent  and
      Xiong, Caiming  and
      Socher, Richard  and
      Lasecki, Walter  and
      Radev, Dragomir",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1204",
    doi = "10.18653/v1/D19-1204",
    pages = "1962--1979",
    abstract = "We present CoSQL, a corpus for building cross-domain, general-purpose database (DB) querying dialogue systems. It consists of 30k+ turns plus 10k+ annotated SQL queries, obtained from a Wizard-of-Oz (WOZ) collection of 3k dialogues querying 200 complex DBs spanning 138 domains. Each dialogue simulates a real-world DB query scenario with a crowd worker as a user exploring the DB and a SQL expert retrieving answers with SQL, clarifying ambiguous questions, or otherwise informing of unanswerable questions. When user questions are answerable by SQL, the expert describes the SQL and execution results to the user, hence maintaining a natural interaction flow. CoSQL introduces new challenges compared to existing task-oriented dialogue datasets: (1) the dialogue states are grounded in SQL, a domain-independent executable representation, instead of domain-specific slot value pairs, and (2) because testing is done on unseen databases, success requires generalizing to new domains. CoSQL includes three tasks: SQL-grounded dialogue state tracking, response generation from query results, and user dialogue act prediction. We evaluate a set of strong baselines for each task and show that CoSQL presents significant challenges for future research. The dataset, baselines, and leaderboard will be released at https://yale-lily.github.io/cosql.",
}
"""

_DESCRIPTION = """\
CoSQL is a large-scale dataset for training and testing task oriented dialog agents with SQL
"""

_HOMEPAGE = "https://yale-lily.github.io/cosql"

_LICENSE = "CC BY-SA 4.0"

# _URL = "https://drive.google.com/uc?export=download&id=14x6lsWqlu6gR-aYxa6cemslDN3qT3zxP"
_URL = "../../../dataset_files/cosql_dataset.zip"


class CoSQL(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="cosql",
            version=VERSION,
            description="A Conversational Text-to-SQL Challenge Towards Cross-Domain Natural Language Interfaces to Databases",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()

    def _info(self):
        features = datasets.Features(
            {
                "query": datasets.Value("string"),
                "history": datasets.Value("string"),
                "utterances": datasets.Value("string"),
                "db_id": datasets.Value("string"),
                "db_path": datasets.Value("string"),
                "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                "db_column_names": datasets.features.Sequence(
                    {
                        "table_id": datasets.Value("int32"),
                        "column_name": datasets.Value("string"),
                    }
                ),
                "db_column_types": datasets.features.Sequence(datasets.Value("string")),
                "db_primary_keys": datasets.features.Sequence({"column_id": datasets.Value("int32")}),
                "db_foreign_keys": datasets.features.Sequence(
                    {
                        "column_id": datasets.Value("int32"),
                        "other_column_id": datasets.Value("int32"),
                    }
                ),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_filepath = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "/cosql_dataset/system_response_generation/cosql_train.json",
                    "db_path": downloaded_filepath + "/cosql_dataset/database",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "/cosql_dataset/system_response_generation/cosql_dev.json",
                    "db_path": downloaded_filepath + "/cosql_dataset/database",
                },
            ),
        ]

    # def get_history(self, db_id, query, dataset):
    #     # with open(dataset_path, "r") as f:
    #     #     dataset = json.load(f)
    #         # add last question
    #     def search_on_dialog(query, dialog_id, dataset):
    #         import re
    #         info = dataset[dialog_id]
    #         turns = info['turns']
    #         find_idx = -1
    #         for turn in turns:
    #             isSQL = turn.get("isSql", False)
    #             if isSQL:
    #                 q = query.lower()
    #                 s = turn['rawSql'].lower()
    #                 consice_q = re.sub('[\W_]+', '', q)
    #                 consice_s = re.sub('[\W_]+', '', s)
    #                 if consice_q == consice_s:
    #                     find_idx = turn['turn_index']
                        
    #         if find_idx > 0:
    #             # print("Find ")
    #             return find_idx, turns[:find_idx]
    #         else:
    #             return None
            
    #     def search_on_database(query, db_id, db_id2dialog_id, dataset):
    #         '''
    #         Args:
    #             query: the SQL query for response generation task
    #             db_id: the database id
    #             db_id2dialog_id: the mapping dict for database id to all dialog ids that belong to this db
    #             dataset: all info json
                
    #         Return:
    #             context: the dialog history context for the SQL
                
    #         '''
    #         dialog_ids = db_id2dialog_id.get(db_id, None)
    #         assert dialog_ids is not None
    #         for dialog_id in dialog_ids:
    #             result = search_on_dialog(query, dialog_id, dataset)
    #             if result is None:
    #                 context = None
    #                 # print("Not found")
    #             else:
    #                 find_idx, history_info = result
    #                 context = history_info[-1]["text"] 
    #                 break
    #         return context

    # # test search_on_dialog func
    #     from collections import defaultdict
    #     db_id2dialog_id = defaultdict(list)
    #     for dialog_id, info in dataset.items():
    #         db_id = info['db_id']
    #         # pprint(info)
    #         db_id2dialog_id[db_id].append(dialog_id)
    # # result = search_on_database(query, db_id, db_id2dialog_id, dataset)
    #     history = None
    #     for dialog_id in dataset.keys():
    #         # print("query: ", query)
    #         result = search_on_database(query, db_id, db_id2dialog_id, dataset)

    #         if result is not None:
    #             history = result
    #             # print("history: ", history)
    #             break
    #     if history is None:
    #         history = ""

    #     print("history: ", history)
    #     return history

    def _generate_examples(self, data_filepath, db_path):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", data_filepath)
        idx = 0 # indexing each training instance
        all_info_filepath = data_filepath.replace("system_response_generation/cosql_dev.json", "cosql_all_info_dialogs.json").replace("system_response_generation/cosql_train.json", "cosql_all_info_dialogs.json")
        with open(data_filepath, encoding="utf-8") as f:
            cosql = json.load(f)
            if "train" in data_filepath:
                # v1 means only last question will be added as history
                # pkl_filepath = "/home/jxqi/unified_cosql/picard/dataset_files/train_context_v1.pkl"
                # v2 means all previous questions will be added as history
                pkl_filepath = "/home/jxqi/unified_cosql/picard/dataset_files/train_context_v2.pkl"
            else:
                # v1 means only last question will be added as history
                # pkl_filepath = "/home/jxqi/unified_cosql/picard/dataset_files/dev_context_v1.pkl"
                # v2 means all previous questions will be added as history
                pkl_filepath = "/home/jxqi/unified_cosql/picard/dataset_files/dev_context_v2.pkl"
            import pickle
            history_l = pickle.load(open(pkl_filepath, "rb"))
            for sample in cosql:
                db_id = sample["database_id"]
                if db_id not in self.schema_cache:
                    self.schema_cache[db_id] = dump_db_json_schema(
                        db_path + "/" + db_id + "/" + db_id + ".sqlite", db_id
                    )
                schema = self.schema_cache[db_id]

                db_info = {
                    "db_id": db_id,
                    "db_path": db_path,
                    "db_table_names": schema["table_names_original"],
                    "db_column_names": [
                        {"table_id": table_id, "column_name": column_name}
                        for table_id, column_name in schema["column_names_original"]
                    ],
                    "db_column_types": schema["column_types"],
                    "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
                    "db_foreign_keys": [
                        {"column_id": column_id, "other_column_id": other_column_id}
                        for column_id, other_column_id in schema["foreign_keys"]
                    ],
                }

                # query = sample["query"]
                # history = self.get_history(query, db_id, dataset)

                yield idx, {
                    "utterances": sample["response"],
                    "query": sample["query"],
                    "history": history_l[idx],
                    **db_info,
                }
                idx += 1


