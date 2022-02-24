import pickle
import re
from datetime import datetime
from functools import lru_cache

import neuralcoref
import pandas as pd
import spacy

# import networkx as nx
# import matplotlib.pyplot as plt
from tqdm import tqdm


class Model(object):
    __nlp = spacy.load("en_core_web_lg")
    neuralcoref.add_to_pipe(__nlp)

    def __init__(self, entity_pairs, coreference=False):
        self.__coreference = coreference
        self.__unwanted_tokens = (
            "PRON",  # pronouns
            "PART",  # particle
            "DET",  # determiner
            "SCONJ",  # subordinating conjunction
            "PUNCT",  # punctuation
            "SYM",  # symbol
            "X",  # other
        )
        self.__entity_pairs = entity_pairs if entity_pairs else []

    def __preprocess_doc(self, input_doc):
        # replace multiple newlines with period
        preprocessed_doc = re.sub(r"\n+", ".", input_doc)
        # remove reference numbers
        preprocessed_doc = re.sub(r"\[\d+\]", " ", preprocessed_doc)
        preprocessed_doc = Model.__nlp(preprocessed_doc)
        if self.__coreference:
            preprocessed_doc = Model.__nlp(preprocessed_doc._.coref_resolved)
        return preprocessed_doc

    @lru_cache(maxsize=25)
    def __refine_entity(self, entity, sentense):
        ent_type = entity.ent_type_
        if ent_type == "":
            ent_type = "NOUN_CHUNK"
            entity = " ".join(
                str(t.text)
                for t in Model.__nlp(str(entity))
                if t.pos_ not in self.__unwanted_tokens and t.is_stop is False
            )
            return entity, ent_type

        if ent_type in ("NOMINAL", "CARDINAL", "ORDINAL") & str(entity).find(" ") == -1:
            refined = ""
            for i in range(len(sentense) - entity.i):
                if entity.nbor(i).pos_ not in ("VERB", "PUNCT"):
                    refined += " " + str(entity.nbor(i))
                else:
                    entity = refined.strip()
                    break
            return entity, ent_type
        else:
            return entity, ent_type

    def fit(self, input_doc):
        processed_doc = self.__preprocess_doc(input_doc)
        # split text into sentences
        sentences = [sent.string.strip() for sent in processed_doc.sents]
        entity_pairs = []
        progress = tqdm(desc="Processing Sentences", unit="", total=len(sentences))
        for sentence in sentences:
            sentence = Model.__nlp(sentence)

            spans = list(sentence.ents) + list(sentence.noun_chunks)  # collect nodes
            spans = spacy.util.filter_spans(spans)
            with sentence.retokenize() as retokenizer:
                [
                    retokenizer.merge(
                        span, attrs={"tag": span.root.tag, "dep": span.root.dep}
                    )
                    for span in spans
                ]

            dependency = [token.dep_ for token in sentence]
            # limit our example to simple sentences with one subject and object
            if (dependency.count("obj") + dependency.count("dobj")) != 1:
                continue
            if (dependency.count("subj") + dependency.count("nsubj")) != 1:
                continue

            for token in sentence:

                if token.dep_ not in ("obj", "dobj"):
                    continue

                subject = [w for w in token.head.lefts if w.dep_ in ("subj", "nsubj")]

                if not subject:
                    continue

                subject = subject[0]

                relation = [w for w in token.ancestors if w.dep_ == "ROOT"]

                if relation:
                    relation = relation[0]
                    if relation.nbor(1).pos_ in ("ADP", "PART"):
                        relation = " ".join((str(relation), str(relation.nbor(1))))
                else:
                    relation = "unknown"

                subject, subject_type = self.__refine_entity(subject, sentence)
                token, object_type = self.__refine_entity(token, sentence)
                entity_pairs.append(
                    [
                        str(subject),
                        str(relation),
                        str(token),
                        str(subject_type),
                        str(object_type),
                    ]
                )
            progress.update(1)
        entity_pairs = list(
            filter(lambda x: not any(str(item) == "" for item in x), entity_pairs)
        )
        self.__entity_pairs.extend(entity_pairs)

    def knowlwdge_graph(self, out_format="list"):
        if len(self.__entity_pairs) == 0:
            raise Exception("Run the method model.fit() first")
        if out_format == "list":
            return self.__entity_pairs
        elif out_format == "DataFrame":
            return pd.DataFrame(
                self.__entity_pairs,
                columns=[
                    "subject",
                    "relation",
                    "object",
                    "subject_type",
                    "object_type",
                ],
            )
        else:
            raise ValueError(f"argument format ={format} is not valid")

    def save(self, file_name=""):
        if len(self.__entity_pairs) == 0:
            raise Exception("Run the method model.fit() first")
        if file_name == "":
            file_name = "Model_{}_{}.pkl".format(
                len(self.__entity_pairs), datetime.today().strftime("%d/%m/%Y_%H:%M:%S")
            )
        else:
            file_name = re.sub(r".", "_", file_name)
            file_name = "{}.pkl".format(file_name)
        with open(file_name, "wb") as handle:
            pickle.dump(self.__entity_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_name):
        entitypairs = []
        with open(file_name, "rb") as handle:
            entitypairs = pickle.load(handle)
        if len(entitypairs) == 0:
            raise Exception(f"file {file_name} contains No Data")
        return Model(coreference=False, entity_pairs=entitypairs)

    def visulize(self):
        pass
