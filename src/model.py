    import pandas as pd
    import re
    import spacy
    import neuralcoref
    import networkx as nx
    import matplotlib.pyplot as plt


    nlp = spacy.load('en_core_web_lg')
    neuralcoref.add_to_pipe(nlp)


class Model(object):

    def __init__(self, coreference=False):
        self.coreference = coreference 
        pass

    def __extract_entity(self, input_doc):
        pass

    def __preprocess_doc(self,input_doc):
        input_doc = re.sub(r'\n+', '.', input_doc)  # replace multiple newlines with period
        input_doc = re.sub(r'\[\d+\]', ' ', input_doc)  # remove reference numbers
        input_doc = nlp(input_doc)
        if self.coreference:
            input_doc = nlp(input_doc._.coref_resolved) 
        return input_doc     

    def fit(self,input_doc):
        processed_doc = self.__preprocess_doc(input_doc)
        # split text into sentences
        sentences = [sent.string.strip() for sent in processed_doc.sents]
        pass

    def visulize(self):
        pass
