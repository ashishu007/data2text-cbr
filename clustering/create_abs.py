import sys
import os
import re
import json
from pathlib import Path
from nltk import Tree, sent_tokenize
import spacy, neuralcoref
from spacy.matcher import Matcher
from spacy.tokens import Token
from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA

import numpy as np
import pandas as pd

from itertools import groupby
import random

from known_ents import named_entities as ne

import numpy as np 
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import pprint
pp = pprint.PrettyPrinter(indent=4)

"""
Credits to [Craig Thomson](https://github.com/nlgcat) for this piece of code.
"""

class OtherTasks:
    def __init__(self):
        # sentence scoring
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')

    # ------------------------------ Sentence Scoring -------------------------------------
    def score_sent(self, tokens_tensor):
        loss = self.model(tokens_tensor, labels=tokens_tensor)[0]
        score = np.exp(loss.cpu().detach().numpy())
        return score
    # ------------------------------ Sentence Scoring -------------------------------------

class BaseMerger():
  def __init__(self, nlp, ent_type, pos='X'):
    self.matcher = Matcher(nlp.vocab)
    self.ent_type = ent_type
    self.pos = pos

  def __call__(self, doc):
    matches = self.matcher(doc)
    spans = []
    for match_id, start, end in matches:
      spans.append(doc[start:end])
    with doc.retokenize() as retokenizer:
      for span in spans:
        span[0].ent_type_ = self.ent_type
        span[0].pos_      = self.pos
        try:
          retokenizer.merge(span)
        except:
          print('--not mergining token')
    return doc

class NumberComparisonMerger(BaseMerger):
  def __init__(self, nlp):
    super().__init__(nlp, ent_type='CARDINAL_COMPARISON')
    self.matcher.add('XY', None, [{'IS_DIGIT': True},{'LOWER': '-'},{'IS_DIGIT': True}])

    for preposition in ['for', 'to', 'by', 'in']:
      pattern = [{'IS_DIGIT': True},{'LOWER': '-'},{'LOWER': preposition},{'LOWER': '-'},{'IS_DIGIT': True}]
      self.matcher.add(f'X{preposition}Y', None, pattern)

class ShotBreakdownMerger(BaseMerger):
  def __init__(self, nlp):
    super().__init__(nlp, ent_type='SHOT_BREAKDOWN')
    self.matcher.add('SHOT-BREAKDOWN', None, [{'ORTH': 'ShotBreakdown'}])


class TreeExtracter:

  # Nodes matching this NER/PoS will be 'abstracted' (replace with tag-label)
  special_ents = {
    'DATE':         set(['SYM', 'NUM', 'PROPN']),
    'TIME':         set(['SYM', 'NUM', 'PROPN']),
    'PERCENT':      set(['SYM', 'NUM', 'PROPN']),
    'MONEY':        set(['SYM', 'NUM', 'PROPN']),
    'ORDINAL':      set(['SYM', 'NUM', 'PROPN']),
    'CARDINAL':     set(['SYM', 'NUM', 'PROPN']),
    'PERSON':       set(['PROPN']),
    'NORP':         set(['PROPN']),
    'FAC':          set(['PROPN']),
    'ORG':          set(['PROPN']),
    'GPE':          set(['PROPN']),
    'LOC':          set(['PROPN']),
    'PRODUCT':      set(['PROPN']),
    'EVENT':        set(['PROPN']),
    'WORK_OF_ART':  set(['PROPN']),
    'LAW':          set(['PROPN']),
    'LANGUAGE':     set(['PROPN']),
  }

  def __init__(self):
    # Files to be operated on
    self.root_file_path   = os.path.join(os.getcwd(), 'files')
    self.json_file  = os.path.join(self.root_file_path, 'all_data_text.json')
    self.json_file_out  = os.path.join(self.root_file_path, 'all_data_text_abstract.json')

    # Delimiter used within node labels
    self.delimiter = '||'
    self.inner_delimiter = '|'

    self.abstract_to_id = {}

    # Spacy with neural coreference resolution
    self.nlp = spacy.load('en_core_web_lg', disable=[])
    neuralcoref.add_to_pipe(self.nlp)
    self.nlp.add_pipe(NumberComparisonMerger(self.nlp))
    self.nlp.add_pipe(ShotBreakdownMerger(self.nlp), last=True)

    self.named_entities = ne
    
  def spacy_parses_to_abstract_text(self, sents):
    tokens = []
    for sent in sents:
      for token in sent:
        if token.ent_type_:
          tokens.append(f'{token.pos_}-{token.ent_type_}')
        else:
          ## Need the whole token, instead of just lemma for template purposes
          # not now, because using original text for template extraction
          # tokens.append(f'{token.pos_}-{token.lemma_}')
          tokens.append(f'{token.pos_}-{token}')
    
    # Remove consecutive duplicates.  PROPN-GPE PROPN-ORG PROPN-ORG => PROPN-GPE PROPN-ORG (e.g. Portland Trail Blazers)
    return ' '.join([x[0] for x in groupby(tokens)])

  def raw_sentence(self, sents):
    return ' '.join([sent.text for sent in sents])

  # Update the JSON dict with the abstract text, also track embeddings and trees
  def process_one_line(self, line):
    if line != '':
      line = re.sub('\(\d+-\d+ \w{0,3}, \d+-\d+ \w{0,3}, \d+-\d+ \w{0,3}\)', '(ShotBreakdown)', line)
      line = re.sub('\s+', ' ', line)

      orig_doc = self.nlp(line)
      doc = self.nlp(orig_doc._.coref_resolved)

      trees         = []
      sents         = []
      sent_num      = 0
      valid_stop    = False
      abs_sents     = []
      
      for sent in doc.sents:
      # for sent in orig_doc.sents:
        if valid_stop:
          raw_sentence      = self.raw_sentence(sents)
          abstract = self.spacy_parses_to_abstract_text(sents)

          abs_sents.append({
            "raw_sent": raw_sentence,
            "abs_sent": abstract
          })

          trees = []
          sents = []
          sent_num += 1

        tree = self.to_nltk_tree(sent.root)
        trees.append(self.tree2dict(tree))
        sents.append(sent)

        entities      = []
        for ent in sent.ents:
          ent_text_arr = []
          for token in ent:
            if token.pos_ in ['NUM', 'NOUN', 'PROPN']:
              ent_text_arr.append(token.text)
          entity = {'text': ' '.join(ent_text_arr), 'ent_type': ent.root.ent_type_, 'tokens': []}
          for token in ent:
            entity['tokens'].append({'text': token.text, 'pos': token.pos_, 'tag': token.tag_})
          entities.append(entity)

        valid_stop = True if sent[-1].text in ['.', '?', '!'] else False

    return abs_sents

  # Convert spacy tokens to nltk trees
  def to_nltk_tree(self, node):
    if node.n_lefts + node.n_rights > 0:
      return Tree(self.format_token(node), [self.to_nltk_tree(child) for child in node.children])
    else:
      return self.format_token(node)

  # Convert an nltk tree to a dict
  def tree2dict(self, tree):
    if isinstance(tree, Tree):
      return {tree.label(): [self.tree2dict(subtree) for subtree in tree]}
    else:
      return str(tree)

  def apply_known_entity(self, token):
    for k, arr in self.named_entities.items():
      if token.orth_ in arr:
        # print(k, token.orth_)
        pos_ner = k.split('-')
        token.pos_ = pos_ner[0]
        token.ent_type_ = pos_ner[1]

  # Format the token, either abstracting it or lemmatising it
  def format_token(self, token):
    self.apply_known_entity(token)
    text = token.lemma_
    if token.ent_type_ in TreeExtracter.special_ents and token.pos_ in TreeExtracter.special_ents[token.ent_type_]:
      text = token.ent_type_
    h = {
      'pos':      token.pos_,
      'tag':      token.tag_,
      'lemma':    token.lemma_,
      'ent_type': token.ent_type_ if token.ent_type_ else 'NULL',
      'ent_id':   token.ent_id_,
      'orth':     token.orth_,
    }
    return self.delimiter.join([k + self.inner_delimiter + v for k,v in h.items()])

print('constructing')
extractor = TreeExtracter()
ot = OtherTasks()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

abs_sents = {
  'game_idx': [],
  'coref_sent': [],
  'abs': [],
  'season': []
}

"""
num_sample_id_1 ==> seasons = [2014]                (num_train_data = ~1200)
num_sample_id_2 ==> seasons = [2014, 2015]          (num_train_data = ~2400)
num_sample_id_3 ==> seasons = [2015, 2016]          (num_train_data = ~3600)
num_sample_id_4 ==> seasons = [2014, 2015, 2016]    (num_train_data = ~4800)
sample_size = {2014: 1226, 2015: 1211, 2016: 2308, 2017: 1228, 2018: 1229}
"""

sample_id = sys.argv[1]
# print(sample_id, type(sample_id))

if sample_id == 'num_sample_id_1':
  seasons = [2014]
elif sample_id == 'num_sample_id_2':
  seasons = [2014, 2015]
elif sample_id == 'num_sample_id_3':
  seasons = [2015, 2016]
elif sample_id == 'num_sample_id_4':
  seasons = [2014, 2015, 2016]

print(seasons)

sent_scores = {}

# seasons = [2014]
for season in seasons:
  print(season)
  data = json.load(open(f'./data/jsons/{season}_w_opp.json', 'r'))
  summ = [' '.join(i['summary']) for i in data]

  for idx, i in enumerate(summ):
    if idx % 100 == 0:
      print(idx)

    sents = sent_tokenize(i)

    out = extractor.process_one_line(i)
    tmp = [idx for item in range(len(out))]
    tmp1 = [season for item in range(len(out))]
    abs_sents['game_idx'].extend(tmp)
    abs_sents['season'].extend(tmp1)

    origs = [j['raw_sent'] for j in out]
    abss = [j['abs_sent'] for j in out]
    abs_sents['coref_sent'].extend(origs)
    abs_sents['abs'].extend(abss)

    # abss1 = []
    # for abs_s in abss:
    #   tmp = []
    #   print(abs_s)
    #   for tok in abs_s.split(' '):
    #     print(tok)
    #     try:
    #       tmp.append(tok.split('-')[1])
    #     except:
    #       pass
    #   print(tmp)
    #   print(' '.join(tmp))
    #   abss1.append(' '.join(tmp))
    #   # abss1.extend([tok.split('-')[1] for tok in abs_s.split(' ')])
    # # abss1 = [tok.split('-')[1] for abs_s in abss for tok in abs_s.split(' ')]
    # print(len(sents))
    # for s in sents:
    #   print(s)
    # print()
    # print(len(origs))
    # for s in origs:
    #   print(s)
    # # print(len(abss1), abss1)

    # for abs_sent in abss:
    #   # text = [tok.split('-')[1] for tok in abs_sent.split(" ")]
    #   text = ' '.join(text).replace("_", " ")
    #   tokens_tensor = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    #   sent_scores[abs_sent] = ot.score_sent(tokens_tensor)

df = pd.DataFrame(abs_sents)
df.to_csv('./clustering/data/abstract_sentences.csv', index=0)