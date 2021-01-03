import torch
import networkx as nx
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel


class TextSummarizer:
    def __init__(self):
        self.threshold = 2
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.model.eval()

    def get_sentence_embedding(self, sentence_text):
        # Create tokens, token' indices, segment ids
        tokenized_text = self.tokenizer.tokenize(f"[CLS] {sentence_text} [SEP]")
        indexed_token = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segment_ids = [1]*len(tokenized_text)
        # Convert to tensors
        tokens_tensor = torch.tensor([indexed_token])
        segments_tensor = torch.tensor([segment_ids])
        # Evaluate, get hidden state layers
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensor)
            hidden_states = outputs[2]
        # Get second to last layer, use mean pooling of all word embeddings
        # Dimensions: [#layer, #batch, #token, #feature]
        token_vecs = hidden_states[-2][0]
        return torch.mean(token_vecs, dim=0)

    def text_ranking(self, sent_embeddings):
        sim_mat = np.zeros([len(sent_embeddings), len(sent_embeddings)])
        cos = torch.nn.CosineSimilarity(dim=0)
        for i in range(len(sent_embeddings)):
            for j in range(len(sent_embeddings)):
                if i != j:
                    sim = cos(sent_embeddings[i][1], sent_embeddings[j][1])
                    sim_mat[i][j] = sim.item()

        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        ranked_sentences = sorted(
            ((scores[i], i, s[0]) for i, s in enumerate(sent_embeddings)), reverse=True, key=lambda tup: tup[0])
        return ranked_sentences

    def summarize(self, text):
        if not text:
            return ''

        sentences = sent_tokenize(text)
        # If text too short, return the original text
        if len(sentences) < self.threshold:
            return text

        sent_embeddings = []
        stop_words = set(stopwords.words('english'))
        for sent in sentences:
            sent_tokens = word_tokenize(sent)
            new_sentence = ' '.join(word for word in sent_tokens if word.lower() not in stop_words)
            torch_embedding = self.get_sentence_embedding(new_sentence)
            sent_embeddings += [(sent, torch_embedding)]

        # Run TextRank
        sentences_ranking = self.text_ranking(sent_embeddings)
        # Get 50% of sentences, sort by sentence order
        # print(sentences_ranking[:len(sentences_ranking) // 2])
        select_sentences = sorted(sentences_ranking[:len(sentences_ranking) // 2], key=lambda tup: tup[1])
        # print(select_sentences)

        return ''.join([sent[2] for sent in select_sentences])
