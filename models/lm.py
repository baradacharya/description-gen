#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import collections
from math import log
import sys

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

class LangModel:
    def fit_corpus(self, corpus): pass

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        return pow(2.0, self.entropy(corpus))

    def entropy(self, corpus):
        # 1/num_words * sum_i_m log(P(s_i)) #m = total no of sentences
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            #num_words += len(s) + 1 #1 EOS
            num_words += len(s)  # 2 SOS, 1 EOS
            sum_logprob += self.logprob_sentence(s)
        return -(1.0/num_words)*(sum_logprob)

    def logprob_sentence(self, sentence): pass
    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence): pass
    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self): pass
    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous): pass
    # required, the list of words the language model suports (including EOS)
    def vocab(self): pass

class Unigram(LangModel):
    def __init__(self, backoff = 0.000001):
        self.model = dict()
        self.lbackoff = log(backoff, 2)

    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence(s)
        self.norm()

    def logprob_sentence(self, sentence):
        p = 0.0
        for i in xrange(len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i])
        p += self.cond_logprob('<EOS>', sentence)
        return p

    def inc_word(self, w):
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        #create a word dictionary with word and it's frequencies.
        for w in sentence:
            self.inc_word(w)
        self.inc_word('<EOS>')

    def norm(self):
        """Normalize and convert to log2-probs."""
        """
        log(#freq(W_i)/#total words) , self.model[word] = log(#freq(W_i) - log(#total words)
        """
        tot = 0.0 #total no of words
        for word in self.model:
            tot += self.model[word]
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) - ltot

    def cond_logprob(self, word, previous):
        #P(A|B) = #freq(W_i)/#total words, we have already normalized and working in log space
        if word in self.model:
            return self.model[word]
        else:
            return self.lbackoff #when the word not present log(0) = -inf, instead of that backoff

    def vocab(self):
        return self.model.keys()

class Bigram(LangModel):
    def __init__(self, backoff=0.000001):
        self.lbackoff = log(backoff, 2)
        self.bigram = dict()
        self.unigram = dict()

        # smoothing index
        self.l = 0  # laplace

    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence(s)
            # self.norm()

    def inc_uni(self, word):
        if word in self.unigram:
            self.unigram[word] += 1
        else:
            self.unigram[word] = 1

    def logprob_sentence(self, sentence):
        # P(s_i) = product(cond prob) #so adding in log space
        p = 0.0
        for i in xrange(1, len(sentence)):
            # log prob of sentences
            p += self.cond_logprob(sentence[i], sentence[:i])
        return p

    def fit_sentence(self, sentence):
        # add bigram as sos + word and trigram as sos + sos + word

        sentence.insert(0, '<SOS>')
        sentence.append("<EOS>")

        # for bigram
        for i in range(1, len(sentence)):

            bi = sentence[i - 1] + " " + sentence[i]
            bi = bi.lower()
            if bi in self.bigram:
                self.bigram[bi] += 1
            else:
                self.bigram[bi] = 1
        # for unigram
        # should add sos to unigram in this case to get p(x|sos) but remove sos while sampleing becoz we don't want p(sos|x)
        for i in range(len(sentence)):
            self.inc_uni(sentence[i].lower())

    def cond_logprob(self, word, previous):

        V = previous[-1]
        VW = previous[-1] + " " + word

        V = V.lower()
        VW = VW.lower()

        if VW in self.bigram:
            C_vw = self.bigram[VW]
        else:
            C_vw = 0

        if V in self.unigram:
            C_v = self.unigram[V]
        else:
            C_v = 0

        total_word = len(self.vocab())
        l_v = self.l * total_word

        if l_v == 0:
            if C_v == 0 and C_vw == 0:
                score = 0
            elif C_vw == 0:
                score = self.lbackoff - log((C_v + l_v), 2)
            elif C_v == 0:
                score = log((C_vw + self.l), 2) - self.lbackoff
            else:
                score = log(C_vw, 2) - log(C_v, 2)
        else:
            score = log((C_vw + self.l), 2) - log((C_v + l_v), 2)

        return score

    def vocab(self):
        return self.unigram.keys()

    def pre_processes(self, corpus):
        # add <sos>, <sos>, and <eos> to sentence
        for s in corpus:
            s.insert(0, '<SOS>')
            s.append("<EOS>")

class Trigram(LangModel):
    def __init__(self, backoff = 0.000001):
        self.lbackoff = log(backoff, 2)

        self.trigram = dict()
        self.bigram = dict()
        self.unigram = dict()

        #smoothing index
        self.l = 0 #laplace

        #linerar_interpolation
        self.l1 = 0.20
        self.l2 = 0.39
        self.l3 = 0.41

    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence(s)
        #self.norm()

    def inc_uni(self, word):
        if word in self.unigram:
            self.unigram[word] += 1
        else:
            self.unigram[word] = 1

    def logprob_sentence(self, sentence):
        #P(s_i) = product(cond prob) #so adding in log space
        p = 0.0
        for i in xrange(2,len(sentence)):
            #log prob of sentences
            #p += self.cond_logprob_linerar_interpolation(sentence[i], sentence[:i])
            p += self.cond_logprob_laplace(sentence[i], sentence[:i])
        return p

    def fit_sentence(self, sentence):
        #add bigram as sos + word and trigram as sos + sos + word

        sentence.insert(0, '<SOS>')
        sentence.insert(0, '<SOS>')
        sentence.append("<EOS>")

        # for trigram
        for i in range(2, len(sentence)):

            tri = sentence[i -2] + " " + sentence[i - 1] + " " + sentence[i]
            tri = tri.lower()
            if tri in self.trigram:
                self.trigram[tri] += 1
            else:
                self.trigram[tri] = 1

        # for bigram
        for i in range(1, len(sentence)):

            bi = sentence[i - 1] +  " " + sentence[i]
            bi = bi.lower()
            if bi in self.bigram:
                self.bigram[bi] += 1
            else:
                self.bigram[bi] = 1
        #for unigram
        for i in range(2,len(sentence)):
            self.inc_uni(sentence[i].lower())
        #self.inc_uni('<EOS>')

    def cond_logprob_linerar_interpolation(self, word, previous):

        V = previous[-1]
        W = word
        UV = previous[-2] + " " + previous[-1]
        VW = previous[-1] + " " + word
        UVW = previous[-2] + " " + previous[-1] + " " + word

        V = V.lower()
        W = W.lower()
        UV = UV.lower()
        VW = VW.lower()
        UVW = UVW.lower()

        C_uvw = 1
        if UVW in self.trigram:
            C_uvw = self.trigram[UVW]

        C_v = 1
        if V in self.unigram:
            C_v = self.unigram[V]

        C_uv = 1
        if UV in self.bigram:
            C_uv = self.bigram[UV]

        C_vw = 1
        if VW in self.bigram:
            C_vw = self.bigram[VW]

        C_w = 1

        if W in self.unigram:
            C_w = self.unigram[W]

        total_word = len(self.vocab())

        """
        q(w_i|W_i-2, w_i-1) = l1 * q_ml(w_i|W_i-2, w_i-1) + l2 * q_ml(w_i|w_i-1)  + l3 * q_ml(w_i)
        = (l1 * C(u,v,w) * C(v) * total_word + l2 C(v,w) * C(u,v) * total_word + l3 * C(w) * C(u,v) * C(v))/ C(u,v) * C(v) * total_word
        put these values
        q_ml(w_i|W_i-2, w_i-1) = C(u,v,w) /C(u,v) 
        q_ml(w_i|w_i-1) = C(v,w) /C(v)
        q_ml(w_i) = C(w) /D (total_word)
        """
        return log(self.l1 * C_uvw * C_v * total_word + self.l2 * C_vw * C_uv * total_word + self.l3 * C_w * C_uv * C_v, 2) \
               - log(C_uv, 2) - log(C_v,2) - log(total_word, 2)

    def cond_logprob_laplace(self, word, previous):

        UV = previous[-2] + " " + previous[-1]
        UVW = previous[-2] + " " + previous[-1] + " " + word
        UV = UV.lower()
        UVW = UVW.lower()


        if UVW.lower() in self.trigram:
            C_uvw = self.trigram[UVW]
        else:
            C_uvw = 0

        if UV in self.bigram:
            C_uv = self.bigram[UV]
        else:
            C_uv = 0

        total_word = len(self.vocab())
        l_v = self.l * total_word

        if l_v == 0 :
            if C_uv == 0 and C_uvw == 0:
                score = 0
            elif C_uvw == 0 :
                score = self.lbackoff - log((C_uv + l_v), 2)
            elif C_uv == 0:
                score = log((C_uvw + self.l), 2) - self.lbackoff
            else:
                score = log(C_uvw, 2) - log(C_uv, 2)
        else:
            score = log((C_uvw + self.l), 2) - log((C_uv + l_v), 2)

        return score
    def vocab(self):
        return self.unigram.keys()
    def pre_processes(self,corpus):
        #add <sos>, <sos>, and <eos> to sentence
        for s in corpus:
            s.insert(0, '<SOS>')
            s.insert(0, '<SOS>')
            s.append("<EOS>")
    def save_model(self):
        import json
        file = open("dict_trigram", 'w+')
        json.dump(self.trigram, file)
        file.close()
        import json
        file = open("dict_bigram", 'w+')
        json.dump(self.bigram, file)
        file.close()
        import json
        file = open("dict_unigram", 'w+')
        json.dump(self.unigram, file)
        file.close()

    def load_model(self):
        import json
        file = open("dict_trigram", 'r+')
        self.trigram = json.load(file)
        file.close()
        import json
        file = open("dict_bigram", 'r+')
        self.bigram = json.load(file)
        file.close()
        import json
        file = open("dict_unigram", 'r+')
        self.unigram = json.load(file)
        file.close()
