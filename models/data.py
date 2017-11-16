#!/bin/python
import numpy as np
import sys


# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

def read_file():
    file = open("train.txt", 'r')
    train_txt = file.read()
    file.close()
    file = open("test.txt", 'r')
    test_txt = file.read()
    file.close()

    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    count_vect.fit(train_txt.split("\n"))
    tokenizer = count_vect.build_tokenizer()

    class Data:
        pass

    data = Data()
    data.train = []
    for s in train_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.train.append(toks)
    data.test = []
    for s in test_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.test.append(toks)

    print("train:", len(data.train), "test:", len(data.test))
    return data
def read_texts(tarfname, dname):
    """Read the data from the homework data file.

    Given the location of the data archive file and the name of the
    dataset (one of brown, reuters, or gutenberg), this returns a
    data object containing train, test, and dev data. Each is a list
    of sentences, where each sentence is a sequence of tokens.
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz", errors = 'replace')
    train_mem = tar.getmember(dname + ".train.txt")
    train_txt = unicode(tar.extractfile(train_mem).read(), errors='replace')
    test_mem = tar.getmember(dname + ".test.txt")
    test_txt = unicode(tar.extractfile(test_mem).read(), errors='replace')
    dev_mem = tar.getmember(dname + ".dev.txt")
    dev_txt = unicode(tar.extractfile(dev_mem).read(), errors='replace')

    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    count_vect.fit(train_txt.split("\n"))
    tokenizer = count_vect.build_tokenizer()
    class Data: pass
    data = Data()
    data.train = []
    for s in train_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.train.append(toks)
    data.test = []
    for s in test_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.test.append(toks)
    data.dev = []
    for s in dev_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.dev.append(toks)
    print(dname," read.", "train:", len(data.train), "dev:", len(data.dev), "test:", len(data.test))
    return data

def learn_unigram(data):
    """Learns a unigram model from data.train.

    It also evaluates the model on data.dev and data.test, along with generating
    some sample sentences from the model.
    """
    from lm import Unigram
    unigram = Unigram()
    unigram.fit_corpus(data.train)
    print("vocab:", len(unigram.vocab()))
    # evaluate on train, test, and dev
    print("train:", unigram.perplexity(data.train))
    print("dev  :", unigram.perplexity(data.dev))
    print("test :", unigram.perplexity(data.test))
    from generator import Sampler
    sampler = Sampler(unigram)
    print("sample: ", " ".join(str(x) for x in sampler.sample_sentence([])))
    print("sample: ", " ".join(str(x) for x in sampler.sample_sentence([])))
    print("sample: ", " ".join(str(x) for x in sampler.sample_sentence([])))
    return unigram

def learn_bigram(data):
    """Learns a unigram model from data.train.

    It also evaluates the model on data.dev and data.test, along with generating
    some sample sentences from the model.
    """
    from lm import Bigram
    bigram = Bigram()
    bigram.fit_corpus(data.train)
    print("vocab:", len(bigram.vocab()))
    # evaluate on train, test, and dev
    print("train:", bigram.perplexity(data.train))

    # add <sos>, <sos>, and <eos> to validation and test data
    bigram.pre_processes(data.dev)
    bigram.pre_processes(data.test)
    print("dev  :", bigram.perplexity(data.dev))
    print("test :", bigram.perplexity(data.test))
    # from generator import Sampler
    # sampler = Sampler(trigram)
    # print("sample: ", " ".join(str(x) for x in sampler.sample_sentence(['SOS','SOS'])))
    # print("sample: ", " ".join(str(x) for x in sampler.sample_sentence(['SOS','SOS'])))
    # print("sample: ", " ".join(str(x) for x in sampler.sample_sentence(['SOS','SOS'])))
    return bigram

def learn_trigram(data):
    """Learns a unigram model from data.train.

    It also evaluates the model on data.dev and data.test, along with generating
    some sample sentences from the model.
    """
    from lm import Trigram
    trigram = Trigram()
    trigram.fit_corpus(data.train)
    #trigram.save_model()
    #trigram.load_model()
    print("vocab:", len(trigram.vocab()))
    # evaluate on train, test, and dev
    print("train:", trigram.perplexity(data.train))

    # add <sos>, <sos>, and <eos> to validation and test data
    trigram.pre_processes(data.dev)
    trigram.pre_processes(data.test)
    print("dev  :", trigram.perplexity(data.dev))
    print("test :", trigram.perplexity(data.test))
    # from generator import Sampler
    # sampler = Sampler(trigram)
    # print("sample: ", " ".join(str(x) for x in sampler.sample_sentence(['SOS','SOS'])))
    # print("sample: ", " ".join(str(x) for x in sampler.sample_sentence(['SOS','SOS'])))
    # print("sample: ", " ".join(str(x) for x in sampler.sample_sentence(['SOS','SOS'])))
    return trigram

def print_table(table, row_names, col_names, latex_file = None):
    row_format ="{:>15} " * (len(col_names) + 1)
    print(row_format.format("", *col_names))
    for row_name, row in zip(row_names, table):
        print(row_format.format(row_name, *row))

if __name__ == "__main__":
    dnames = ["brown", "reuters", "gutenberg"]
    datas = []
    models = []
    # Learn the models for each of the domains, and evaluate it
    for dname in dnames:
        print("-----------------------")
        print(dname)
        data = read_texts("data/corpora.tar.gz", dname)
        datas.append(data)
        #model = learn_unigram(data)
        #model = learn_bigram(data)
        model = learn_trigram(data)
        models.append(model)
    # compute the perplexity of all pairs
    n = len(dnames)
    perp_dev = np.zeros((n,n))
    perp_test = np.zeros((n,n))
    perp_train = np.zeros((n,n))
    for i in xrange(n):
        for j in xrange(n):
            perp_dev[i][j] = models[i].perplexity(datas[j].dev)
            perp_test[i][j] = models[i].perplexity(datas[j].test)
            perp_train[i][j] = models[i].perplexity(datas[j].train)

    print("-------------------------------")
    print("x train")
    print_table(perp_train, dnames, dnames, "table-train.tex")
    print("-------------------------------")
    print("x dev")
    print_table(perp_dev, dnames, dnames, "table-dev.tex")
    print("-------------------------------")
    print("x test")
    print_table(perp_test, dnames, dnames, "table-test.tex")

# if __name__ == "__main__":
#
#     data = read_file()
#     model = learn_bigram(data)
#     #model = learn_trigram(data)