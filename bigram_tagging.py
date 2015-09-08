"""
Implementation of bigram part-of speech (POS) tagger based on first-order hidden
Markov models from scratch.
"""

from __future__ import division
from __future__ import unicode_literals
import numpy as np
import pandas as pd
import codecs


POS_UNIVERSAL = ('VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP',
                 'CONJ', 'DET', 'NUM', 'PRT', 'X', '.')

POS_STATES = np.array(POS_UNIVERSAL)


def viterbi(i_obs, i_states, lstart_p, ltrans_p, lemit_p):
    """
    Return the best path, given an HMM model and a sequence of observations
    :param i_obs: index of observations in obs_states
    :param i_states: index of states
    :param start_p: 2D array of log initial probabilities (requires explicit reshape)
    :param trans_p: 2D array of log transition probabilities
    :param emit_p: 2D array of log emission probabilities
    :return:
    best_path: 1D array best corresponding hidden states to observations
    (internal, published for debugging)
    path: 2D array of best state for each step and hidden state
    logV: 2D array of best log probability for each step and state
    """
    """"""

    n_obs = i_obs.size
    n_states = i_states.size  # number of states
    logV = np.zeros((n_states, n_obs))  # initialise viterbi table
    path = np.zeros((n_states, n_obs), dtype=np.int)  # initialise the best path table
    best_path = np.zeros(n_obs, dtype=np.int)  # this will be your output

    # B- base case
    logV[:, [0]] = lstart_p + lemit_p[:, [i_obs[0]]]
    path[:, 0] = i_states

    # C- Inductive case
    for t in xrange(1, n_obs):  # loop through time
        for s in xrange(0, n_states):  # loop through the states @(t-1)
            tp = logV[:, t-1] + ltrans_p[:, s] + lemit_p[s, i_obs[t]]
            path[s, t], logV[s, t] = tp.argmax(), tp.max()

    # D - Backpoint
    best_path[n_obs - 1] = logV[:, n_obs - 1].argmax() # last state
    for t in xrange(n_obs - 1, 0, -1):  # states of (last-1)th to 0th time step
        best_path[t - 1] = path[best_path[t], t]

    return best_path, path, logV


def read_corpus(file_id):
    """
    Read a corpus in a CLL file format with "words" and "pos" columns
    :param file_id:
    :return:
    """
    f = open(file_id)
    lines = f.readlines()
    f.close()
    words = []  # List of words in corpus
    tags = []  # List of tags corresponding to each word
    n_sents = 0  # Sentences are separated by a empty string
    sents = [[]]  # List of sentences. Each sentence is a list of words
    t_sents = [[]]  # List of corresponding tags for each word in sentences.
    for line in lines:
        split = line.split()
        if len(split) == 2:
            words.append(split[0])
            tags.append(split[1])
            sents[n_sents].append(split[0])
            t_sents[n_sents].append(split[1])
        else:
            if sents[n_sents] != []:
                n_sents += 1
                sents.append([])
                t_sents.append([])
    words = np.array(words)
    tags = np.array(tags)
    if sents[-1] == []:
        sents = sents[:-1]
        t_sents = t_sents[:-1]
    sents = np.array(sents)
    t_sents = np.array(t_sents)
    return words, tags, sents, t_sents


def read_words(file_id):
    """
    Read a corpus in a CLL file format with only "words" column
    :param file_id:
    :return:
    """
    f = open(file_id)
    lines = f.readlines()
    f.close()
    words = []
    n_sents = 0
    sents = [[]]
    for line in lines:
        line = line.strip()
        if line:
            words.append(line)
            sents[n_sents].append(line)
        else:
            if sents[n_sents] != []:
                n_sents += 1
                sents.append([])
    words = np.array(words)
    if sents[-1] == []:
        sents = sents[:-1]
    sents = np.array(sents)
    return words, sents


def write_corpus(file_id, sents, t_sents):
    """
    Writes a Corpus in CLL file format, with "words" and "pos" columns.
    Inserts a empty line between sentences.
    :param file_id:
    :return:
    """
    f = codecs.open(file_id, "w", encoding='utf-8')
    for i, sent in enumerate(sents):
        for j, word in enumerate(sent):
            f.write("{}\t{}\n".format(word.decode('utf-8'), t_sents[i][j]))
        f.write("\n")
    f.close()


def where_in_states(values, states):
    """
    Return a flat array of indexes of occurrences of values array in
    states array.
    :param values:
    :param states:
    :return:
    """
    return np.array([np.where(states == i) for i in values]).flatten()


def testing_viterbi():
    """
    Example taken from Borodovsky & Ekisheva (2006), pp 80-81
    :return:
    """
    states = np.array(['H', 'L'])
    i_states = np.arange(0, states.size)
    obs = np.array(['G', 'G', 'C', 'A', 'C', 'T', 'G', 'A', 'A'])
    obs_states = np.array(['A', 'C', 'G', 'T'])
    i_obs = where_in_states(obs, obs_states)
    start_p = np.array([0.5, 0.5]).reshape((states.size, 1))
    trans_p = np.array([[0.5, 0.5],
                        [0.4, 0.6]])
    emit_p = np.array([[0.2, 0.3, 0.3, 0.2],
                       [0.3, 0.2, 0.2, 0.3]])
    lstart_p = np.log(start_p)
    ltrans_p = np.log(trans_p)
    lemit_p = np.log(emit_p)
    best_path, path, logV = viterbi(i_obs, i_states, lstart_p, ltrans_p, lemit_p)
    print(states[best_path])
    print(states[path])
    print(logV)


def bigrams(array):
    """
    Returns an array of bigrams given a 1D array of words or tags.
    :param array:
    :return:
    """

    return np.array([(array[i:i+2]) for i in xrange(len(array) - 1)])


def train(file_id):
    """
    Estimate HMM model parameters using maximum likelihood method, i.e.
    Calculating relative frequency distributions.

    :param file_id: tagged corpus file in CLL format
    :return:
    start_p: frequency of tags of first word in each sentence.
             array POS_STATES.size
    trans_p: frequency of tags from one state to another for each bigram.
             matrix POS_STATES.size x POS_STATES.size
    emit_p: frequency of words for each tag.
            matrix POS_STATES.size x unique_words.size
    unique_words: array of unique words in corpus
    """
    # read corpus data
    words, tags, sents, t_sents = read_corpus(file_id)
    t_bigrams = bigrams(tags)

    # Calculate frequency of tags of first word in each sentence.
    t_first = [t_sent[0] for t_sent in t_sents]
    start_f = np.zeros(POS_STATES.size, dtype=np.int)
    start_f = pd.DataFrame(start_f)
    start_f.index = POS_STATES
    for tag in t_first:
        start_f.loc[tag, 0] += 1

    # Calculate frequency between states in bigrams
    trans_f = np.zeros((POS_STATES.size, POS_STATES.size), dtype=np.int)
    trans_f = pd.DataFrame(trans_f)
    trans_f.index = POS_STATES
    trans_f.columns = POS_STATES
    for i, j in t_bigrams:
        trans_f.loc[i, j] += 1

    # Calculate frequency of each word by tag
    unique_words = np.unique(words)
    emit_f = np.zeros((POS_STATES.size, unique_words.size), dtype=np.int)
    emit_f = pd.DataFrame(emit_f)
    emit_f.index = POS_STATES
    emit_f.columns = unique_words
    for tag, word in zip(tags, words):
        emit_f.loc[tag, word] += 1

    return start_f.values, trans_f.values, emit_f.values, unique_words


def freq2prob(start_f, trans_f, emit_f):
    """
    Convert frequencies in probabilities

    :param start_f:
    :param trans_f:
    :param emit_f:
    :return:
    """

    start_p = np.zeros(start_f.shape)
    start_p = start_f / sum(start_f)
    trans_p = np.zeros(trans_f.shape)
    for i in xrange(POS_STATES.size):
        trans_p[i, :] = trans_f[i, :] / np.sum(trans_f[i, :])
    emit_p = np.zeros(emit_f.shape)
    for i in xrange(POS_STATES.size):
        emit_p[i, :] = emit_f[i, :] / np.sum(emit_f[i, :])
    return start_p, trans_p, emit_p


def generate_model(file_id, model_id):
    """
    Estimate model form data given in file_id, and save parameters in
    model_id file.
    :return:
    """

    start_f, trans_f, emit_f, obs_states = train(file_id)
    np.savez(model_id, start_f=start_f, trans_f=trans_f,
             emit_f=emit_f, states=POS_STATES, obs_states=obs_states)


def add_one_smoothing(emit_f, obs_states, words):
    """
    Assign frequency of one to each new word that doesn't appeared on train
    data.
    :param emit_p:
    :param obs_states:
    :param: words
    :return:
    """

    new_words = []
    for word in words:
        if not(word in obs_states) and not(word in new_words):
            new_words.append(word)
    obs_states = np.append(obs_states, new_words)
    new_words_f = np.zeros((emit_f.shape[0], len(new_words)))
    emit_f = np.append(emit_f, new_words_f, axis=1)
    emit_f += 1  # Add one!
    return emit_f, obs_states


def load_model(model_id):
    """

    :param model_id:
    :return:
    """

    model = np.load("{}.npz".format(model_id))
    start_f = model["start_f"]
    trans_f = model["trans_f"]
    emit_f = model["emit_f"]
    obs_states = model["obs_states"]
    return start_f, trans_f, emit_f, obs_states


def evaluate_model(file_id, start_f, trans_f, emit_f, obs_states, smooth):
    """
    Evaluate model in model_id for corpus given in file_id and generate
    output_id file of ConLL file format.

    :param file_id: eval corpus file in CLL format, without tags
    :param model_id: hmm model in npz format
    :param output_id: result corpus file in CLL format
    :return:
    Generate new corpus file output_id in CLL format.
    """

    words, sents = read_words(file_id)
    i_states = np.arange(0, POS_STATES.size)
    emit_f, obs_states = smooth(emit_f, obs_states, words)
    start_p, trans_p, emit_p = freq2prob(start_f, trans_f, emit_f)
    lstart_p = np.log(start_p.reshape((start_p.size, 1)))
    ltrans_p = np.log(trans_p)
    lemit_p = np.log(emit_p)
    # For each sentence as observations, obtain tags using viterbi
    t_sents = []
    for sent in sents:
        i_obs = where_in_states(sent, obs_states)
        best_path, path, logV = viterbi(i_obs, i_states,
                                        lstart_p, ltrans_p, lemit_p)
        t_sents.append(POS_STATES[best_path].tolist())
    return sents, t_sents

