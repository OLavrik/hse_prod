import spacy
import contextualSpellCheck
# pip install cyhunspell
from hunspell import HunSpell
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from sklearn.linear_model import LogisticRegression
import textdistance



class SpellChecker(object):
    def __init__(self, dic_path, aff_path):
        self.hunspell = HunSpell(dic_path, aff_path)

    def count_errors(self, sentence):
        pass

    def predict_candidate(self, sentence):
        pass

    def format_sentence(self, sentence):
        pass


class SpacySpellChecker(SpellChecker):
    def __init__(self, dic_path, aff_path):
        super().__init__(dic_path, aff_path)
        self.nlp = spacy.load('en_core_web_sm')
        contextualSpellCheck.add_to_pipe(self.nlp)

    def count_errors(self, sentence):
        doc = self.nlp(sentence)
        return len(doc._.suggestions_spellCheck)

    def predict_candidate(self, sentence):
        doc = self.nlp(sentence)
        return doc._.suggestions_spellCheck

    def format_sentence(self, sentence):
        doc = self.nlp(sentence)
        return doc._.outcome_spellCheck


class MyOwnSpellChecker(SpellChecker):
    def __init__(self, dic_path, aff_path, use_any_embedding=False):
        super().__init__(dic_path, aff_path)
        self.word_vect = KeyedVectors.load_word2vec_format(
            "./SO_vectors_200.bin", binary=True)
        self.use_any_embedding = use_any_embedding

    def prepare_data(self, data):
        pass

    def get_dist(self, word, candidate):
        try:
            return self.word_vect.distance(word, candidate)
        except:
            return -1

    def get_features(self, candidate, word):
        max_len = max(len(word), len(candidate))
        levenshtein = (1 - textdistance.damerau_levenshtein(word, candidate)) / max_len
        jaro_winkler = (1 - textdistance.jaro_winkler(word, candidate)) / max_len
        len_lcsstr = len(textdistance.lcsstr(word, candidate)) / max_len
        if self.use_any_embedding:
            dist = self.get_dist(word, candidate)
            return np.asarray([levenshtein, jaro_winkler, len_lcsstr, dist])
        return [levenshtein, jaro_winkler, len_lcsstr]

    def fit(self, incorrect_words, correct_words):
        X_train_list, y_train_list = [], []
        for word, good_candidate in (zip(incorrect_words, correct_words), "Build features", len(incorrect_words)):
            X_train_list.append(self.get_features(good_candidate, word))
            y_train_list.append(1)

            bad_candidate = None
            for _bc in self.hunspell.suggest(word):
                if _bc != good_candidate:
                    bad_candidate = _bc
                    break
            if bad_candidate is not None:
                X_train_list.append(self.get_features(bad_candidate, word))
                y_train_list.append(0)

        X_train = np.asarray(X_train_list)
        y_train = np.asarray(y_train_list)
        self._clr = LogisticRegression(n_jobs=-1, random_state=self.__seed)
        self._clr.fit(X_train, y_train)

    def suggest(self, word):
        word_candidates = self.hunspell.suggest(word)
        if len(word_candidates) == 0:
            return word_candidates, np.asarray([])
        features = np.vstack([self.get_features(it, word) for it in word_candidates])
        return word_candidates, features

    def count_errors(self, sentence):
        errors = 0
        for word in sentence.split(" "):
            word_candidates, features = self.suggest(word)
            if len(word_candidates) != 0:
                errors += 1
        return errors

    def predict_candidate(self, sentence):
        res = {}
        for word in sentence.split(" "):
            word_candidates, features = self.suggest(word)
            if len(word_candidates) == 0:
                res[word] = {}
            else:
                scores = self._clr.predict_proba(features)[:, 1]
                dict_ = {}
                for val, prob in zip(word_candidates, scores):
                    dict_[val] = prob
                res[word] = dict_
        return res

    def format_sentence(self, sentence):
        new_sentence = sentence
        res = {}
        for word in sentence.split(" "):
            word_candidates, features = self.suggest(word)
            if len(word_candidates) == 0:
                res[word] = {}
            else:
                scores = self._clr.predict_proba(features)[:, 1]
                max_ = 0
                val_max = word
                for val, prob in zip(word_candidates, scores):
                    if prob > max_:
                        max_ = prob
                        val_max = val
                new_sentence.replace(word, val_max)
        return new_sentence


if __name__ == "__main__":
    data_folder = "resource"
    dic_path = "./resource/index.dic"
    aff_path = "./resource/index.aff"
    sp = SpacySpellChecker(dic_path, aff_path)
    my = MyOwnSpellChecker(dic_path, aff_path)

    sp.count_errors('noww')
    sp.format_sentence("noww")
    sp.predict_candidate("noww")

    incorrect_words, correct_words = [], []
    with open("./resource/spell-errors.txt") as train_data:
        for line in train_data, "Preprocessing":
            correct, misspells = line.strip().split(":")
            correct = correct.strip()
            incorrect = misspells.strip().split(",")[0].strip()
            if not correct.isalpha():
                continue
            correct_words.append(correct)
            incorrect_words.append(incorrect)

    my.fit(incorrect_words, correct_words)
    my.count_errors('noww')
    my.format_sentence("noww")
    my.predict_candidate("noww")
