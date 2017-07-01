from __future__ import print_function, division

from collections import Counter, defaultdict


class Instance(object):
    def __init__(self, attribs, gold):
        """attribs is a list of active attributes for the instance
        """
        self.attribs = attribs
        self.gold = gold


class Perceptron(object):
    def __init__(self, instances):
        self.instances = instances
        self.ninstances = len(instances)

        klasses = Counter()
        for inst in self.instances:
            klasses[inst.gold] += 1
        self.klasses = list(sorted(klasses))

        self.weights = Counter()
        self.total = Counter()
        # Track when each weight was last updated for efficient averaging
        self.last = defaultdict(int)

    def _score(self, inst, klass):
        return sum(self.weights[attrib, klass] for attrib in inst.attribs)

    def _classify(self, inst):
        return max((self._score(inst, klass), klass)
                   for klass in self.klasses)[1]

    def _update_weights(self, inst, klass, value):
        for attrib in inst.attribs:
            feat = (attrib, klass)
            self.total[feat] += self.weights[feat] * (self.nupdates -
                                                      self.last[feat])
            self.last[feat] = self.nupdates
            self.weights[feat] += value

    def _update(self, inst):
        self.nupdates += 1
        predicted = self._classify(inst)
        if predicted != inst.gold:
            self._update_weights(inst, inst.gold, +1.0)
            self._update_weights(inst, predicted, -1.0)
        return predicted == inst.gold

    def _iteration(self):
        ncorrect = sum(self._update(inst) for inst in self.instances)
        return ncorrect/float(self.ninstances)

    def _average(self, niterations):
        divisor = float(self.ninstances*niterations)
        for feat in self.weights:
            self.total[feat] += self.weights[feat] * (self.nupdates -
                                                      self.last[feat])
            self.weights[feat] = self.total[feat]/divisor

    def train(self, niterations):
        self.nupdates = 0
        for i in range(niterations):
            accuracy = self._iteration()
            print(accuracy)
        self._average(niterations)

    def test(self, instances):
        for inst in instances:
            inst.predicted = self._classify(inst)