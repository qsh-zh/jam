#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : matching.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/02/2017
#
# Qinsheng modified based on Jacinle.
# Distributed under terms of the MIT license.


import collections
import fnmatch
import re

__all__ = ["NameMatcher", "IENameMatcher"]


class NameMatcher:
    def __init__(self, rules=None):
        if rules is None:
            self._rules = []
        elif isinstance(rules, dict):
            self._rules = list(rules.items())
        else:
            assert isinstance(rules, collections.Iterable)
            self._rules = list(rules)

        self._map = {}
        self._compiled_rules = []
        self._compiled = False

        self._matched = []
        self._unused = set()
        self._last_stat = None

    @property
    def rules(self):
        return self._rules

    def map(self):
        assert self._compiled
        return self._map

    def append_rule(self, rule):
        self._rules.append(tuple(rule))

    def insert_rule(self, index, rule):
        self._rules.insert(index, rule)

    def pop_rule(self, index=None):
        self._rules.pop(index)

    def begin(self, *, force_compile=False):
        if not self._compiled or force_compile:
            self.compile()
        self._matched = []
        self._unused = set(range(len(self._compiled_rules)))

    def end(self):
        return self._matched, {self._compiled_rules[i][0] for i in self._unused}

    def match(self, k):
        for i, (rule_str, re_rule, rule_value) in enumerate(self._compiled_rules):
            if re_rule.match(k):
                if i in self._unused:
                    self._unused.remove(i)
                self._matched.append((k, rule_str, rule_value))
                return rule_value
        return None

    def compile(self):
        self._map = dict()
        self._compiled_rules = []

        for rule_str, rule_value in self._rules:
            self._map[rule_str] = rule_value
            re_rule = fnmatch.translate(
                rule_str
            )  # shell-style pattern converted to regular expression
            re_rule = re.compile(re_rule, flags=re.IGNORECASE)
            self._compiled_rules.append((rule_str, re_rule, rule_value))
        self._compiled = True

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._last_stat = self.end()

    def get_last_stat(self):
        return self._last_stat


class IENameMatcher:
    def __init__(self, include=None, exclude=None):
        if include is None:
            self.include = None
        else:
            self.include = NameMatcher([(i, True) for i in include])

        if exclude is None:
            self.exclude = None
        else:
            self.exclude = NameMatcher([(e, True) for e in exclude])
        self._last_stat = None

    def begin(self):
        if self.include is not None:
            self.include.begin()
        if self.exclude is not None:
            self.exclude.begin()
        self._last_stat = (set(), set())

    def end(self):
        if self.include is not None:
            self.include.end()
        if self.exclude is not None:
            self.exclude.end()

        if len(self._last_stat[0]) < len(self._last_stat[1]):
            self._last_stat = ("included", self._last_stat[0])
        else:
            self._last_stat = ("excluded", self._last_stat[1])

    def match(self, k):
        if self.include is None:
            ret = True
        else:
            ret = bool(self.include.match(k))

        if self.exclude is not None:
            ret = ret and not bool(self.exclude.match(k))

        if ret:
            self._last_stat[0].add(k)
        else:
            self._last_stat[1].add(k)
        return ret

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()

    def get_last_stat(self):
        return self._last_stat
