
import csv
import numpy as np
import os

class SplitsBuilder(object):
    def __init__(self, train_test_splits_file):
        BASE_DIR = os.getenv("BASE_DIR")
        self._train_test_splits_file = os.path.join(BASE_DIR,train_test_splits_file)
        self._splits = {}

    def train_split(self):
        return self._splits["train"]

    def test_split(self):
        return self._splits["test"]

    def val_split(self):
        return self._splits["val"]

    def _parse_train_test_splits_file(self):
        
        with open(self._train_test_splits_file, "r") as f:
            data = [row for row in csv.reader(f)]
        return np.array(data)

    def get_splits(self, keep_splits=["train, val"]):
        if not isinstance(keep_splits , list):
            try:
                keep_splits = list(keep_splits[:])
            except:
                keep_splits = [keep_splits]
        # Return only the split
        s = []
        for ks in keep_splits:
            s.extend(self._parse_split_file()[ks])
        return s


class CSVSplitsBuilder(SplitsBuilder):
    def _parse_split_file(self):
        if not self._splits:
            data = self._parse_train_test_splits_file()
            for s in ["train", "test", "val", "overfit"]:
                self._splits[s] = [r[0] for r in data if r[1] == s]
        return self._splits
