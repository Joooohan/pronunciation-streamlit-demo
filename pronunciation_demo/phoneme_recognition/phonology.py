import os
import re
from typing import List

import cmudict
import pandas as pd

folder = os.path.dirname(__file__)
mapping_file = os.path.join(folder, "mapping.csv")
cmu = cmudict.dict()


def remove_duplicates(sentence: List[str]) -> List[str]:
    """Remove repeated phones `dd` -> `d`."""
    if len(sentence) < 2:
        return sentence
    else:
        c1 = sentence[0]
        c2 = sentence[1]
        if c1 == c2:
            return remove_duplicates([c1] + sentence[2:])
        else:
            return [c1] + remove_duplicates(sentence[1:])


def convert(orig_list: List[str], orig: str, dest: str) -> List[str]:
    """Convert timit specific phones to arpabet `dcl` -> `d`."""
    data = pd.read_csv(mapping_file).fillna("")
    mapping_dict = {data[orig][i]: data[dest][i] for i in range(len(data))}
    mapping_dict[""] = ""
    dest_list = list(map(lambda x: mapping_dict[x], orig_list))
    dest_list = remove_duplicates(dest_list)
    dest_list = [p for p in dest_list if p != ""]  # remove empty symbols
    return dest_list


def transcribe(sentence: str) -> List[str]:
    """Transcribe a sentence into a sequence of phonemes."""
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:"]'
    sentence = re.sub(chars_to_ignore_regex, "", sentence).lower()
    words = sentence.split(" ")
    transcription = list(map(lambda x: cmu[x][0], words))
    word_lengths = list(map(lambda x: len(x), transcription))
    transcription = sum(transcription, [])
    pattern = re.compile(r"\d")
    transcription = list(map(lambda x: pattern.sub("", x), transcription))
    transcription = list(map(lambda x: x.lower(), transcription))
    assert len(transcription) == sum(word_lengths)
    return transcription, word_lengths
