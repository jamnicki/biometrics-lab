import json
from pathlib import Path

import numpy as np
import pandas as pd
import wespeaker


class SpeechUsersDB:
    def __init__(self, lang: str = "english"):
        self.model = wespeaker.load_model(lang)
        self.data = pd.DataFrame(columns=["id", "speech_repr"])

        with open("../data/speech/exp_speech_embeddings.json", "rb") as f:
            self.cached_embeddings = json.load(f)

    def CosDist(self, source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def get_embedding(self, audio_fpath):
        try:
            return self.model.extract_embedding(audio_fpath)
        except Exception as e:
            print("Błąd podczas wykrywania głosu!")
            print(e)

    def add_record(self, id_, audio_fpath, cache=False):
        # Dodawanie nowego rekordu do DataFrame
        if id_ in self.data['id'].astype('str').values:
            print("Osoba już znajduje się w bazie danych!")
            return False
        if cache:
            speech_repr = self.cached_embeddings[str(audio_fpath)]
        else:
            speech_repr = self.get_embedding(audio_fpath)

        new_row = pd.DataFrame({"id": [id_], "speech_repr": [speech_repr]})
        self.data = pd.concat([self.data, new_row], ignore_index=True)
        print(f"Pomyślnie dodano nową osobe {id_=} :) ")
        return True

    def verify_user(self, audio_path, identity, threshold = 0.3, cache=False):
        if str(identity) not in self.data["id"].astype('str').values:
                print(f"Nie ma takiej osoby w bazie danych {identity=}")
                return None, False
        if cache:
            input_speech_repr = self.cached_embeddings[str(audio_path)]
        else:
            input_speech_repr = self.get_embedding(audio_path)
            
        target_identity_speech_repr = self.data[self.data["id"].astype('str') == str(identity)][
            "speech_repr"
        ].item()
        cos_dist = self.CosDist(input_speech_repr, target_identity_speech_repr)
        authorized = cos_dist < threshold
        return cos_dist, authorized

    def save_db(self, path):
        self.data.to_pickle(path)

    def load_db(self, path):
        self.data = pd.read_pickle(path)
