import json
from pathlib import Path

import numpy as np
import pandas as pd
from deepface import DeepFace


class UsersDB:
    def __init__(self, model):
        # Tworzenie pustego DataFrame
        self.model = model
        self.data = pd.DataFrame(columns=["id", "face_repr"])

        with open("./data/exp_img_embeddings.json", "rb") as f:
            self.cached_embeddings = json.load(f)

    def CosDist(self, source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def add_record(self, id_, img_path):
        # Dodawanie nowego rekordu do DataFrame
        if id_ in self.data['id'].values:
            print("Osoba już znajduje się w bazie danych!")
            return False
        pred = DeepFace.represent(
            img_path=img_path, model_name=self.model, enforce_detection=False
        )
        if pred is None:
            print("Twarz nie wykryta!")
            return False

        face_repr = pred[0]["embedding"]
        new_row = pd.DataFrame({"id": [id_], "face_repr": [face_repr]})
        self.data = pd.concat([self.data, new_row], ignore_index=True)
        print(f"Pomyślnie dodano nową osobe {id_=} :) ")
        return True

    def verify_user(self, img_path, identity, threshold = 0.3, cache=False):
        if identity not in self.data["id"].astype('str').values:
                print(f"Nie ma takiej osoby w bazie danych {identity=}")
                return None, False
        if cache:
            img_fname = Path(img_path).name
            input_face_repr = self.cached_embeddings[self.model][img_fname]
        else:
            input_face_repr = self.get_img_embedding(img_path)
        target_identity_face_repr = self.data[self.data["id"].astype('str') == identity][
            "face_repr"
        ].item()
        cos_dist = self.CosDist(input_face_repr, target_identity_face_repr)
        authorized = cos_dist < threshold
        return cos_dist, authorized

    def get_img_embedding(self, img_path):
        return DeepFace.represent(
            img_path=img_path, model_name=self.model, enforce_detection=False
        )[0]["embedding"]

    def save_db(self, path):
        self.data.to_pickle(path)

    def load_db(self, path):
        self.data = pd.read_pickle(path)
