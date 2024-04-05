import numpy as np
import pandas as pd
from deepface import DeepFace


class UsersDB:
    def __init__(self, model):
        # Tworzenie pustego DataFrame
        self.model = model
        self.data = pd.DataFrame(columns=["id", "face_repr"])

    def CosDist(self, source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def add_record(self, id_, img_path):
        # Dodawanie nowego rekordu do DataFrame
        pred = DeepFace.represent(
            img_path=img_path, model_name=self.model, enforce_detection=False
        )
        if pred is None:
            print("Face not detected!")
            return False

        face_repr = pred[0]["embedding"]
        new_row = pd.DataFrame({"id": [id_], "face_repr": [face_repr]})
        self.data = pd.concat([self.data, new_row], ignore_index=True)
        # print(f"Pomyślnie dodano nową osobe {id_=} :) ")

    def verify_user(self, img_path, identity, threshold):
        input_face_repr = DeepFace.represent(
            img_path=img_path, model_name=self.model, enforce_detection=False
        )[0]["embedding"]
        if identity not in self.data["id"].values:
            print(f"Nie ma takiej osoby w bazie danych {identity=}")
            return None, False

        target_identity_face_repr = self.data[self.data["id"] == identity][
            "face_repr"
        ].item()
        cos_dist = self.CosDist(input_face_repr, target_identity_face_repr)
        authorized = cos_dist < threshold
        return cos_dist, authorized

    def save_db(self, path):
        self.data.to_csv(path, index=False)

    def load_db(self, path):
        self.data = pd.read_csv(path)
