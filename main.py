import streamlit as st
from PIL import Image
import os
import cv2
from users_db import UsersDB
from pathlib import Path

# MODEL = "VGG-Face"
# USER_DB_PATH = Path("./data/user_db.pkl")
MODEL = "Facenet"
USER_DB_PATH = Path("./data/user_db_facenet.pkl")



# Funkcja do tworzenia nowego profilu
def create_profile(db, login):
    if db.add_record(login, "tmp/uploaded_registry_image.jpg"):
        st.success(f"Nowy profil dla {login} został utworzony!")
        return db
    else:
        st.error(f"Nowy profil dla {login} nie został utworzony!")
        return db

# Funkcja do autoryzacji
def authenticate(db, login):
    print(len(db.data))
    result = db.verify_user("tmp/uploaded_auth_image.jpg", login)
    if result[1]:
        print(f"Similarity {result[0]}")
        st.success(f"Witaj {login}!")
        return db
    else:
        print(f"Similarity {result[0]}")
        st.error(f"Błąd autoryzacji dla użytkownika {login}!")
        return db
    

def main():
    db = UsersDB(model=MODEL)
    db.load_db(USER_DB_PATH)
    st.title("Aplikacja do autoryzacji za pomocą zdjęć twarzy")

    # Wybór akcji: Tworzenie profilu vs. Autoryzacja
    action = st.radio("Wybierz akcję:", ("Stwórz nowy profil", "Autoryzuj"))

    if action == "Stwórz nowy profil":
        st.header("Stwórz nowy profil")
        name = st.text_input("Podaj login:")
        image = st.file_uploader("Wgraj zdjęcie:", type=["jpg", "png", "jpeg"])
        
        if name and image:
            st.image(image, use_column_width=True)
            with open(os.path.join('tmp', 'uploaded_registry_image.jpg'), 'wb') as f:
                f.write(image.read())
            db = create_profile(db, name)
            db.save_db(USER_DB_PATH)
            os.remove(os.path.join('tmp', 'uploaded_registry_image.jpg'))
            print(len(db.data))

    elif action == "Autoryzuj":
        st.header("Autoryzacja")
        name = st.text_input("Podaj login:")
        image_auth = st.file_uploader("Wgraj zdjęcie:", type=["jpg", "png", "jpeg"])
        if name and image_auth:
            st.image(image_auth, use_column_width=True)
            with open(os.path.join('tmp', 'uploaded_auth_image.jpg'), 'wb') as f:
                f.write(image_auth.read())
            db = authenticate(db, name)
            os.remove(os.path.join('tmp', 'uploaded_auth_image.jpg'))

if __name__ == "__main__":
    main()