import streamlit as st
from PIL import Image
import os
import cv2
from speech_users_db import SpeechUsersDB
from pathlib import Path


LANG = "english"
USER_DB_PATH = Path("data/speech/audio_registration_db.pkl")



# Funkcja do tworzenia nowego profilu
def create_profile(db, login):
    with st.spinner(f'Tworzenie profilu dla {login}...'):
        if db.add_record(login, "tmp/uploaded_registry_audio.wav"):
            st.success(f"Nowy profil dla {login} został utworzony!")
            return db
        else:
            st.error(f"Nowy profil dla {login} nie został utworzony!")
            return db

# Funkcja do autoryzacji
def authenticate(db, login):
    with st.spinner(f'Autoryzacja profilu dla {login}...'):
        result = db.verify_user("tmp/uploaded_auth_audio.wav", login)
        if result[1]:
            print(f"Similarity {result[0]}")
            st.success(f"Witaj {login}!  Cos_dist = {result[0]:.4}")
            return db
        else:
            print(f"Similarity {result[0]}")
            st.error(f"Błąd autoryzacji dla użytkownika {login}! Cos_dist = {result[0]:.4}")
            return db
    

def main():
    db = SpeechUsersDB(lang=LANG)
    db.load_db(USER_DB_PATH)
    st.title("Aplikacja do autoryzacji za pomocą audio")

    # Wybór akcji: Tworzenie profilu vs. Autoryzacja
    action = st.radio("Wybierz akcję:", ("Stwórz nowy profil", "Autoryzuj"))

    if action == "Stwórz nowy profil":
        st.header("Stwórz nowy profil")
        name = st.text_input("Podaj login:")
        audio_path = st.file_uploader("Wgraj audio:", type=["wav"])
        
        if name and audio_path:
            st.audio(audio_path)
            with open(os.path.join('tmp', 'uploaded_registry_audio.wav'), 'wb') as f:
                f.write(audio_path.read())
            db = create_profile(db, name)
            db.save_db(USER_DB_PATH)
            os.remove(os.path.join('tmp', 'uploaded_registry_audio.wav'))
            print(len(db.data))

    elif action == "Autoryzuj":
        st.header("Autoryzacja")
        name = st.text_input("Podaj login:")
        audio_auth = st.file_uploader("Wgraj audio:", type=["wav"])
        if name and audio_auth:
            st.audio(audio_auth)
            with open(os.path.join('tmp', 'uploaded_auth_audio.wav'), 'wb') as f:
                f.write(audio_auth.read())
            db = authenticate(db, name)
            os.remove(os.path.join('tmp', 'uploaded_auth_audio.wav'))

if __name__ == "__main__":
    main()