import os
import time
import random
from io import BytesIO

import cv2
import mediapipe as mp
import numpy as np
import requests
import simpleaudio as sa
from openai import OpenAI
from pydub import AudioSegment


# =======================================================
#   CONFIG
# =======================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "YOUR_ELEVEN_KEY")

VOICE_ID = "ftXV24GhUyFDyLJLFU7x"
MODEL_ID = "eleven_multilingual_v2"

TEXT_SPEED = 2
PHRASE_INTERVAL = 6.0          # cada 6 segundos
FONT_SCALE = 1.4               # texto grande

openai_client = OpenAI(api_key=OPENAI_API_KEY)


# =======================================================
#   PIXEL + BLACK & WHITE
# =======================================================

def pixel_bw(frame, pixel_size=22):
    h, w, _ = frame.shape

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    small = cv2.resize(gray, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    _, bw = cv2.threshold(big, 128, 255, cv2.THRESH_BINARY)

    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)



# =======================================================
#   EMOTION COLOR
# =======================================================

def emotion_color(text):
    t = text.lower()

    if any(k in t for k in ["glitch", "weird", "strange", "extraño", "rare"]):
        return (255, 60, 255)
    if any(k in t for k in ["dream", "sueño", "memoria", "quieto"]):
        return (120, 180, 255)
    if any(k in t for k in ["robot", "machine", "error"]):
        return (255, 220, 40)
    return (255, 255, 255)



# =======================================================
#   TYPEWRITER TEXT
# =======================================================

def wrap_text(text, max_chars=36):
    words = text.split(" ")
    lines = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur += " " + w if cur else w
    if cur:
        lines.append(cur)
    return lines


def draw_typewriter(frame, text, idx, box):
    x, y, w, h = box
    col = emotion_color(text)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    frame[y:y+h, x:x+w] = cv2.addWeighted(
        overlay[y:y+h, x:x+w], 0.65,
        frame[y:y+h, x:x+w], 0.35, 0
    )

    lines = wrap_text(text)
    flat = " ".join(lines)
    sub = flat[:idx]

    reveal_lines = wrap_text(sub)

    y_text = y + 50
    for line in reveal_lines:
        cv2.putText(frame, line, (x + 40, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, col, 2, cv2.LINE_AA)
        y_text += int(40 * FONT_SCALE)

    idx = min(len(flat), idx + TEXT_SPEED)
    return idx


# =======================================================
#   ELEVEN LABS TTS
# =======================================================

def speak(text):
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
        headers = {
            "xi-api-key": ELEVEN_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        }
        payload = {"text": text, "model_id": MODEL_ID}

        r = requests.post(url, headers=headers, json=payload)
        if r.status_code != 200:
            print("TTS error:", r.status_code, r.text[:200])
            return

        audio = AudioSegment.from_file(BytesIO(r.content), format="mp3")
        wav_buf = BytesIO()
        audio.export(wav_buf, format="wav")
        wav = wav_buf.getvalue()

        wave_obj = sa.WaveObject(
            wav,
            num_channels=audio.channels,
            bytes_per_sample=audio.sample_width,
            sample_rate=audio.frame_rate,
        )
        wave_obj.play()

    except Exception as e:
        print("TTS exception:", e)



# =======================================================
#   OPENAI — GENERATE PHRASE
# =======================================================

SYSTEM_PROMPT = """
Hablas raro, corto y punk.
Spanglish permitido. Política suave.
Nada cursi. Línea única o dos líneas cortas.
Describe el mood del mundo como si fueras un robot cansado.
"""

def generate_phrase():
    try:
        resp = openai_client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Frase breve y extraña, tono cyberpunk."}
            ],
            max_output_tokens=40
        )
        return resp.output[0].content[0].text.strip().replace("\n", " ")
    except:
        return random.choice([
            "la ciudad respira glitch.",
            "todo arde pero seguimos online.",
            "memory leak, corazón quieto."
        ])


# =======================================================
#   MAIN
# =======================================================

def main():
    mp_face = mp.solutions.face_mesh
    face = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("VIOLETA ROBOT", cv2.WINDOW_NORMAL)

    last_text = "sistema cargando..."
    idx = 0
    last_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        styled = pixel_bw(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face.process(rgb)

        h, w, _ = styled.shape

        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            max_lm = len(lms)

            # ====== Eye landmarks (full upper + lower lids)
            LEFT_EYE = [33,133, 159,158,157, 145,144,163]
            RIGHT_EYE = [263,362, 386,385,384, 374,373,380]

            eye_ids = LEFT_EYE + RIGHT_EYE

            for i in eye_ids:
                if i < max_lm:
                    px = int(lms[i].x * w)
                    py = int(lms[i].y * h)
                    cv2.rectangle(styled, (px-3, py-3), (px+3, py+3),
                                  (255,255,255), -1)

        # text box
        box_w = int(w * 0.9)
        box_h = 180
        box_x = (w - box_w) // 2
        box_y = int(h * 0.65)

        now = time.time()
        if now - last_time > PHRASE_INTERVAL:
            last_text = generate_phrase()
            idx = 0
            last_time = now
            speak(last_text)

        idx = draw_typewriter(styled, last_text, idx,
                              (box_x, box_y, box_w, box_h))

        cv2.imshow("VIOLETA ROBOT", styled)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()

