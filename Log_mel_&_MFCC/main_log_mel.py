import time
import numpy as np
import pyaudio
from ai_edge_litert import interpreter
from collections import deque
from scipy.signal import resample_poly
from gpiozero import LED
import wave

MODEL_PATH = "./model_log_mel.tflite"

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
TARGET_RATE = 16000
DEVICE_INDEX = 0

WINDOW_DURATION_S = 1.0
STEP_DURATION_S = 0.5

CHUNK = int(RATE * STEP_DURATION_S)

BUFFER_MAX_LEN = int(WINDOW_DURATION_S / STEP_DURATION_S)

LED_GO = LED(26)
LED_STOP = LED(6)
LED_ON = LED(27)
LED_OFF = LED(4)
LED_NO = LED(5)

CLASS_NAMES = np.array(['go', 'no', 'off', 'on', 'stop', '_unknown_'])

print("Carregando modelo TFLite...")

interpreter = interpreter.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Modelo carregado com sucesso.")
print("Input shape:", input_details[0]['shape'])
print("Output shape:", output_details[0]['shape'])

audio = pyaudio.PyAudio()

stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=CHUNK
)

LED_GO.off()
LED_STOP.off()
LED_ON.off()
LED_OFF.off()
LED_NO.off()

LED_GO.on()
LED_STOP.on()
LED_ON.on()
LED_OFF.on()
LED_NO.on()

time.sleep(5)

LED_GO.off()
LED_STOP.off()
LED_ON.off()
LED_OFF.off()
LED_NO.off()

print("\nIniciando inferência com janela deslizante... (Pressione Ctrl+C para parar)")

audio_buffer = deque(maxlen=BUFFER_MAX_LEN)

LAST_DETECTION_TIME = 0
COOLDOWN_S = 1.5

try:
    while True:
        # 1. Ler 0.5s de áudio (CHUNK)
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        # 2. Adicionar ao buffer
        audio_buffer.append(data)

        # 3. Se o buffer ainda não estiver cheio (primeiro 1s),
        # pular a inferência e continuar enchendo.
        if len(audio_buffer) < BUFFER_MAX_LEN:
            continue
            
        # 4. Concatenar os 2 blocos de 0.5s para formar 1.0s de áudio
        full_buffer_bytes = b''.join(audio_buffer)
        
        # 5. Converter buffer para numpy (int16) - 1.0s a 48kHz
        audio_data = np.frombuffer(full_buffer_bytes, dtype=np.int16)

        # 6. Resample de 48kHz para 16kHz
        resampled_data = resample_poly(audio_data, TARGET_RATE, RATE)

        # 7. Normalizar para float32
        resampled_data_float = resampled_data.astype(np.float32) / 32768.0

        # 8. PAdicionar dimensão de batch
        input_data = np.expand_dims(resampled_data_float, axis=0).astype(np.float32)
        
        # 9. Executar a inferência
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # 10. Obter resultados
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction_index = np.argmax(output_data[0])
        predicted_class = CLASS_NAMES[prediction_index]
        confidence = np.max(output_data[0])

        # 11. Aplicar "Cooldown" para evitar detecções duplicadas
        current_time = time.time()
        
        if predicted_class not in ['_unknown_'] and (current_time - LAST_DETECTION_TIME > COOLDOWN_S):
            print(f"DETECÇÃO: {predicted_class} (Confiança: {confidence:.2f})")

            if confidence >= 0.75:
                # Acionar LEDs conforme a detecção
                if predicted_class == 'go' :
                    LED_GO.on()
                    time.sleep(1)
                    LED_GO.off()
                elif predicted_class == 'stop':
                    LED_STOP.on()
                    time.sleep(1)
                    LED_STOP.off()
                elif predicted_class == 'on':
                    LED_ON.on()
                    time.sleep(1)
                    LED_ON.off()
                elif predicted_class == 'off':
                    LED_OFF.on()
                    time.sleep(1)
                    LED_OFF.off()
                elif predicted_class == 'no':
                    LED_NO.on()
                    time.sleep(1)
                    LED_NO.off()
                
            LAST_DETECTION_TIME = current_time # Reinicia o cooldown

except KeyboardInterrupt:
    print("\nParando a gravação.")

finally:
    print("Liberando recursos...")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    print("Pronto.")