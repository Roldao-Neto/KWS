import time
import numpy as np
import pyaudio
import librosa
import tensorflow as tf
from collections import deque
from scipy.signal import resample_poly
from gpiozero import LED
from scipy.fftpack import dct

def process_audio_MFCC(audio_samples):
    SAMPLE_RATE = 16000
    N_FFT = 512
    HOP_LENGTH = 160
    N_MELS = 64

    if not isinstance(audio_samples, np.ndarray):
        raise TypeError("audio_samples must be a numpy.ndarray.")
    if audio_samples.ndim != 1:
        raise ValueError("audio_samples must be a 1-D array.")

    if audio_samples.dtype != np.float32:
        audio_samples = audio_samples.astype(np.float32)
    audio_samples = audio_samples / max(np.max(np.abs(audio_samples)), 1e-6)

    if audio_samples.shape[0] < SAMPLE_RATE:
        pad_width = SAMPLE_RATE - audio_samples.shape[0]
        audio_samples = np.pad(audio_samples, (0, pad_width), mode="constant")
    else:
        audio_samples = audio_samples[:SAMPLE_RATE]

    audio = tf.convert_to_tensor(audio_samples, dtype=tf.float32)

    stft = tf.signal.stft(audio, frame_length=N_FFT, frame_step=HOP_LENGTH, fft_length=N_FFT)
    spectrogram = tf.abs(stft)

    num_spectrogram_bins = spectrogram.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=20,
        upper_edge_hertz=SAMPLE_RATE / 2,
    )

    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    means = tf.math.reduce_mean(log_mel_spectrogram)
    stds = tf.math.reduce_std(log_mel_spectrogram)
    log_mel_spectrogram = (log_mel_spectrogram - means) / (stds + 1e-6)

    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :13]
    mfccs = tf.expand_dims(tf.expand_dims(mfccs, axis=-1), axis=0)

    return mfccs.numpy().astype(np.float32)

MODEL_PATH = "./model_MFCC.tflite"

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

audio = pyaudio.PyAudio()

stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=CHUNK
)

print("Carregando modelo TFLite...")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Modelo carregado com sucesso.")
print("Input shape:", input_details[0]['shape'])

print("\nIniciando inferência com janela deslizante... (Pressione Ctrl+C para parar)")

audio_buffer = deque(maxlen=BUFFER_MAX_LEN)

LAST_DETECTION_TIME = 0
COOLDOWN_S = 1.5

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

        # 8. Pré-processar (MFCCs)
        preprocessed_data = process_audio_MFCC(resampled_data_float)
        
        # 9. Executar a inferência
        interpreter.set_tensor(input_details[0]['index'], preprocessed_data)
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
                if predicted_class == 'go':
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