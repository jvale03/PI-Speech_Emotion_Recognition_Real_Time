import numpy as np
import parselmouth
import pandas as pd
import os
import librosa
import soundfile as sf
import shutil

### Pre Processing ###
######################

def normalize(audio):
    return audio / np.max(np.abs(audio))

def segment_by_duration(audio, sr, segment_duration):
    samples_per_segment = int(segment_duration * sr)
    return [audio[i:i+samples_per_segment] for i in range(0, len(audio), samples_per_segment)]

def pre_process(audio, sr, segment_duration=2.0):
    audio_norm = normalize(audio)
    segments = segment_by_duration(audio_norm, sr, segment_duration)
    return segments, sr


### Features Extraction ###
###########################

def extract_features(file_path, mood_label=None):
    signal, sample_rate = librosa.load(file_path, sr=None)
    snd = parselmouth.Sound(file_path)

    # Pitch
    pitches, magnitudes = librosa.piptrack(y=signal, sr=sample_rate)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
    pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0

    # Jitter & Shimmer
    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
    jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer_local = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    # HNR
    hnr = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr_value = parselmouth.praat.call(hnr, "Get mean", 0, 0)

    # energy
    rms_energy = librosa.feature.rms(y=signal)[0]
    energy_mean = np.mean(rms_energy)
    energy_std = np.std(rms_energy)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Speech rate
    onset_env = librosa.onset.onset_strength(y=signal, sr=sample_rate)
    _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sample_rate)
    syllable_count = len(beat_frames)
    duration_sec = librosa.get_duration(y=signal, sr=sample_rate)
    speech_rate = syllable_count / duration_sec if duration_sec > 0 else 0

    # Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=signal)[0]
    zcr_mean = np.mean(zcr)

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)[0]
    spectral_flux = np.mean(np.diff(spectral_centroid))
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate)[0]
    rolloff_mean = np.mean(spectral_rolloff)

    feature_dict = {
        'File': os.path.basename(file_path),
        'Speech Duration': duration_sec,
        'Pitch': pitch_mean,
        'Pitch Std': pitch_std,
        'Speech Rate': speech_rate,
        'Jitter': jitter_local,
        'Shimmer': shimmer_local,
        'Energy Mean': energy_mean,
        'Energy Std': energy_std,
        'HNR': hnr_value,
        'ZCR': zcr_mean,
        'Spectral Flux': spectral_flux,
        'Spectral Centroid Mean': np.mean(spectral_centroid),
        'Spectral Rolloff Mean': rolloff_mean,
        'Mood': mood_label if mood_label is not None else 'unknown'
    }

    for i in range(13):
        feature_dict[f'mfcc_{i}'] = mfcc_mean[i]

    return feature_dict


### Pre Processing & Feature Extraction ###
###########################################

def process_and_extraction(file_path, segment_duration=2.0, mood_label=None):
    signal, sr = librosa.load(file_path, sr=None)
    # Pre Processing
    segments, sr = pre_process(signal, sr, segment_duration)

    segment_features = []

    for i, segment in enumerate(segments):
        # Ignore short segmets
        if len(segment) < int(segment_duration * sr * 0.5):
            continue

        # Salva segmento temporário para usar com parselmouth
        temp_path = f"temp_segment_{i}.wav"
        sf.write(temp_path, segment, sr)

        try:
            features = extract_features(temp_path, mood_label=mood_label)
            features['Segment'] = i
            features['Original File'] = os.path.basename(file_path)
            segment_features.append(features)
        except Exception as e:
            print(f"Error in segment {i} from {file_path}: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return segment_features


### Loop to Process Multiple Audio Files ###
############################################

def batch_extract_from_main_folder(main_folder, output_csv='features.csv', segment_duration=2.0):
    all_features = []


    for mood_label in os.listdir(main_folder):
        print(f"Processing {mood_label}...")
        mood_path = os.path.join(main_folder, mood_label)
        if not os.path.isdir(mood_path):
            continue  


        for filename in os.listdir(mood_path):
            if not filename.lower().endswith(('.wav')):
                continue

            file_path = os.path.join(mood_path, filename)
            try:
                features_list = process_and_extraction(file_path, segment_duration, mood_label=mood_label)
                all_features.extend(features_list)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")


    df = pd.DataFrame(all_features)
    df.to_csv(output_csv, index=False)
    print(f"All features saved in: {output_csv}")


### Organize Files By Emotions ###
##################################

Crema_folder = "../Raw_DataSet/Crema"
Ravdess_folder = "../Raw_DataSet/Ravdess"
Savee_folder = "../Raw_DataSet/Savee"
Tess_folder = "../Raw_DataSet/Tess"

# keyword → destination folder
Crema_rules = {
    "hap": "../Emotions/Happy",
    "ps": "../Emotions/Surprised",
    "neu": "../Emotions/Neutral",
    "sad": "../Emotions/Sad",
    "dis": "../Emotions/Disgust",
    "fea": "../Emotions/Fear",
    "ang": "../Emotions/Angry"
}
Ravdess_rules = {
    "01": "../Emotions/Neutral",
    "02": "../Emotions/Calm",
    "03": "../Emotions/Happy",
    "04": "../Emotions/Sad",
    "05": "../Emotions/Angry",
    "06": "../Emotions/Fear",
    "07": "../Emotions/Disgust",
    "08": "../Emotions/Surprised"
}
Savee_rules = {
    "h": "../Emotions/Happy",
    "su": "../Emotions/Surprised",
    "n": "../Emotions/Neutral",
    "sa": "../Emotions/Sad",
    "d": "../Emotions/Disgust",
    "f": "../Emotions/Fear",
    "a": "../Emotions/Angry"
}
Tess_rules = {
    "hap": "../Emotions/Happy",
    "ps": "../Emotions/Surprised",
    "neu": "../Emotions/Neutral",
    "sad": "../Emotions/Sad",
    "dis": "../Emotions/Disgust",
    "fea": "../Emotions/Fear",
    "ang": "../Emotions/Angry"
}

def audio_files_by_emotion():
    # Ensure all destination folders exist
    for destination_path in Crema_rules.values():
        os.makedirs(destination_path, exist_ok=True)

    for root, _, files in os.walk(Crema_folder):
        for filename in files:
            filename_lower = filename.lower()
            for keyword, destination_path in Crema_rules.items():
                if keyword in filename_lower:
                    source_path = os.path.join(root, filename)
                    target_path = os.path.join(destination_path, filename)
                    print(f"Copying: {source_path} → {target_path}")
                    shutil.copy2(source_path, target_path)
                    break 

    for destination_path in Ravdess_rules.values():
        os.makedirs(destination_path, exist_ok=True)

    for root, _, files in os.walk(Ravdess_folder):
        for filename in files:
            if filename.endswith(".wav"):
                parts = filename.split("-")
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    emotion_label = Ravdess_rules.get(emotion_code)
                    if emotion_label:
                        source_path = os.path.join(root, filename)
                        target_path = os.path.join(emotion_label, filename)
                        print(f"Copying: {source_path} → {target_path}")
                        shutil.copy2(source_path, target_path)

    # Ensure all destination folders exist
    for destination_path in Savee_rules.values():
        os.makedirs(destination_path, exist_ok=True)

    for root, _, files in os.walk(Savee_folder):
        for filename in files:
            filename_lower = filename.lower()
            for keyword, destination_path in Savee_rules.items():
                if keyword in filename_lower:
                    source_path = os.path.join(root, filename)
                    target_path = os.path.join(destination_path, filename)
                    print(f"Copying: {source_path} → {target_path}")
                    shutil.copy2(source_path, target_path)
                    break 

    # Ensure all destination folders exist
    for destination_path in Tess_rules.values():
        os.makedirs(destination_path, exist_ok=True)

    for root, _, files in os.walk(Tess_folder):
        for filename in files:
            filename_lower = filename.lower()
            for keyword, destination_path in Tess_rules.items():
                if keyword in filename_lower:
                    source_path = os.path.join(root, filename)
                    target_path = os.path.join(destination_path, filename)
                    print(f"Copying: {source_path} → {target_path}")
                    shutil.copy2(source_path, target_path)
                    break 