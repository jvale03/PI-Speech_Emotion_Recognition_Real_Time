import numpy as np

def normalize(audio):
    return audio / np.max(np.abs(audio))

def segment_by_duration(audio, sr, segment_duration):
    samples_per_segment = int(segment_duration * sr)
    return [audio[i:i+samples_per_segment] for i in range(0, len(audio), samples_per_segment)]

def pre_process(audio, sr, segment_duration=2.0):
    audio_norm = normalize(audio)
    segments = segment_by_duration(audio_norm, sr, segment_duration)
    return segments, sr