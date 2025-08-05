"""
audio_to_video_pipeline.py

Author: Sujit Udhane

Description:
This script implements a modular pipeline for converting human speech audio into 
scene-by-scene video using machine learning models. It integrates audio transcription, 
feature extraction, summarization, and video generation using the VACE framework.

Modules Used:
- Librosa: For audio feature extraction
- Whisper: For speech-to-text transcription
- Flan-T5: For context-aware summarization and visual prompt generation
- MoviePy: For video stitching
- Subprocess: For invoking external video generation scripts

Device Compatibility:
Supports Apple M1/M2 chips using Metal Performance Shaders (MPS) if available.
"""

import os
import subprocess
import numpy as np
import librosa
import whisper
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import html
from moviepy import VideoFileClip, concatenate_videoclips
import argparse

# Automatically select device (MPS for Mac, else CPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"

def transcribe_and_extract_features(audio_path):
    """
    Transcribes audio and extracts key features such as MFCCs, pitch, and energy.

    Parameters:
        audio_path (str): Path to the input audio file.

    Returns:
        transcript (str): Transcribed text from audio.
        audio_features (dict): Dictionary containing average MFCCs, pitch, and energy.
    """
    print("🔊 Loading audio and extracting features...")
    y, sr = librosa.load(audio_path, sr=16000)

    # MFCCs (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    avg_mfcc = np.mean(mfccs, axis=1)

    # Pitch estimation using PYIN
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    avg_pitch = np.nanmean(f0)

    # Energy (Root Mean Square)
    energy = librosa.feature.rms(y=y)[0]
    avg_energy = np.mean(energy)

    # Save MFCC plot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig("mfcc_plot.png")
    plt.close()

    print("🧠 Loading Whisper model...")
    model = whisper.load_model("base")
    print(f"📝 Transcribing audio file: {audio_path}")
    result = model.transcribe(audio_path)
    transcript = result["text"]
    print("✅ Transcription complete.\n")

    audio_features = {
        "avg_mfcc": avg_mfcc.tolist(),
        "avg_pitch": float(avg_pitch) if not np.isnan(avg_pitch) else None,
        "avg_energy": float(avg_energy)
    }

    print("✅ Audio Features \n", audio_features)
    return transcript, audio_features


def clean_scene_output(raw_scenes):
    """
    Cleans and deduplicates scene outputs.

    Parameters:
        raw_scenes (list): List of raw scene strings.

    Returns:
        cleaned_scenes (list): List of cleaned and numbered scenes.
    """
    cleaned_scenes = []
    seen = set()
    scene_counter = 1

    for raw in raw_scenes:
        cleaned = re.sub(r"Scene \d+:\s*Scene \d+:\s*", "", raw).strip()
        if "narrator talks to the camera" in cleaned.lower() or cleaned in seen:
            continue
        cleaned_scenes.append(f"Scene {scene_counter}: {cleaned}")
        seen.add(cleaned)
        scene_counter += 1

    return cleaned_scenes

def summarize_with_google_flan_with_chunking(transcript, audio_features):
    """
    Summarizes transcript into scene-by-scene video prompts using Flan-T5.

    Parameters:
        transcript (str): Transcribed text.
        audio_features (dict): Extracted audio features.

    Returns:
        final_scenes (list): List of structured scene descriptions.
    """
    print("Loading Flan-T5 model...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")

    sentences = re.split(r'(?<=[.!?]) +', transcript)
    transcript_chunks = [' '.join(sentences[i:i+10]) for i in range(0, len(sentences), 10)]

    all_scenes = []
    seen_scenes = set()
    scene_counter = 1

    for idx, chunk in enumerate(transcript_chunks):
        print(f"\nProcessing chunk {idx+1}/{len(transcript_chunks)}...")

        audio_summary = (
            f"Average pitch: {audio_features['avg_pitch']}, "
            f"energy: {audio_features['avg_energy']}, "
            f"MFCC: {audio_features['avg_mfcc'][:5]}"
        )

        prompt = (
            f"Write a scene-by-scene video script based on the following transcript. "
            f"For each scene, describe what the viewer sees, what text appears on screen, and what the narrator says. "
            f"Use the audio features to highlight emotional or important moments.\n\n"
            f"Transcript:\n{chunk}\n\nAudio Features:\n{audio_summary}"
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        outputs = model.generate(**inputs, max_new_tokens=1024, num_beams=4, early_stopping=True)
        story = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        story = html.unescape(story)

        scenes = re.split(r"(Scene \d+:)", story)
        for i in range(1, len(scenes), 2):
            scene_text = scenes[i] + scenes[i+1]
            scene_text = scene_text.strip()
            if "narrator speaks to the camera" in scene_text.lower() or scene_text in seen_scenes:
                continue
            all_scenes.append(f"Scene {scene_counter}: {scene_text}")
            seen_scenes.add(scene_text)
            scene_counter += 1

    final_scenes = clean_scene_output(all_scenes)
    print("\n✅ Final Scene-by-Scene Script:\n")
    for scene in final_scenes:
        print(scene.strip(), "\n")

    return final_scenes


def generate_video_per_scene(scenes, output_dir, output_file_name):
    """
    Generates individual video clips for each scene and stitches them together.

    Parameters:
        scenes (list): List of scene descriptions.
        output_dir (str): Directory to save intermediate and final video files.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_clips = []

    for i, scene in enumerate(scenes, start=1):
        scene_text = f"Scene {i}: {scene}"
        prompt_file = os.path.join(output_dir, f"scene_{i}_prompt.txt")
        output_video = os.path.join(output_dir, f"scene_{i}.mp4")

        with open(prompt_file, "w") as f:
            f.write(scene_text)

        command = [
            "python", "Wan2.1-Mac/generate.py",
            "--task", "t2v-1.3B",
            "--size", "832*480",
            "--frame_num", "17",
            "--sample_steps", "20",
            "--ckpt_dir", "Wan2.1-Mac/Wan2.1-T2V-1.3B",
            "--offload_model", "True",
            "--device", device,
            "--prompt", scene_text,
            "--save_file", output_video
        ]

        try:
            subprocess.run(command, check=True)
            video_clips.append
        except subprocess.CalledProcessError as e:
            print(f"❌ Error generating scene {i}: {e}")

    # Stitch all clips
    final_video = concatenate_videoclips(video_clips)
    final_video.write_videofile(os.path.join(output_dir, output_file_name))

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio to Video Pipeline")
    parser.add_argument("--input_file", type=str, required=True, help="Input audio file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output video and intermediate files")
    parser.add_argument("--output_file_name", type=str, required=True, help="Name of the final output video file")

    args = parser.parse_args()

    audio_file_path = args.input_file  # Use input_file argument from CLI
    output_dir = args.output_dir
    output_file_name = args.output_file_name

    # Step 1: Transcribe and extract features
    transcript, audio_features = transcribe_and_extract_features(audio_file_path)
    print("Transcript:\n", transcript)
    print("Audio Features:\n", audio_features)

    # Step 2: Summarize using BART with audio features
    story = summarize_with_google_flan_with_chunking(transcript, audio_features)
    print("\nGenerated Story:\n", story)
   
    # Step 3: Generate video
    generate_video_per_scene(story, output_dir, output_file_name)


