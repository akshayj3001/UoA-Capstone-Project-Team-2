# audio_to_video_pipeline.py

import os
import subprocess
import numpy as np
import librosa
import whisper
import torch
import matplotlib.pyplot as plt
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import html
from moviepy import VideoFileClip, concatenate_videoclips

# Use MPS on Mac M1 if available
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Step 1: Transcribe audio and extract features
def transcribe_and_extract_features(audio_path):
    print("🔊 Loading audio and extracting features...")
    y, sr = librosa.load(audio_path, sr=16000)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    avg_mfcc = np.mean(mfccs, axis=1)

    # Extract pitch
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    avg_pitch = np.nanmean(f0)

    # Extract energy
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

# Step 2: Summarize transcript using BART with audio features
def summarize_with_bart(transcript, audio_features):
    print("🧠 Loading BART summarization model...")
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    prompt = (
        "Summarize the following transcript into a scene-by-scene video script. "
        "Each scene should include visual description, text overlay, and narration. "
        "Use the provided audio features to emphasize emotionally intense or important segments.\n\n"
        f"Transcript:\n{transcript}\n\n"
        f"Audio Features:\n"
        f"- Average Pitch: {audio_features['avg_pitch']}\n"
        f"- Average Energy: {audio_features['avg_energy']}\n"
        f"- MFCC Summary: {audio_features['avg_mfcc'][:5]}...\n"
    )

    inputs = tokenizer.batch_encode_plus(
        [prompt],
        max_length=1024,
        return_tensors="pt",
        truncation=True
    ).to(device)

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=512,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Step 3: Generate video using VACE
#def generate_video_from_story(story_text, output_path="results/output_video.mp4"):
def generate_video_from_story(story_text, output_path="results/output_video.mp4"):
    print("🎬 Generating video from story...")
    os.makedirs("results", exist_ok=True)

    # Convert list to string
    story_string = "\n".join(story_text)

    # Save to file
    prompt_file = "story_prompt.txt"
    with open(prompt_file, "w") as f:
        f.write(story_string)


    # Use the string version in the command
    command = [
        "python", "Wan2.1-Mac/generate.py",
        "--task", "t2v-1.3B",
        "--size", "832*480",
        "--frame_num", "17",
        "--sample_steps", "15",
        "--ckpt_dir", "Wan2.1-Mac/Wan2.1-T2V-1.3B",
        "--offload_model", "True",
        "--device", "mps",
        "--prompt_file", prompt_file,
        "--save_file", output_path
    ]

    try:
        subprocess.run(command, check=True)
        print(f"✅ Video generation complete. Output saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print("❌ Error during video generation:", e)




def generate_video_per_scene(scenes, output_dir="results"):
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
            "--sample_steps", "15",
            "--ckpt_dir", "Wan2.1-Mac/Wan2.1-T2V-1.3B",
            "--offload_model", "True",
            "--device", "mps",
            "--prompt", scene_text,
            "--save_file", output_video
        ]

        try:
            subprocess.run(command, check=True)
            video_clips.append(VideoFileClip(output_video))
        except subprocess.CalledProcessError as e:
            print(f"❌ Error generating scene {i}: {e}")

    # Stitch all clips
    final_video = concatenate_videoclips(video_clips)
    final_video.write_videofile(os.path.join(output_dir, "final_output.mp4"))


        
# Part 4: Summarize transcript using Google FLAN with audio features
def summarize_with_google_flan(transcript, audio_features):
    print("Loading Flan-T5 model...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")

    # Enhanced prompt for structured scene-by-scene output

    prompt = (
        "You are a video scriptwriter. Convert the following transcript into a scene-by-scene video script. "
        "Each scene should include:\n"
        "- Scene Title\n"
        "- Visual Description\n"
        "- Text Overlay (if any)\n"
        "- Narration\n\n"
        "Use the audio features to emphasize emotionally intense or important segments.\n\n"
        f"Transcript:\n{transcript}\n\n"
        f"Audio Features:\n"
        f"- Average Pitch: {audio_features['avg_pitch']}\n"
        f"- Average Energy: {audio_features['avg_energy']}\n"
        f"- MFCC Summary: {audio_features['avg_mfcc'][:5]}...\n\n"
        "Format the output like this:\n"
        "Scene 1:\n"
        "Title: A dramatic opening in the Arctic\n"
        "Visuals: Aerial view of melting ice sheets\n"
        "Text Overlay: 'Arctic Meltdown'\n"
        "Narration: The Arctic ice cap has shrunk dramatically over the past 25 years...\n\n"
        "Scene 2:\n"
        "... and so on."
    )

    # Tokenize and generate output
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        num_beams=4,
        early_stopping=True
    )

    story = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    story = html.unescape(story)  # Converts &gt; to >

    # Optional: Split into scenes
    scenes = re.split(r"(Scene \d+:)", story)
    structured_scenes = []
    for i in range(1, len(scenes), 2):
        structured_scenes.append(scenes[i] + scenes[i+1])


    print("Structured Scene-by-Scene Story:\n")
    for scene in structured_scenes:
        print(scene.strip(), "\n")

    return structured_scenes


def clean_scene_output(raw_scenes):
    cleaned_scenes = []
    seen = set()
    scene_counter = 1

    for raw in raw_scenes:
        # Remove duplicate "Scene X: Scene X:" patterns
        cleaned = re.sub(r"Scene \d+:\s*Scene \d+:\s*", "", raw).strip()

        # Skip generic scenes
        if "narrator talks to the camera" in cleaned.lower():
            continue

        # Skip duplicates
        if cleaned in seen:
            continue

        cleaned_scenes.append(f"Scene {scene_counter}: {cleaned}")
        seen.add(cleaned)
        scene_counter += 1

    return cleaned_scenes


#part-5 

def simple_chunk_transcript(transcript, max_sentences=10):
    sentences = re.split(r'(?<=[.!?]) +', transcript)
    return [' '.join(sentences[i:i+max_sentences]) for i in range(0, len(sentences), max_sentences)]

def summarize_with_google_flan_with_chunking(transcript, audio_features):
    print("Loading Flan-T5 model...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")

    # Split transcript into chunks
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
            f"Use the audio features to highlight emotional or important moments. "
            f"Do not repeat previous scenes. Continue from where the last scene left off.\n\n"
            f"Transcript:\n{chunk}\n\n"
            f"Audio Features:\n{audio_summary}"
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            num_beams=4,
            early_stopping=True
        )

        story = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        story = html.unescape(story)

        print(f"Raw output from chunk {idx+1}:\n{story}\n")

        # Extract scenes
        scenes = re.split(r"(Scene \d+:)", story)
        for i in range(1, len(scenes), 2):
            scene_text = scenes[i] + scenes[i+1]
            scene_text = scene_text.strip()

            # Filter out generic scenes
            if "narrator speaks to the camera" in scene_text.lower():
                continue

            # Avoid duplicates
            if scene_text not in seen_scenes:
                all_scenes.append(f"Scene {scene_counter}: {scene_text}")
                seen_scenes.add(scene_text)
                scene_counter += 1

                
    final_scenes = clean_scene_output(all_scenes)
    for scene in final_scenes:
        print(scene, "\n")

    print("\n✅ Final Scene-by-Scene Script:\n")
    for scene in final_scenes:
        print(scene.strip(), "\n")

    return final_scenes



# Main execution
if __name__ == "__main__":
    audio_file_path = "input_data/AlGore_2009.mp3"  # Replace with your audio file path

    # Step 1: Transcribe and extract features
    #transcript, audio_features = transcribe_and_extract_features(audio_file_path)
    #print("Transcript:\n", transcript)
    #print("Audio Features:\n", audio_features)

    # Step 2: Summarize using BART with audio features
    #story = summarize_with_google_flan_with_chunking(transcript, audio_features)
    #print("\nGenerated Story:\n", story)

    #story=["Scene 1: Last year I showed these two slides that demonstrate that the Arctic ice cap, which for most of the last 3 million years has been the size of the lower 48 states, has shrunk by 40%. This understates the seriousness of this particular problem because it doesn't show the thickness of the ice. The Arctic ice cap is in a sense of the beating heart of the global climate system. It expands in winter and contracts in summer.", "Scene 2: Professor Katie Walter from the University of Alaska went out with another team to another shallow lake last winter. She's okay. The question is whether we will be and one reason is this enormous heat sink heats up Greenland from the north. This is an annual melting river, but the volumes are much larger than ever. This is the Kangaroo Lucic River in southwest Greenland. If you want to know how sea level rises from land-based ice melting, this is where it reaches the sea.", "Scene 3: Drying around the world has led to a dramatic increase in fires. And the disasters around the world have been increasing at an absolutely extraordinary and unprecedented rate. Four times as many in the last 30 years as in the previous 75, this is a completely unsustainable pattern. In the last five years, we've added 70 million tons of CO2 every 24 hours. 25 million tons every day to the oceans. Look carefully at the area of the Eastern Pacific from the Americas extending westward and on either side of the Indian subcontinent where there is a radical depletion of oxygen in the oceans. The biggest single cause of global warming along with deforestation, which is 20% of it, is the burning of fossil fuels. Oil is a problem and coal is the most serious problem.", "Scene 4: The United States is one of the largest emitters along with China and the proposal has been to build a lot more coal plants. But we're beginning to see a sea change. Here are the ones that have been canceled in the last few years with some green alternatives of proposed. However, there is a political battle in our country and the coal industry and the oil industry spent a quarter of a billion dollars in the last calendar year promoting clean coal, which is an oxymoron. That image reminded me of something around Christmas in my home in Tennessee, a billion gallons of coal sludge was spilled. This is probably sawd on the news. This all over the country is the second largest waste stream in America. This happened around Christmas. One of the coal industries adds around the country was this one. This is the source of much of the coal in West Virginia.", "Scene 5: Al Gore, Nancy Pelosi, Harry Reid. They don't know what they're talking about. So the Alliance for Climate Protection has launched two campaigns. This is one of them, part of one of them. The polarity we view climate change is a very serious threat to our business. That's why we've made it our primary goal to spend a large sum of money on an advertising effort to help bring out and complicate the truth about coal.", 'Scene 6: Clean Coal Facility.', "Scene 7: Machineries kind of loud, but that's the sound of clean coal technology.", "Scene 8: Amazing! Machineries kind of loud, but that's the sound of clean coal technology.", "Scene 9: Today's Clean Coal Technology.", "Scene 10: America is in crisis, the economy, national security, the climate crisis, the threat that links them all, our addiction to carbon-based fuels like dirty coal and foreign oil. But now there's a bold new solution to get us out of this mess. We power America with 100% clean electricity within 10 years. A plan to put America back to work, make us more secure and help stop global warming. Finally, a solution that's big enough to solve our problems. Repower America.", "Scene 11: Future's over here. Wind, sun, a new energy grid. New investments to create high-paying jobs. Repower America. It's time to get real. There's an old African proverb that says, if you want to go quickly, go alone. If you want to go far, go together. We need to go far quickly. Thank you very much."]

    story=["Scene 1: Last year I showed these two slides that demonstrate that the Arctic ice cap, which for most of the last 3 million years has been the size of the lower 48 states, has shrunk by 40%. This understates the seriousness of this particular problem because it doesn't show the thickness of the ice. The Arctic ice cap is in a sense of the beating heart of the global climate system. It expands in winter and contracts in summer.", "Scene 2: Professor Katie Walter from the University of Alaska went out with another team to another shallow lake last winter. She's okay. The question is whether we will be and one reason is this enormous heat sink heats up Greenland from the north. This is an annual melting river, but the volumes are much larger than ever. This is the Kangaroo Lucic River in southwest Greenland. If you want to know how sea level rises from land-based ice melting, this is where it reaches the sea."]
    
    # Step 3: Generate video
    #generate_video_from_story(story)
    generate_video_per_scene(story)
