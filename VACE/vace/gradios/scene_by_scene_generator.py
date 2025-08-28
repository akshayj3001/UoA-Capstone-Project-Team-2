import requests
import json
import re
import pprint


def parse_video_scene_summary(text):
    # Extract rows using regex
    rows = re.findall(r"\| (\d+) \| (.*?) \| (.*?) \| (.*?) \| (.*?) \| (.*?) \|", text, re.DOTALL)

    # Convert each row to a dictionary
    scene_data = []
    for row in rows:
        scene = {
            "#": row[0].strip(),
            "Title": row[1].strip(),
            "Visual Description": row[2].strip(),
            "Camera / Lighting": row[3].strip(),
            "Audio Design": row[4].strip(),
            "Generation Prompt": row[5].strip()
        }
        pprint.pprint(scene)
        scene_data.append(scene)

    return scene_data

def generate_scenes_summary(transcript, audio_features):
    url = "http://127.0.0.1:11434/api/generate"

    # Format the prompt to include both transcript and audio features
    prompt = (
        "Generate a structured video scene summary based on the following inputs:\n\n"
        f"Transcript:\n{transcript}\n\n"
        f"Audio Features:\n{json.dumps(audio_features, indent=2)}\n\n"
        "Respond in structured scenes with clear generation prompts."
    )

    payload = {
        "model": "gpt-oss:20b",
        "prompt": prompt,
        "stream": False  # Optional, defaults to False for /api/generate
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json().get("response", "")
        print(result)
        return parse_video_scene_summary(result)
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def generate_scenes_summary_by_chat_endpoint(transcript, audio_features):
    url = "http://127.0.0.1:11434/api/chat"

    # Format the prompt to include both transcript and audio features
    prompt = (
        "Generate a structured video scene summary based on the following inputs:\n\n"
        f"Transcript:\n{transcript}\n\n"
        f"Audio Features:\n{json.dumps(audio_features, indent=2)}\n\n"
        "Respond in structured scenes with clear generation prompts."
    )

    payload = {
        "model": "gpt-oss:20b",
        "messages": [
            {"role": "system", "content": "Always respond in structured scenes with clear generation prompts."},
            {"role": "user", "content": prompt}
        ],
        "stream": True  # Important: Ollama streams responses
    }

    response = requests.post(url, json=payload, stream=True)

    # Collect streamed content
    full_response = ""
    for line in response.iter_lines(decode_unicode=True):
        if line:
            try:
                event = json.loads(line)
                if "message" in event:
                    full_response += event["message"]["content"]
            except json.JSONDecodeError:
                continue  # Skip malformed lines

    print(full_response)
    return full_response


