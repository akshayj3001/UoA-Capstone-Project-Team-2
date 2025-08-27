"""
transcript_to_video_prompt.py

Description:
This script generates video prompts from transcribed text and key audio features using Open AI's GPT-OSS 20b model.

Author: Akshay Jayakumar
Date: 10 August 2025
Version: v1.0
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
import re
import json

from LambdaException import LambdaException


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


MODEL = None
TOKENIZER = None
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompt.txt")



try:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        PROMPT = f.read()
except FileNotFoundError as excp:
    log.error(f"Prompt file not found: {str(excp)}")
    raise

def load_model():

    try:
        # Using the GPT-OSS 20b model
        model_id = "openai/gpt-oss-20b"
        global MODEL, TOKENIZER
        
        # Loading the tokenizer and model
        TOKENIZER = AutoTokenizer.from_pretrained(model_id)
        MODEL = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="cuda"
        )
        log.info("Model and tokenizer loaded")
    
    except Exception as excp:
        log.error(f"Error loading model or tokenizer: {str(excp)}")
        raise LambdaException(f"Error loading model or tokenizer: {str(excp)}") from excp
    



def generate_video_prompt(transcript, audio_features):
    """
    Generates a video prompt from the transcript and audio features.

    Parameters:
        transcript (str): Transcribed text.
        audio_features (dict): Extracted audio features.
    """

    log.info("Initiating prompt generation")
    try:
        global MODEL, TOKENIZER, PROMPT

        if not MODEL or not TOKENIZER:
            load_model()
        else:
            log.info("Model and tokenizer pre-loaded")

        prompt = PROMPT.format(transcript=transcript, audio_summary=audio_features)
        message = [
            {"role": "system", "content": "Always respond in structured scenes with clear generation prompts."},
            {"role": "user", "content": prompt},
        ]
        log.info("Prompt formatted with transcript and audio features")

        inputs = TOKENIZER.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(MODEL.device)

        # Generate output
        generated = MODEL.generate(**inputs, max_new_tokens=3000)
        video_prompt = TOKENIZER.decode(generated[0][inputs["input_ids"].shape[-1]:])

        log.info(f"Generated video prompt: \n{video_prompt}")

        match = re.search(r"\[.*\]", video_prompt, re.DOTALL)

        if match:
            json_str = match.group(0)   # Extracted JSON string

        return json_str

    except Exception as excp:
        log.error(f"Error occurred in prompt generation: {str(excp)}")
        raise LambdaException(f"Error occurred in prompt generation: {str(excp)}") from excp

