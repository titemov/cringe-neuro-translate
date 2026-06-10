import os
import datetime
import torch
import whisper
from nemo.collections.asr.models import SortformerEncLabelModel

# --- 1. CONFIGURATION ---
AUDIO_PATH = "test.mp3"  # MUST be mono, 16000Hz wav file
OUTPUT_SRT = "output.srt"

# Use GPU if available
device = "cuda" #if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# --- 2. HELPER FUNCTIONS ---
def parse_nemo_segments(predicted_segments):
    """
    Parses NeMo's raw list of strings format:
    [['start end speaker', 'start end speaker']]
    """
    segments = []
    # NeMo diarize returns a nested list
    for speaker_list in predicted_segments:
        for entry in speaker_list:
            start, end, speaker = entry.split()
            segments.append({
                'start': float(start),
                'end': float(end),
                'speaker': speaker.replace("speaker_", "Speaker ")
            })
    return sorted(segments, key=lambda x: x['start'])


def get_speaker_for_time(timestamp, diar_segments):
    """Finds which speaker matches the time using a midpoint calculation."""
    for seg in diar_segments:
        if seg['start'] <= timestamp <= seg['end']:
            return seg['speaker']
    
    # Fallback if time falls exactly into a tiny blank gap
    if diar_segments:
        closest_seg = min(diar_segments, key=lambda x: min(abs(timestamp - x['start']), abs(timestamp - x['end'])))
        return closest_seg['speaker']
    return "Unknown Speaker"


def format_srt_time(seconds):
    """Converts seconds (float) into SRT format: HH:MM:SS,mmm"""
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_int = divmod(remainder, 60)
    milliseconds = int(round((seconds - total_seconds) * 1000))
    
    # Handle tiny math overflow rounding edge cases
    if milliseconds >= 1000:
        milliseconds -= 1000
        seconds_int += 1
        if seconds_int >= 60:
            seconds_int -= 60
            minutes += 1
            if minutes >= 60:
                minutes -= 60
                hours += 1
                
    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d},{milliseconds:03d}"


# --- 3. THE PIPELINE EXECUTION ---
def main():
    if not os.path.exists(AUDIO_PATH):
        print(f"Error: Could not find audio file at '{AUDIO_PATH}'")
        return

    # STEP A: Run NVIDIA NeMo Diarization
    print("Loading NVIDIA NeMo Sortformer model...")
    # NOTE: NeMo pulls this down from Hugging Face automatically
    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
    diar_model = diar_model.to(device)
    
    print("Extracting speaker timestamps (Diarization)...")
    raw_nemo_output = diar_model.diarize(audio=AUDIO_PATH, batch_size=1)
    diar_segments = parse_nemo_segments(raw_nemo_output)
    
    # STEP B: Run OpenAI Whisper (Tiny Model)
    print("Loading OpenAI Whisper (Tiny) model...")
    whisper_model = whisper.load_model("tiny", device=device)
    
    print("Transcribing audio text...")
    whisper_result = whisper_model.transcribe(AUDIO_PATH)
    whisper_segments = whisper_result['segments']

    # STEP C: Combine Outputs & Write to .SRT
    print(f"Aligning speaker timestamps with text and writing to {OUTPUT_SRT}...")
    
    with open(OUTPUT_SRT, "w", encoding="utf-8") as f:
        for index, whisper_seg in enumerate(whisper_segments, start=1):
            # Midpoint strategy to find out who was speaking during this chunk
            midpoint = (whisper_seg['start'] + whisper_seg['end']) / 2
            speaker = get_speaker_for_time(midpoint, diar_segments)
            
            # Format times and string text
            start_str = format_srt_time(whisper_seg['start'])
            end_str = format_srt_time(whisper_seg['end'])
            text = whisper_seg['text'].strip()
            
            # Write standardized SRT block
            f.write(f"{index}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"[{speaker}]: {text}\n\n")
            
    print("Pipeline complete! Open your generated .srt file to view results.")

if __name__ == "__main__":
    main()
