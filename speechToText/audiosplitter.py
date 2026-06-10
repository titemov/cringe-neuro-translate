import os
import re
import subprocess
import tempfile

# --- CONFIGURATION ---
AUDIO_FILE = "test.mp3"
SRT_FILE = "output.srt"
OUTPUT_DIR = "speaker_outputs"


def parse_time_to_seconds(time_str):
    """Converts SRT timestamp (HH:MM:SS,mmm) to seconds (float)."""
    hours, minutes, seconds = time_str.split(":")
    seconds, milliseconds = seconds.split(",")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds) / 1000.0
    )


def parse_srt(srt_path):
    """Parses SRT file flexibly, ignoring source tags and handling varied line breaks."""
    with open(srt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    speaker_segments = {}
    current_start = None
    current_end = None

    # Timestamp regex helper (matches 'HH:MM:SS,mmm --> HH:MM:SS,mmm')
    time_pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})"
    )
    # Speaker regex helper (matches '[Speaker X]:')
    speaker_pattern = re.compile(r"\[(Speaker\s+\d+)\]:")

    for line in lines:
        line = line.strip()

        # 1. Look for the timestamp line
        time_match = time_pattern.search(line)
        if time_match:
            current_start, current_end = time_match.groups()
            continue

        # 2. Look for the speaker line (only if we recently found a timestamp)
        if current_start and current_end:
            speaker_match = speaker_pattern.search(line)
            if speaker_match:
                speaker = speaker_match.group(1)

                start_secs = parse_time_to_seconds(current_start)
                end_secs = parse_time_to_seconds(current_end)
                duration = end_secs - start_secs

                if duration > 0:
                    if speaker not in speaker_segments:
                        speaker_segments[speaker] = []
                    speaker_segments[speaker].append((start_secs, duration))

                # Reset timestamps so we look for the next block
                current_start = None
                current_end = None

    return speaker_segments


def process_speaker_audio(audio_path, speaker, segments, output_dir):
    """Extracts and concatenates segments for a single speaker using FFmpeg."""
    if not segments:
        return

    print(f"Processing {speaker} ({len(segments)} segments)...")
    os.makedirs(output_dir, exist_ok=True)

    # Clean speaker name for file naming (e.g., "Speaker 0" -> "Speaker_0")
    safe_speaker_name = speaker.replace(" ", "_")
    final_output_path = os.path.join(output_dir, f"{safe_speaker_name}.mp3")

    # Temporary directory to hold cut pieces
    with tempfile.TemporaryDirectory() as temp_dir:
        concat_list_path = os.path.join(temp_dir, "concat_list.txt")

        with open(concat_list_path, "w", encoding="utf-8") as concat_file:
            for i, (start, duration) in enumerate(segments):
                temp_segment_path = os.path.join(temp_dir, f"seg_{i}.mp3")

                # FFmpeg command to extract a specific segment
                # -ss before -i enables fast seeking
                cmd_extract = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(start),
                    "-t",
                    str(duration),
                    "-i",
                    audio_path,
                    "-acodec",
                    "copy",  # Stream copy avoids re-encoding quality loss
                    "-loglevel",
                    "error",
                    temp_segment_path,
                ]

                subprocess.run(cmd_extract, check=True)

                # Write the absolute path to the concatenation list
                # Escaping single quotes for ffmpeg safe formatting
                safe_path = temp_segment_path.replace("'", "'\\''")
                concat_file.write(f"file '{safe_path}'\n")

        # FFmpeg command to stitch all segments together seamlessly
        cmd_concat = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_path,
            "-acodec",
            "copy",
            "-loglevel",
            "error",
            final_output_path,
        ]

        subprocess.run(cmd_concat, check=True)
    print(f"Saved: {final_output_path}")


def main():
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: Audio file '{AUDIO_FILE}' not found.")
        return
    if not os.path.exists(SRT_FILE):
        print(f"Error: SRT file '{SRT_FILE}' not found.")
        return

    print("Parsing SRT file...")
    speaker_data = parse_srt(SRT_FILE)

    if not speaker_data:
        print("No speaker segments found. Check your SRT formatting.")
        return

    for speaker, segments in speaker_data.items():
        process_speaker_audio(AUDIO_FILE, speaker, segments, OUTPUT_DIR)

    print("\nDone! All speaker voices have been successfully separated.")


if __name__ == "__main__":
    main()