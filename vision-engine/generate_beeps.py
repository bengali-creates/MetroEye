"""
Generate Beep Sounds (Standalone)
==================================

Generates beep WAV files without requiring pygame for playback
Just creates the files, doesn't play them
"""

import wave
import struct
import math
from pathlib import Path


def create_beep_sound(
    filename: str,
    frequency: int = 1000,
    duration: float = 0.5,
    repeat: int = 1,
    gap: float = 0.2,
    volume: float = 0.8
):
    """
    Create a beep sound WAV file

    Args:
        filename: Output filename
        frequency: Beep frequency in Hz
        duration: Duration of each beep in seconds
        repeat: Number of times to repeat
        gap: Gap between repeats in seconds
        volume: Volume (0.0 to 1.0)
    """
    sample_rate = 44100

    # Create beep pattern
    all_samples = []

    for _ in range(repeat):
        # Generate beep
        num_samples = int(sample_rate * duration)
        for i in range(num_samples):
            # Sine wave
            sample = volume * math.sin(2 * math.pi * frequency * i / sample_rate)
            # Envelope (fade in/out to avoid clicks)
            envelope = 1.0
            fade_samples = int(sample_rate * 0.01)  # 10ms fade
            if i < fade_samples:
                envelope = i / fade_samples
            elif i > num_samples - fade_samples:
                envelope = (num_samples - i) / fade_samples

            sample *= envelope
            all_samples.append(int(sample * 32767))

        # Add gap
        gap_samples = int(sample_rate * gap)
        all_samples.extend([0] * gap_samples)

    # Write WAV file
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(sample_rate)

        # Pack samples as bytes
        for sample in all_samples:
            wav_file.writeframes(struct.pack('<h', sample))


def main():
    """Generate all beep sounds"""

    print("=" * 60)
    print("Generating MetroEye Alert Beeps")
    print("=" * 60)

    # Create directories
    beep_dir = Path("beep_sounds")
    beep_dir.mkdir(exist_ok=True)

    frontend_sounds = Path("../frontend/public/sounds")
    frontend_sounds.mkdir(parents=True, exist_ok=True)

    # Define beep configurations
    beeps = [
        {
            "name": "driver_alert",
            "frontend_name": "driver_alert_beep.wav",
            "frequency": 1000,
            "duration": 0.5,
            "repeat": 3,
            "gap": 0.2,
            "description": "Driver Alert (Critical)"
        },
        {
            "name": "warning_beep",
            "frontend_name": "warning_beep.wav",
            "frequency": 800,
            "duration": 0.3,
            "repeat": 2,
            "gap": 0.15,
            "description": "Warning Beep"
        },
        {
            "name": "emergency_buzzer",
            "frontend_name": "emergency_buzzer.wav",
            "frequency": 1200,
            "duration": 0.3,
            "repeat": 5,
            "gap": 0.1,
            "description": "Emergency Buzzer"
        }
    ]

    # Generate beeps
    print("\n1. Generating beep sounds...")
    for beep in beeps:
        filename = str(beep_dir / f"{beep['name']}.wav")
        print(f"   Creating: {beep['description']}")
        create_beep_sound(
            filename=filename,
            frequency=beep['frequency'],
            duration=beep['duration'],
            repeat=beep['repeat'],
            gap=beep['gap']
        )

        # Copy to frontend
        import shutil
        frontend_file = frontend_sounds / beep['frontend_name']
        shutil.copy2(filename, frontend_file)

        # Get file size
        size_kb = Path(filename).stat().st_size / 1024
        print(f"   OK {beep['name']}.wav ({size_kb:.1f} KB)")

    print("\n2. Beep sounds ready!")
    print(f"\n   Vision-engine: {beep_dir.absolute()}")
    print(f"   Frontend: {frontend_sounds.absolute()}")
    print("\n" + "=" * 60)
    print("SUCCESS: All beep sounds generated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
