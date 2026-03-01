"""
Setup Audio Files for MetroEye
===============================

Generates beep sounds and copies them to frontend public folder
"""

import os
import shutil
from pathlib import Path
from simple_pa_system import SimplePASystem

def setup_audio_files():
    """Generate beep sounds and copy to frontend"""

    print("=" * 60)
    print("Setting up MetroEye Audio System")
    print("=" * 60)

    # Initialize PA system (this generates beep sounds)
    print("\n1️⃣ Generating beep sounds...")
    pa = SimplePASystem()

    # Paths
    beep_dir = Path("beep_sounds")
    frontend_sounds = Path("../frontend/public/sounds")

    # Create frontend sounds directory if needed
    frontend_sounds.mkdir(parents=True, exist_ok=True)

    # Copy driver alert beep to frontend
    print("\n2️⃣ Copying sounds to frontend...")

    beep_files = [
        ("driver_alert.wav", "driver_alert_beep.wav"),  # (source, dest)
        ("warning_beep.wav", "warning_beep.wav"),
        ("emergency_buzzer.wav", "emergency_buzzer.wav")
    ]

    for source_name, dest_name in beep_files:
        source = beep_dir / source_name
        dest = frontend_sounds / dest_name

        if source.exists():
            shutil.copy2(source, dest)
            size_kb = dest.stat().st_size / 1024
            print(f"  ✓ {dest_name} ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ {source_name} not found")

    print(f"\n✓ Audio setup complete!")
    print(f"\nFrontend sounds location: {frontend_sounds.absolute()}")
    print(f"Vision-engine sounds: {beep_dir.absolute()}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    setup_audio_files()
