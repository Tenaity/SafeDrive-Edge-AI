import os
import time
import pygame


MAX_PLAY_SEC = 6.0


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    voice_path = os.path.join(base_dir, "voices", "startup.mp3")

    if not os.path.exists(voice_path):
        print(f"[STARTUP VOICE] Missing: {voice_path}")
        return

    try:
        if pygame.mixer.get_init() is None:
            pygame.mixer.init()

        sound = pygame.mixer.Sound(voice_path)
        length_sec = float(sound.get_length())

        play_sec = min(length_sec, MAX_PLAY_SEC)

        channel = pygame.mixer.Channel(7)
        channel.play(sound)

        print(f"[STARTUP VOICE] Playing: {voice_path}")
        print(f"[STARTUP VOICE] Length={length_sec:.2f}s, max_play={play_sec:.2f}s")

        start = time.time()

        while channel.get_busy():
            if time.time() - start >= play_sec:
                channel.stop()
                break
            time.sleep(0.05)

        print("[STARTUP VOICE] Done")

    except Exception as e:
        print(f"[STARTUP VOICE] Audio error: {e}")


if __name__ == "__main__":
    main()
