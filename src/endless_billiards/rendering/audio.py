"""Procedural, zero-dependency audio synthesizer creating retro-arcade cue sounds."""

from __future__ import annotations

import math
import struct
import pygame


class SoundManager:
    """Synthesizes procedural sound effects using raw PCM wave buffers."""

    __slots__ = ("_enabled", "_cue_hit", "_rail_bounce", "_pocket_sink", "_ui_blip")

    def __init__(self, master_volume: float = 0.8) -> None:
        self._enabled: bool = False
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self._enabled = True
        except pygame.error:
            self._enabled = False
            return

        self._cue_hit = self._synth_noise_burst(duration=0.08, decay=35.0, volume=0.9 * master_volume)
        self._rail_bounce = self._synth_sine_thud(freq=95.0, duration=0.1, decay=20.0, volume=0.7 * master_volume)
        self._pocket_sink = self._synth_frequency_slide(start_f=380.0, end_f=120.0, duration=0.18, volume=0.85 * master_volume)
        self._ui_blip = self._synth_sine_thud(freq=580.0, duration=0.05, decay=40.0, volume=0.3 * master_volume)

    def _synth_sine_thud(self, freq: float, duration: float, decay: float, volume: float) -> pygame.mixer.Sound:
        sample_rate = 44100
        num_samples = int(sample_rate * duration)
        raw = bytearray()

        for i in range(num_samples):
            t = i / sample_rate
            amp = math.exp(-decay * t) * volume
            sample = int(32767.0 * amp * math.sin(2.0 * math.pi * freq * t))
            sample = max(-32768, min(32767, sample))
            packed = struct.pack("<hh", sample, sample)
            raw.extend(packed)

        return pygame.mixer.Sound(buffer=bytes(raw))

    def _synth_noise_burst(self, duration: float, decay: float, volume: float) -> pygame.mixer.Sound:
        import random

        sample_rate = 44100
        num_samples = int(sample_rate * duration)
        raw = bytearray()

        for i in range(num_samples):
            t = i / sample_rate
            amp = math.exp(-decay * t) * volume
            noise = random.uniform(-1.0, 1.0)
            sample = int(32767.0 * amp * noise)
            sample = max(-32768, min(32767, sample))
            raw.extend(struct.pack("<hh", sample, sample))

        return pygame.mixer.Sound(buffer=bytes(raw))

    def _synth_frequency_slide(self, start_f: float, end_f: float, duration: float, volume: float) -> pygame.mixer.Sound:
        sample_rate = 44100
        num_samples = int(sample_rate * duration)
        raw = bytearray()
        phase = 0.0

        for i in range(num_samples):
            t = i / sample_rate
            factor = i / max(1, num_samples)
            freq = start_f + (end_f - start_f) * factor
            phase += 2.0 * math.pi * freq * (1.0 / sample_rate)

            amp = (1.0 - factor) * volume
            sample = int(32767.0 * amp * math.sin(phase))
            sample = max(-32768, min(32767, sample))
            raw.extend(struct.pack("<hh", sample, sample))

        return pygame.mixer.Sound(buffer=bytes(raw))

    def play_cue_hit(self, power_factor: float = 1.0) -> None:
        if self._enabled:
            self._cue_hit.set_volume(max(0.1, min(1.0, power_factor)))
            self._cue_hit.play()

    def play_rail_bounce(self) -> None:
        if self._enabled:
            self._rail_bounce.play()

    def play_pocket_sink(self) -> None:
        if self._enabled:
            self._pocket_sink.play()

    def play_ui_blip(self) -> None:
        if self._enabled:
            self._ui_blip.play()