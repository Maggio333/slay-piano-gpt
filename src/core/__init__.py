"""Współdzielone moduły: architektura (gpt) + render ABC->MIDI (abc_to_midi).

Alias zgodności: checkpointy zapisane przed reorganizacją src/ (gdy gpt.py leżał
w korzeniu src/) trzymają obiekt GPTConfig z odwołaniem do modułu `gpt`. Rejestrujemy
`gpt` jako alias `core.gpt`, by te wagi wczytywały się bez przepisywania.
"""
import sys
from . import gpt

sys.modules.setdefault("gpt", gpt)
