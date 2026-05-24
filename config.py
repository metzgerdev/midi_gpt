"""Constants shared by the retained track-generation inference path."""

# Drum-loop analysis. SAMPLE_RATE and FRAME_RATE are inherited from EnCodec-24k,
# which framed audio at 75 fps; the codec is long gone but the numbers stayed.
SAMPLE_RATE = 24_000
FRAME_RATE = 75
KICK_BAND_HZ = 160.0

# GRID_REF_BPM is NOT the output tempo. It is the tempo the grid arithmetic is
# defined against: GRID_FRAMES below is "two bars at this tempo", the window
# `onset_grid` reads, and the target `align_to_grid_tempo` stretches loops toward.
# Output tempo comes from --bpm or a genre's "bpm" and is applied only by
# write_midi. Changing this re-interprets every drum loop and silently corrupts
# the kick grid; it does not make the music faster.
GRID_REF_BPM = 130.0

GRID_BARS = 2
GRID_STEPS_PER_BAR = 16
GRID_STEPS = GRID_BARS * GRID_STEPS_PER_BAR
GRID_FRAMES = round(GRID_BARS * 4 * (60.0 / GRID_REF_BPM) * FRAME_RATE)

# Symbolic bass/arp representation: one token per sixteenth-note step.
NOTE_BARS = 4
NOTE_STEPS = NOTE_BARS * GRID_STEPS_PER_BAR
NOTE_MIDI_LO = 24
NOTE_MIDI_HI = 84
NOTE_REST = 0
NOTE_SUSTAIN = 1
NOTE_PITCH0 = 2
NOTE_VOCAB = NOTE_PITCH0 + (NOTE_MIDI_HI - NOTE_MIDI_LO + 1)  # 63
NOTE_BOS = NOTE_VOCAB
NOTE_EOS = NOTE_VOCAB + 1
NOTE_VOCAB_SIZE = NOTE_VOCAB + 2  # 65

# Checkpoint-compatible conditioning widths.
CLAP_DIM = 512
CHORD_DIM = 12
