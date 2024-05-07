import os
import sys
import note_seq
import numpy as np
from note_seq.protobuf import music_pb2
import my_melody_utils
def extract_melodies(note_sequence, keep_longest_split=False, mode='bcd',part=1):
  """Extracts all melodies in a polyphonic note sequence.
  
  Args:
    note_sequence: A polyphonic NoteSequence object.
    keep_longest_split: Whether to discard all subsequences with tempo changes
        other than the longest one.
    
  Returns:
    List of monophonic NoteSequence objects.
  """
  splits = note_seq.sequences_lib.split_note_sequence_on_time_changes(
      note_sequence)

  if keep_longest_split:
    ns = max(splits, key=lambda x: len(x.notes))
    splits = [ns]

  melodies = []
  for split_ns in splits:
    qs = note_seq.sequences_lib.quantize_note_sequence(split_ns,
                                                       steps_per_quarter=4)
    if mode == 'bcd':
      try:
          melody.from_quantized_sequence(qs,
                                        ignore_polyphonic_notes=True,
                                        mode='part',
                                        instrument=0,
                                        part=part,
                                        gap_bars=np.inf)
      except note_seq.NonIntegerStepsPerBarError
      melody_ns = melody.to_sequence()
      melodies.append(melody_ns)
    else:
      #TODO
      instruments = list(set([note.instrument for note in qs.notes]))

      for instrument in instruments:
        melody = myns_utils.Melody()
        try:
          melody.from_quantized_sequence(qs,
                                        ignore_polyphonic_notes=True,
                                        mode='inst'
                                        instrument=instrument,
                                        part=0,
                                        gap_bars=np.inf)
        except note_seq.NonIntegerStepsPerBarError:
          continue
        melody_ns = melody.to_sequence()
        melodies.append(melody_ns)

  return melodies

def select_particular_part(ns, dataset_name, part_id):
  pns = music_pb2.NoteSequence()
  if dataset_name == 'BCD':
    for oldnote in ns.notes:
      if oldnote.part == part_id:

        note = pns.notes.add()
        note.part = part_id
        note.voice = oldnote.voice
        note.instrument = oldnote.instrument
        note.program = oldnote.program
        note.start_time = oldnote.start_time

        # Fix negative time errors from incorrect MusicXML
        if note.start_time < 0:
          note.start_time = 0

        note.end_time = oldnote.end_time
        note.pitch = oldnote.pitch
        note.velocity = oldnote.velocity

        note.numerator = oldnote.numerator
        note.denominator = oldnote.denominator
  elif dataset_name == 'lakh':
    return
  else:
    print("dataset name error!\n")
    return
  return pns