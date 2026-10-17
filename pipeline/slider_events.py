"""Faithful Python port of osu.Game SliderEventGenerator.

Source: osu.Game/Rulesets/Objects/SliderEventGenerator.cs
"""
from __future__ import annotations

from dataclasses import dataclass

TAIL_LENIENCY = -36.0
MAX_LENGTH = 100000.0

TICK = "tick"
LEGACY_LAST_TICK = "legacy_last_tick"
HEAD = "head"
TAIL = "tail"
REPEAT = "repeat"


@dataclass
class SliderEventDescriptor:
    type: str
    time: float
    span_index: int
    span_start_time: float
    path_progress: float


def _generate_ticks(span_index, span_start_time, span_duration, reversed_, length, tick_distance, min_distance_from_end):
    d = tick_distance
    while d <= length:
        if d >= length - min_distance_from_end:
            break
        path_progress = d / length
        time_progress = (1 - path_progress) if reversed_ else path_progress
        yield SliderEventDescriptor(
            type=TICK,
            span_index=span_index,
            span_start_time=span_start_time,
            time=span_start_time + time_progress * span_duration,
            path_progress=path_progress,
        )
        d += tick_distance


def generate(start_time, span_duration, velocity, tick_distance, total_distance, span_count) -> list[SliderEventDescriptor]:
    length = min(MAX_LENGTH, total_distance)
    tick_distance = min(max(tick_distance, 0.0), length)
    min_distance_from_end = velocity * 10

    events: list[SliderEventDescriptor] = []
    events.append(SliderEventDescriptor(HEAD, start_time, 0, start_time, 0.0))

    for span in range(span_count):
        span_start_time = start_time + span * span_duration
        reversed_ = span % 2 == 1

        if tick_distance != 0:
            ticks = list(_generate_ticks(span, span_start_time, span_duration, reversed_,
                                         length, tick_distance, min_distance_from_end))
            if reversed_:
                ticks.reverse()
            events.extend(ticks)

        if span < span_count - 1:
            events.append(SliderEventDescriptor(
                REPEAT,
                span_start_time + span_duration,
                span,
                start_time + span * span_duration,
                (span + 1) % 2,
            ))

    total_duration = span_count * span_duration
    final_span_index = span_count - 1
    final_span_start_time = start_time + final_span_index * span_duration

    legacy_last_tick_time = max(start_time + total_duration / 2,
                                (final_span_start_time + span_duration) + TAIL_LENIENCY)
    legacy_last_tick_progress = (legacy_last_tick_time - final_span_start_time) / span_duration
    if span_count % 2 == 0:
        legacy_last_tick_progress = 1 - legacy_last_tick_progress

    events.append(SliderEventDescriptor(LEGACY_LAST_TICK, legacy_last_tick_time,
                                        final_span_index, final_span_start_time, legacy_last_tick_progress))
    events.append(SliderEventDescriptor(TAIL, start_time + total_duration,
                                        final_span_index, start_time + (span_count - 1) * span_duration,
                                        span_count % 2))

    return events
