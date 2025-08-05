from dataclasses import dataclass
from html import escape
from math import isfinite
from typing import Any, Collection, Mapping

from airium import Airium

from utils.progress.model import BarData, Mark


def render_progress_bar(data: BarData, metrics: Mapping[str, Any]):
    a = Airium()
    with a.div(style=css(width='100%', padding='5px 0', font_family='monospace')):
        format_bar(a, data)
        format_metrics(a, metrics)
    return str(a)


def format_bar_text_html(data: BarData):
    items_per_sec = data.count / data.elapsed_time if data.elapsed_time > 0 else 0
    eta_sec = (data.total - data.count) / items_per_sec if items_per_sec > 0 and data.count < data.total else 0
    elapsed_str = format_time(data.elapsed_time)
    eta_str = format_time(eta_sec) if data.count < data.total else format_time(0)

    text = ''
    if data.description:
        text += f'<b>{esc(data.description)}</b>: '
    text += f'{data.fraction:.1%} [{data.count:d}/{data.total:d}]'
    if data.suffix:
        text += f' | {esc(data.suffix)}'
    text += f' [<b>{esc(elapsed_str)}</b>/<{esc(eta_str)}, {items_per_sec:.2f} it/s]'

    return text


def format_bar(a: Airium, data: BarData):
    total = max(data.total, data.count)
    ticks = to_ticks(data.markers, total)
    hist = tick_hist(data.markers, total, bucket_size=50)

    with a.div(
        style=css(
            position='relative',
            height='calc(1em * 5/3)',
            width='100%',
            margin_bottom='2em' if any(m.is_major for m in ticks) else '1em',
        )
    ):
        # Heatmap of marker density
        if hist:
            counts = sorted(set(hist))
            max_val = max(counts) + 1
            densities = [count / max_val for count in counts]
            custom_props = {
                count: (f'--h{i:d}', f'color(from currentColor srgb r g b / {density**2:.2f})')
                for i, (count, density) in enumerate(zip(counts, densities, strict=True))
            }
            stops = []

            for i, count in enumerate(hist):
                custom_prop, _ = custom_props[count]
                position = i / (len(hist) - 1) * 100 if len(hist) > 1 else 0
                stops.append(f'var({custom_prop}) {position:.1f}%')

            css_vars = ';'.join(f'{custom_prop}:{value}' for custom_prop, value in custom_props.values())

            a.div(
                style=css(
                    position='absolute',
                    top='100%',
                    height='3.5px',
                    left='0',
                    width='100%',
                    background=f'linear-gradient(to right, {", ".join(stops)})',
                )
                + f';{css_vars}'
            )

        # Markers
        for tick in (t for t in ticks if t.is_major):
            if tick.fraction < 0.9514:
                hpos = dict(
                    left=f'{tick.fraction * 100:.1f}%',
                    border_left='0.5px solid currentColor',
                )
            else:
                hpos = dict(
                    right=f'{(1 - tick.fraction) * 100:.1f}%',
                    border_right='0.5px solid currentColor',
                )

            a.div(
                _t=esc(tick.label) if tick.label else '&nbsp;',
                style=css(
                    position='absolute',
                    top='100%',
                    font_size='70%',
                    padding='3px 2px 0',
                    **hpos,
                ),
            )

        # Triangle indicator
        with a.div(style=css(position='absolute', bottom='-4px', left=f'calc({data.fraction * 100:.1f}% - 4px)')):
            a.div(
                style=css(
                    width=0,
                    height=0,
                    border_left='4px solid transparent',
                    border_right='4px solid transparent',
                    border_bottom='4px solid currentColor',
                )
            )

        # Progress bar
        a.div(
            style=css(
                position='absolute',
                top=0,
                left=0,
                height='100%',
                width=f'{data.fraction * 100:.1f}%',
                background_color='color(from currentColor srgb r g b / 0.1)',
                border_bottom='1px solid currentColor',
            )
        )

        # Text overlay
        a.div(
            style=css(
                position='absolute',
                top=0,
                left=0,
                width='100%',
                height='100%',
                text_align='center',
                line_height='calc((1em * 5/3) / 0.9)',
                font_size='90%',
                white_space='nowrap',
                overflow='hidden',
                text_overflow='ellipsis',
                border_bottom='1px dashed color(from currentColor srgb r g b / 0.5)',
            ),
            _t=format_bar_text_html(data),
        )


def format_metrics(a: Airium, metrics: Mapping[str, Any]):
    with a.div(
        style=css(
            display='grid',
            grid_template_columns=f'repeat({len(metrics)}, minmax(80px, 1fr))',
            gap='5px 0px',
            width='100%',
            margin='1em 0',
            font_size='0.85em',
        )
    ):
        for key in metrics.keys():
            a.div(
                style=css(
                    font_weight='bold',
                    border_bottom='0.5px solid currentColor',
                    padding='2px 10px',
                    text_align='left',
                    overflow='hidden',
                    text_overflow='ellipsis',
                    white_space='nowrap',
                ),
                _t=esc(key),
            )

        for value in metrics.values():
            val_str = f'{value:.4g}' if isinstance(value, float) else str(value)
            a.div(
                style=css(
                    padding='2px 10px',
                    text_align='left',
                    overflow='hidden',
                    text_overflow='ellipsis',
                    white_space='nowrap',
                ),
                _t=esc(val_str),
            )


def format_time(seconds: float) -> str:
    if not isinstance(seconds, (int, float)) or not isfinite(seconds) or seconds < 0:
        return '??:??:??'
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h == 0:
        return f'{m:02d}:{s:02d}'
    else:
        return f'{h:d}:{m:02d}:{s:02d}'


def esc(value: Any) -> str:
    """Escape HTML entities like < and >"""
    # Don't need to scape quotes because Airium already does that for attribute values
    return escape(str(value), quote=False)


def css(**props):
    """Convert a mapping of properties into a CSS string"""
    return '; '.join(
        f'{k.replace("_", "-")}: {v}'  #
        for k, v in props.items()
        if isinstance(v, str) and v.strip()
    )


@dataclass(slots=True)
class Tick:
    fraction: float
    label: str
    is_major: bool


def to_ticks(markers: Collection[Mark], total: int, major_spacing: float = 0.095) -> list[Tick]:
    """
    Prepare markers for display.

    Marks will be converted to a fractional form. Marks that are sufficiently far
    apart will be flagged as "major". The last marker that has a label is always
    major, and any marks within major_spacing of it will not be major.

    Args:
        markers: The markers to prepare.
        total: The number of steps in the progress bar if it were at 100%.
        major_spacing: The minimum fractional space between major marks.
    """
    if not markers:
        return []

    # Convert to fractions first
    ticks = [
        Tick(fraction=mark.count / total if total > 0 else 1, label=mark.label.strip(), is_major=False)
        for mark in markers
    ]

    # Find the last marker with a label and mark it as major
    last_major_fraction = float('inf')
    for i in reversed(range(len(ticks))):
        if ticks[i].label:
            ticks[i] = Tick(ticks[i].fraction, ticks[i].label, True)
            last_major_fraction = ticks[i].fraction
            break

    # Process remaining markers from left to right
    prev_major = float('-inf')
    for i in range(len(ticks)):
        tick = ticks[i]
        # Skip if already marked as major (the last labeled one)
        if tick.is_major:
            prev_major = tick.fraction
            continue

        if (
            tick.label
            and tick.fraction - prev_major >= major_spacing
            and last_major_fraction - tick.fraction >= major_spacing
        ):
            prev_major = tick.fraction
            ticks[i] = Tick(tick.fraction, tick.label, True)

    return ticks


def tick_hist(markers: Collection[Mark], total: int, bucket_size: int) -> list[int]:
    """Create a histogram of markers."""
    if not markers or total == 0:
        return []

    hist = [0] * bucket_size
    for mark in markers:
        if mark.count > total:
            continue
        bucket = int(mark.count / total * bucket_size)
        if bucket >= bucket_size:
            bucket = bucket_size - 1
        hist[bucket] += 1
    return hist
