from dataclasses import dataclass
from math import isfinite
from typing import Any, Collection, Mapping
from html import escape

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
    markers = prep_markers(data.markers, max(data.total, data.count))

    with a.div(
        style=css(
            position='relative',
            height='calc(1em * 5/3)',
            width='100%',
            margin_bottom='2em' if any(m.is_major for m in markers) else '1em',
        )
    ):
        # Markers
        for mark in markers:
            if mark.fraction < 0.9514:
                hpos = dict(
                    left=f'{mark.fraction * 100:.1f}%',
                    border_left='0.5px solid currentColor',
                )
            else:
                hpos = dict(
                    right=f'{(1 - mark.fraction) * 100:.1f}%',
                    border_right='0.5px solid currentColor',
                )

            if mark.is_major:
                vpos = dict(
                    top='100%',
                    font_size='70%',
                    padding='3px 1px 0',
                )
            else:
                vpos = dict(
                    top='100%',
                    height='1.5px',
                    font_size='0',
                    padding='1px 1px 0',
                )

            a.div(
                _t=esc(mark.label) if mark.label else '&nbsp;',
                style=css(
                    position='absolute',
                    **hpos,
                    **vpos,
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
                    border_bottom='1px solid currentColor',
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


def prep_markers(markers: Collection[Mark], total: int, major_spacing: float = 0.045):
    ms: list[Tick] = []
    last_major = float('-inf')
    for mark in markers:
        fraction = mark.count / total if total > 0 else 1
        if mark.label.strip() and fraction - last_major >= major_spacing:
            last_major = fraction
            is_major = True
        else:
            is_major = False
        ms.append(Tick(fraction, mark.label.strip(), is_major))

    return ms
