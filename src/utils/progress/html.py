from math import isfinite
from typing import Any, Mapping

from airium import Airium

from utils.progress.model import BarData


def format_time(seconds: float) -> str:
    if not isinstance(seconds, (int, float)) or not isfinite(seconds) or seconds < 0:
        return '??:??:??'
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h == 0:
        return f'{m:02d}:{s:02d}'
    else:
        return f'{h:d}:{m:02d}:{s:02d}'


def render_progress_bar(data: BarData, metrics: Mapping[str, Any]):
    a = Airium()
    with a.div(style=css(width='100%', padding='5px 0', font_family='monospace')):
        format_bar(a, data)
        format_metrics(a, metrics)
    return str(a)


def format_bar_text(data: BarData):
    items_per_sec = data.count / data.elapsed_time if data.elapsed_time > 0 else 0
    eta_sec = (data.total - data.count) / items_per_sec if items_per_sec > 0 and data.count < data.total else 0
    elapsed_str = format_time(data.elapsed_time)
    eta_str = format_time(eta_sec) if data.count < data.total else format_time(0)

    text = ''
    if data.description:
        text += f'{data.description}: '
    text += f'{data.fraction * 100:.1f}% [{data.count}/{data.total}]'
    if data.suffix:
        text += f' | {data.suffix}'
    text += f' [{elapsed_str}/<{eta_str}, {items_per_sec:.2f} it/s]'

    return text


def format_bar(a: Airium, data: BarData):
    with a.div(style=css(position='relative', height='calc(1em * 5/3)', width='100%')):
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
                line_height='calc(1em * 5/3)',
                font_size='0.9em',
                white_space='nowrap',
                overflow='hidden',
                text_overflow='ellipsis',
                border_bottom='1px dashed color(from currentColor srgb r g b / 0.5)',
            ),
            _t=format_bar_text(data),
        )


def format_metrics(a: Airium, metrics: Mapping[str, Any]):
    with a.div(
        style=css(
            display='grid',
            grid_template_columns=f'repeat({len(metrics)}, minmax(80px, 1fr))',
            gap='5px 0px',
            width='100%',
            margin_top='10px',
            font_size='0.85em',
        )
    ):
        for key in metrics.keys():
            a.div(
                style=css(
                    font_weight='bold',
                    border_bottom='1px solid currentColor',
                    padding_block='2px',
                    padding_inline='10px',
                    text_align='left',
                    overflow='hidden',
                    text_overflow='ellipsis',
                    white_space='nowrap',
                ),
                _t=key,
            )

        for value in metrics.values():
            val_str = f'{value:.4g}' if isinstance(value, float) else str(value)
            a.div(
                style=css(
                    padding_block='2px',
                    padding_inline='10px',
                    text_align='left',
                    overflow='hidden',
                    text_overflow='ellipsis',
                    white_space='nowrap',
                ),
                _t=val_str,
            )


def css(**props):
    return '; '.join(f'{k.replace("_", "-")}: {v}' for k, v in props.items())
