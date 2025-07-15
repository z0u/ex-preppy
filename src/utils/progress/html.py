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
    with a.div(style='width: 100%; padding: 5px 0; font-family: monospace;'):
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
    with a.div(style='position: relative; height: calc(1em * 5/3); width: 100%;'):
        # Triangle indicator
        with a.div(style=f'position: absolute; bottom: -4px; left: calc({data.fraction * 100:.1f}% - 4px);'):
            a.div(
                style="""
                width: 0;
                height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 4px solid currentColor;
                """
            )

        # Progress bar
        a.div(
            style=f"""
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            width: {data.fraction * 100:.1f}%;
            background-color: color(from currentColor srgb r g b / 0.1);
            border-bottom: 1px solid currentColor;
            """
        )

        # Text overlay
        with a.div(
            style="""
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            text-align: center;
            line-height: calc(1em * 5/3);
            font-size: 0.9em;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            border-bottom: 1px dashed color(from currentColor srgb r g b / 0.5);
            """
        ):
            a(format_bar_text(data))


def format_metrics(a: Airium, metrics: Mapping[str, Any]):
    with a.div(
        style=f"""
        display: grid;
        grid-template-columns: repeat({len(metrics)}, minmax(80px, 1fr));
        gap: 5px 0px;
        width: 100%;
        margin-top: 10px;
        font-size: 0.85em;
        """
    ):
        for key in metrics.keys():
            a.div(
                style="""
                font-weight: bold;
                border-bottom: 1px solid currentColor;
                padding-block: 2px;
                padding-inline: 10px;
                text-align: left;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                """,
                _t=key,
            )

        for value in metrics.values():
            val_str = f'{value:.4g}' if isinstance(value, float) else str(value)
            a.div(
                style="""
                padding-block: 2px;
                padding-inline: 10px;
                text-align: left;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                """,
                _t=val_str,
            )
