from __future__ import annotations

import re
from textwrap import dedent
from typing import TYPE_CHECKING, cast, override

if TYPE_CHECKING:
    import pandas as pd

RGB = tuple[float, float, float]


def rgb_to_hex(rgb: RGB) -> str:
    return ''.join(f'{int(c * 255):02x}' for c in rgb)


class ColorTableFormatter:
    escape: str

    def color_swatch(self, rgb: RGB) -> str:
        raise NotImplementedError

    def style(self, df: pd.DataFrame) -> pd.io.formats.style.Styler:
        styled = df.style.format(escape='html')
        styled = styled.hide(axis='index')
        if 'hsv' in df.columns:
            styled = styled.hide(['hsv'], axis='columns')
        styled = styled.format_index(lambda x: self.col_rename(cast(str, x)), axis=1)
        styled = styled.format(
            lambda rgb: self.color_swatch(cast(tuple, rgb)),
            subset=['rgb'],
        )
        return styled

    def to_str(self, df: pd.DataFrame) -> str:
        raise NotImplementedError

    def col_rename(self, col: str) -> str:
        return col


class ColorTableHtmlFormatter(ColorTableFormatter):
    def __init__(self, delta_suffix: str = '-delta'):
        super().__init__()
        self.delta_suffix = delta_suffix
        self.escape = 'html'

    @override
    def color_swatch(self, rgb):
        hex_color = rgb_to_hex(rgb)
        return f'<div style="background-color: #{hex_color}; width: 1em; height: 1em; display: inline-block; border: 1px solid #8888;"></div>'

    @override
    def style(self, df):
        from pandas.api.types import is_numeric_dtype

        styled = super().style(df)
        # For HTML tables, apply the numerical formatting here: no further processing happens.
        delta_cols = [col for col in df.columns if col.endswith(self.delta_suffix)]
        numeric_cols = [col for col in df.columns if is_numeric_dtype(df[col])]
        styled.format(lambda x: f'{x:.3f}', numeric_cols)
        styled.format(lambda x: f'{x:+.3f}', delta_cols)
        return styled

    @override
    def to_str(self, df):
        return self.style(df).to_html()

    @override
    def col_rename(self, col):
        if col in ('rgb', 'hsv'):
            return col.upper()
        if self.delta_suffix is not None and col.endswith(self.delta_suffix):
            col = f'Δ {col[:3]}'
        col = col.title()
        return col


class ColorTableLatexFormatter(ColorTableFormatter):
    def __init__(
        self,
        implicit_plus: re.Pattern = re.compile(r'\bdelta$', re.IGNORECASE),
        group_with_previous: re.Pattern = re.compile(r'\bdelta$|\brgb$', re.IGNORECASE),
        white_point=0.85,
        black_point=0.001,
        underline_header_groups=False,
    ):
        super().__init__()
        self.escape = 'latex'
        self.implicit_plus = implicit_plus
        self.group_with_previous = group_with_previous
        self.white_point = white_point
        self.black_point = black_point
        self.underline_header_groups = underline_header_groups

    @override
    def color_swatch(self, rgb):
        hex_color = rgb_to_hex(rgb).upper()
        return rf'\swatch{{{hex_color}}}'

    @override
    def style(self, df):
        from pandas.api.types import is_numeric_dtype

        styled = super().style(df)
        # For LaTeX tables, apply minimal formatting and defer to siunitx.
        numeric_cols = [col for col in df.columns if is_numeric_dtype(df[col])]
        styled.format(lambda x: f'{x: .9f}', numeric_cols)
        return styled

    @property
    def preamble(self) -> str:
        return dedent(r"""
            \usepackage{array}
            \usepackage{tikz}         % Graphics commands
            \usepackage{booktabs}     % Professional-looking tables
            \usepackage[x11names]{xcolor}
            \usepackage{siunitx}      % Numerical column formatting in tables
            \usepackage{pgfmath}      % For tabular heat maps
            \usepackage{collcell}     % Columnar cell formatting

            % Draws a color swatch (filled square)
            \newcommand{\swatch}[1]{%
                \tikz[baseline=0.2ex, scale=0.8, rounded corners=0.1em]{
                    \definecolor{swatchcolor}{HTML}{#1} \fill[color=swatchcolor] (0,0)
                    rectangle (1em,1em); \draw[line width=0.05pt, color=black,
                    opacity=0.5] (0,0) rectangle (1em,1em);
                }%
            }

            % Simple tabular heat map: make numbers grayscale according to magnitude
            \newcommand{\grayscaleparams}[6]{%
                % #1 = value, #2 = whitevalue, #3 = blackvalue, #4 = whitecolor, #5 = blackcolor, #6 = maxintensity
                \pgfmathsetmacro{\normalized}{max(0, min(1, (abs(#1) - #2) / (#3 - #2)))}%
                \pgfmathsetmacro{\intensity}{#6 * (1 - \normalized)}%
                \textcolor{#4!\intensity!#5}{\num{#1}}%
            }
            \newcommand{\grayscaleparamsplus}[6]{%
                \pgfmathsetmacro{\normalized}{max(0, min(1, (abs(#1) - #2) / (#3 - #2)))}%
                \pgfmathsetmacro{\intensity}{#6 * (1 - \normalized)}%
                \textcolor{#4!\intensity!#5}{\num[print-mantissa-implicit-plus=true]{#1}}%
            }

            % Convenience wrappers with defaults
            \newcommand{\grayscale}[1]{%
                \grayscaleparams{#1}{0.0001}{0.005}{white}{black}{70}%
            }
            \newcommand{\grayscaleplus}[1]{%
                \grayscaleparamsplus{#1}{0.0001}{0.005}{white}{black}{70}%
            }

            \newcolumntype{g}{>{\collectcell\grayscale}c<{\endcollectcell}}
            \newcolumntype{G}{>{\collectcell\grayscaleplus}c<{\endcollectcell}}
        """)

    @override
    def to_str(self, df: pd.DataFrame, *, caption: str | None = None, label: str | None = None):
        r"""
        Convert a dataframe to LaTeX output with color swatches.

        Depends on a custom `\swatch` command; see `preamble` property.
        """
        from pandas.api.types import is_numeric_dtype

        col_formats = []
        for col in df.columns:
            if is_numeric_dtype(df[col]):
                if self.implicit_plus.search(col):
                    col_formats.append('G')
                else:
                    col_formats.append('g')
            else:
                if col.lower() in ('rgb', 'hsv'):
                    col_formats.append('c')
                else:
                    col_formats.append('l')

        latex = (
            self.style(df)
            .to_latex(
                hrules=True,
                column_format=' '.join(col_formats),
                siunitx=True,
            )
            .strip()
        )
        # Replace default rules with booktabs
        latex = latex.replace('\\hline', '\\midrule')
        # Insert grouped header row and cmidrules
        latex = self._add_grouped_headers(latex, df)
        # Align to support column selection modes to make editing tables easier
        latex = self._align_ampersands(latex)

        head = dedent(rf"""
            \begin{{table}}
            \centering
            \label{{{label or 'tab:placeholder'}}}
            \caption{{{caption or 'Placeholder'}}}
            \sisetup{{
                round-mode = places,
                round-precision = 3,
                table-auto-round = true,
                % drop-zero-decimal = true,
            }}
        """).strip()

        foot = dedent(rf"""
            \end{{table}}
        """).strip()  # noqa: F541

        return f'{head}\n{latex}\n{foot}'

    @override
    def col_rename(self, col):
        return col.title()

    def _add_grouped_headers(self, latex: str, df: pd.DataFrame) -> str:
        """Add multicolumn grouped headers and cmidrules."""
        import re

        # Detect column groups
        groups = []
        col_idx = 0
        i = 0

        # Process remaining numeric columns
        while i < len(df.columns):
            col = df.columns[i]
            # Check if this is a base column with a delta
            if i + 1 < len(df.columns) and self.group_with_previous.search(df.columns[i + 1]):
                groups.append((col_idx, 2, self.col_rename(col)))
                col_idx += 2
                i += 2
            else:
                # Standalone column
                groups.append((col_idx, 1, self.col_rename(col)))
                col_idx += 1
                i += 1

        # Build the multicolumn header line
        header_parts = []
        cmidrules = []
        for start_idx, span, header_text in groups:
            header_parts.append(rf'\multicolumn{{{span}}}{{c}}{{{{{header_text}}}}}')
            cmidrules.append(rf'\cmidrule(lr){{{start_idx + 1}-{start_idx + span}}}')

        header_line = ' & '.join(header_parts) + r' \\'
        cmidrule_line = ' '.join(cmidrules)

        # Find the first header line (after \toprule) and insert our grouped header
        pattern = r'(\\toprule\n)(.*?)(\\\\)'

        def replacement(match):
            toprule = match.group(1)
            # original_header = match.group(2)
            # backslash = match.group(3)
            if self.underline_header_groups:
                return f'{toprule}{header_line}\n{cmidrule_line}'
            else:
                return f'{toprule}{header_line}'

        latex = re.sub(pattern, replacement, latex, count=1)

        return latex

    def _align_ampersands(self, latex: str) -> str:
        """Align ampersands in table rows for prettier LaTeX source."""
        lines = latex.split('\n')

        table_start = table_end = None
        for i, line in enumerate(lines):
            if r'\midrule' in line and table_start is None:
                table_start = i + 1
            if r'\bottomrule' in line:
                table_end = i
                break
        if table_start is None or table_end is None:
            return latex

        # Collect rows and split into cells
        rows = [
            (i, list(re.split(r'(?:&|\\\\\s*$)', lines[i]))[:-1])
            for i in range(table_start, table_end)
            if re.search(r'\\\\\s*$', lines[i])
        ]
        if not rows:
            return latex

        # Compute max width for each column (except last)
        columns = list(zip(*[cells for _, cells in rows], strict=False))  # transpose
        col_widths = [
            max((len(cell) for cell in col), default=0)  #
            for col in columns
        ]

        # Rebuild aligned rows
        for i, cells in rows:
            aligned = [
                cell.ljust(w)  #
                for w, cell in zip(col_widths, cells, strict=False)
            ]
            lines[i] = '&'.join(aligned) + r'\\'
        return '\n'.join(lines)
