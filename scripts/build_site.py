#!/usr/bin/env python
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# --- Configuration ---
# Use pathlib for robust path handling
SCRIPT_DIR = Path(__file__).parent.resolve()
WORKSPACE_ROOT = SCRIPT_DIR.parent
SITE_DIR = WORKSPACE_ROOT / '_site'
DOCS_DIR = WORKSPACE_ROOT / 'docs'
CSS_FILE = SCRIPT_DIR / 'build-site-nb.css'
NBCONVERT_CONFIG = SCRIPT_DIR / 'build-site-config.py'
README_FILE = WORKSPACE_ROOT / 'README.md'
CSS_MARKER = '/* Custom styles for prose width */'  # Used to check if CSS is already injected

# --- Helper Functions ---


def run_command(cmd_list, check=True, cwd=None, capture_output=False):
    """Run a command using subprocess and handles errors."""
    print(f'Running command: {" ".join(map(str, cmd_list))}')
    try:
        result = subprocess.run(
            cmd_list,
            check=check,
            cwd=cwd,
            text=True,
            capture_output=capture_output,
            # Pass environment variables to ensure uv finds its context
            env=os.environ,
        )
        if capture_output:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f'Error running command: {e}', file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f'Error: Command not found: {cmd_list[0]}', file=sys.stderr)
        sys.exit(1)


# --- Build Steps ---


def prepare_dirs():
    """Remove the existing site directory and creates a new empty one."""
    print('Preparing directories...')
    if SITE_DIR.exists():
        print(f'  Removing existing directory: {SITE_DIR}')
        shutil.rmtree(SITE_DIR)
    print(f'  Creating directory: {SITE_DIR}')
    SITE_DIR.mkdir()


def copy_content():
    """Copy content from docs and the root README into the site directory."""
    print('Copying content...')
    # Copy README.md to _site/index.md
    target_readme = SITE_DIR / 'index.md'
    print(f'  Copying {README_FILE} to {target_readme}')
    shutil.copy(README_FILE, target_readme)

    # Copy everything from docs/* to _site/
    # Use copytree for directories, handle files separately if needed,
    # or iterate and copy individually if more control is required.
    # shutil.copytree needs the destination *not* to exist, or dirs_exist_ok=True (Python 3.8+)
    # Since SITE_DIR is fresh, we can copy contents item by item or use copytree carefully.
    # Let's iterate to mimic the `cp -r docs/* _site/` behavior more closely.
    print(f'  Copying contents of {DOCS_DIR} to {SITE_DIR}')
    if DOCS_DIR.is_dir():
        for item in DOCS_DIR.iterdir():
            target_item = SITE_DIR / item.name
            if item.is_dir():
                shutil.copytree(item, target_item, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target_item)  # copy2 preserves metadata like cp -p
    else:
        print(f'Warning: Docs directory {DOCS_DIR} not found.', file=sys.stderr)


def convert_markdown_to_ipynb():
    """Find Markdown files in _site, converts them to Jupyter notebooks."""
    print('Converting Markdown files to temporary Notebooks...')
    md_files = list(SITE_DIR.rglob('*.md'))  # Use rglob for recursive search
    if not md_files:
        print('  No markdown files found to convert.')
        return

    for md_file in md_files:
        ipynb_file = md_file.with_suffix('.ipynb')
        print(f'  Converting {md_file.relative_to(WORKSPACE_ROOT)} to {ipynb_file.relative_to(WORKSPACE_ROOT)}...')

        try:
            # Read markdown content, handling potential encoding issues
            md_content = md_file.read_text(encoding='utf-8').splitlines(keepends=True)

            # Construct the notebook JSON structure
            notebook_json = {
                'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': md_content}],
                'metadata': {
                    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                    'language_info': {'name': 'python'},
                },
                'nbformat': 4,
                'nbformat_minor': 5,
            }

            # Write the JSON to the .ipynb file
            with open(ipynb_file, 'w', encoding='utf-8') as f:
                json.dump(notebook_json, f, indent=1)  # Use indent for readability

            # Check if the file was created and is not empty
            if not ipynb_file.exists() or ipynb_file.stat().st_size == 0:
                raise IOError(f'Failed to create or write to {ipynb_file}')

            # Remove original markdown file after successful conversion
            print(f'  Removing original markdown file: {md_file.relative_to(WORKSPACE_ROOT)}')
            md_file.unlink()

        except Exception as e:
            print(f'Error converting {md_file}: {e}', file=sys.stderr)
            sys.exit(1)


def convert_notebooks():
    """Convert all .ipynb files in _site to HTML using nbconvert."""
    print('Converting all Notebooks to HTML...')
    notebook_files = list(SITE_DIR.rglob('*.ipynb'))  # Use rglob for recursive search
    if not notebook_files:
        print('  No notebooks found to convert.')
        return

    # We need to run nbconvert via uv to ensure it uses the correct environment
    # Build the command list
    cmd = ['uv', 'run', '--', 'jupyter', 'nbconvert', '--config', str(NBCONVERT_CONFIG), '--to', 'html']
    cmd.extend(map(str, notebook_files))  # Add all notebook paths to the command

    run_command(cmd, cwd=WORKSPACE_ROOT)  # Run from workspace root

    # Delete the intermediate ipynb files
    print('  Deleting intermediate notebook files...')
    for ipynb_file in notebook_files:
        try:
            print(f'    Deleting {ipynb_file.relative_to(WORKSPACE_ROOT)}')
            ipynb_file.unlink()
        except OSError as e:
            print(f'Error deleting file {ipynb_file}: {e}', file=sys.stderr)
            # Decide if this is fatal or just a warning
            # sys.exit(1)


def fix_links():
    """Adjust links in generated HTML files."""
    print('Fixing internal links in HTML files...')
    html_files = list(SITE_DIR.rglob('*.html'))
    if not html_files:
        print('  No HTML files found to fix links in.')
        return

    # Regex to find hrefs like "docs/something.ipynb" or just "something.ipynb"
    # and convert them to "something.html"
    # 1. Remove 'docs/' prefix if present: href="docs/([^"]*)" -> href="\1"
    # 2. Change '.ipynb' extension to '.html': href="([^"#?]*)\.ipynb([^"]*)" -> href="\1.html\2"
    # Combine into one pass if possible, or two sequential replacements.
    # Let's do two sequential replacements for clarity, mimicking the sed approach.

    # Pattern 1: Remove 'docs/' prefix from hrefs
    docs_prefix_pattern = re.compile(r'href="docs/([^"]*)"')
    # Pattern 2: Replace .ipynb with .html in hrefs, handling anchors and query params
    ipynb_ext_pattern = re.compile(r'href="([^"#?]+)\.ipynb(#|\?|")')

    for html_file in html_files:
        print(f'  Processing links in {html_file.relative_to(WORKSPACE_ROOT)}...')
        try:
            content = html_file.read_text(encoding='utf-8')
            original_content = content  # Keep a copy for comparison

            # Step 1: Remove 'docs/' prefix
            content = docs_prefix_pattern.sub(r'href="\1"', content)

            # Step 2: Replace .ipynb with .html
            # We need a function for replacement to correctly handle the trailing character (#, ?, or ")
            def replace_ipynb(match):
                base_link = match.group(1)
                trailing_char = match.group(2)
                return f'href="{base_link}.html{trailing_char}'

            content = ipynb_ext_pattern.sub(replace_ipynb, content)

            # Only write back if changes were made
            if content != original_content:
                print(f'    Links updated in {html_file.relative_to(WORKSPACE_ROOT)}')
                html_file.write_text(content, encoding='utf-8')
            else:
                print(f'    No link changes needed for {html_file.relative_to(WORKSPACE_ROOT)}')

        except Exception as e:
            print(f'Error processing links in {html_file}: {e}', file=sys.stderr)
            # Decide whether to continue or exit
            # sys.exit(1)


def add_nojekyll():
    """Create a .nojekyll file in the site directory."""
    print('Adding .nojekyll file...')
    nojekyll_file = SITE_DIR / '.nojekyll'
    try:
        nojekyll_file.touch()
        print(f'  Created {nojekyll_file}')
    except OSError as e:
        print(f'Error creating {nojekyll_file}: {e}', file=sys.stderr)
        sys.exit(1)


def inject_css():
    """Inject custom CSS into the head of HTML files."""
    print('Injecting custom CSS...')
    if not CSS_FILE.is_file():
        print(f'Error: CSS file not found at {CSS_FILE}', file=sys.stderr)
        sys.exit(1)

    try:
        css_rules = CSS_FILE.read_text(encoding='utf-8')
        style_block = f'<style>\n{css_rules}\n</style>'
    except Exception as e:
        print(f'Error reading CSS file {CSS_FILE}: {e}', file=sys.stderr)
        sys.exit(1)

    html_files = list(SITE_DIR.rglob('*.html'))
    if not html_files:
        print('  No HTML files found to inject CSS into.')
        return

    # Pattern to find the closing </head> tag (case-insensitive)
    head_end_pattern = re.compile(r'</head>', re.IGNORECASE)

    for html_file in html_files:
        print(f'  Processing CSS for {html_file.relative_to(WORKSPACE_ROOT)}...')
        try:
            content = html_file.read_text(encoding='utf-8')

            # Check if CSS marker is already present
            if CSS_MARKER in content:
                print(f'    Skipping {html_file.relative_to(WORKSPACE_ROOT)}, custom CSS already present.')
                continue

            # Find the position to insert the CSS
            match = head_end_pattern.search(content)
            if match:
                insert_pos = match.start()
                new_content = content[:insert_pos] + '\n' + style_block + '\n' + content[insert_pos:]
                print(f'    Injecting CSS into {html_file.relative_to(WORKSPACE_ROOT)}')
                html_file.write_text(new_content, encoding='utf-8')
            else:
                print(
                    f'    Warning: Could not find </head> tag in {html_file.relative_to(WORKSPACE_ROOT)}. CSS not injected.',
                    file=sys.stderr,
                )

        except Exception as e:
            print(f'Error injecting CSS into {html_file}: {e}', file=sys.stderr)
            # Decide whether to continue or exit
            # sys.exit(1)

    print('CSS injection complete.')


# --- Main Execution ---


def main():
    prepare_dirs()
    copy_content()
    convert_markdown_to_ipynb()
    convert_notebooks()
    fix_links()
    add_nojekyll()
    inject_css()
    print(f'\nSite build complete. Wrote to {SITE_DIR.relative_to(os.getcwd())}/')


if __name__ == '__main__':
    main()
