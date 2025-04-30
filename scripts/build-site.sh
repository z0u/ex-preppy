#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
# Treat unset variables as an error.
# Pipeline fails if any command fails.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to prepare directories
prepare_dirs() {
    rm -rf _site
    mkdir _site
}

# Function to copy content
copy_content() {
    cp README.md _site/index.md
    cp -r docs/* _site/
}

# Function to convert markdown files to temporary notebooks
convert_markdown_to_ipynb() {
    find _site -name '*.md' -print0 | while IFS= read -r -d $'\0' md_file; do
        ipynb_file="${md_file%.md}.ipynb"
        echo "  Converting $md_file to $ipynb_file..."

        # Read markdown content line by line, preserving newlines, and format as JSON array of strings
        json_source=$(python -c 'import json, sys; print(json.dumps([line for line in sys.stdin.readlines()]))' < "$md_file")

        # Define the notebook JSON structure as a format string
        notebook_json_format='{
"cells": [
{
"cell_type": "markdown",
"metadata": {},
"source": %s
}
],
"metadata": {
"kernelspec": {
"display_name": "Python 3",
"language": "python",
"name": "python3"
},
"language_info": {
"name": "python"
}
},
"nbformat": 4,
"nbformat_minor": 5
}'

        # Create the JSON structure using printf with the format string variable
        printf "$notebook_json_format" "$json_source" > "$ipynb_file"

        if [ ! -s "$ipynb_file" ]; then
            echo "Error: Failed to create $ipynb_file" >&2
            exit 1
        fi
        # Remove original markdown file after successful conversion
        rm "$md_file"
    done
}

# Function to convert ALL notebooks (original + generated from md)
convert_notebooks() {
    # Use uv run to execute jupyter within the project environment
    find _site -name '*.ipynb' -exec uv run -- jupyter nbconvert --config "$SCRIPT_DIR/build-site-config.py" --to html {} +
    # Delete the intermediate ipynb files (original and generated)
    find _site -name '*.ipynb' -delete
}

# Function to fix internal links (simplified)
fix_links() {
    find _site -name '*.html' -print0 | while IFS= read -r -d $'\0' file; do
        tmp_file=$(mktemp)
        # Remove docs/ prefix, then replace .ipynb extensions
        sed -e 's%href="docs/%href="%g' \
            -e 's%href="\([^"#?]*\)\.ipynb\([^"]*\)"%href="\1.html\2"%g' \
            "$file" > "$tmp_file" && mv "$tmp_file" "$file"
        rm -f "$tmp_file"
    done
}

# Function to add .nojekyll file
add_nojekyll() {
    touch _site/.nojekyll
}

# Function to inject custom CSS into HTML files
inject_css() {
    echo "Injecting custom CSS..."
    # Define the CSS rules using cat and a heredoc
    css_rules=$(cat "$SCRIPT_DIR/build-site-nb.css")
    css_rules="<style>$css_rules</style>"

    # Escape backslashes and ampersands for awk's -v variable assignment
    # Preserve newlines within the variable
    escaped_css_for_awk=$(printf '%s' "$css_rules" | sed -e 's/\\/\\\\/g' -e 's/&/\\&/g')

    # Create a temporary file safely
    temp_file=$(mktemp) || { echo "Error: Failed to create temporary file" >&2; exit 1; }
    # Ensure temp file is cleaned up on exit, error, or interrupt
    trap 'rm -f "$temp_file"' EXIT HUP INT QUIT TERM

    # Find all HTML files in _site and inject the CSS before </head>
    # Add error handling to the loop and awk/mv commands
    find _site -name '*.html' -print0 | while IFS= read -r -d $'\0' html_file; do
        # Check if CSS is already injected to avoid duplicates
        # Use -F for fixed string search (safer and potentially faster)
        if ! grep -qF '/* Custom styles for prose width */' "$html_file"; then
            echo "  Injecting CSS into $html_file..."
            # Use awk to insert the CSS block before the closing </head> tag
            # If awk fails or mv fails, the '&&' will short-circuit and the '||' part will execute
            awk -v css="$escaped_css_for_awk" '
            /<\/head>/ { print css }
            { print }
            ' "$html_file" > "$temp_file" && mv "$temp_file" "$html_file" || {
                echo "Error: Failed to process or move file for $html_file" >&2
                # temp_file will be removed by trap
                exit 1 # Exit script due to error
            }
            # Recreate temp file for the next iteration if mv succeeded
            # (Alternatively, create temp file inside the loop, but mktemp in a loop can be slow)
            # Let's stick with one temp file and ensure it's empty/truncated before next use by awk's > redirection.
        else
            echo "    Skipping $html_file, custom CSS already present."
        fi
    done || {
        # This catches errors from 'find' or if the 'while' loop body exits with non-zero status (like our exit 1 above)
        echo "Error: find command or processing loop failed." >&2
        # temp_file will be removed by trap
        exit 1 # Exit script
    }

    # Trap will clean up the temp file on normal exit too
    echo "CSS injection complete."
}

# Main script execution
echo "Preparing directories..."
prepare_dirs

echo "Copying content..."
copy_content

echo "Converting Markdown files to temporary Notebooks..."
convert_markdown_to_ipynb

echo "Converting all Notebooks to HTML..."
convert_notebooks

echo "Fixing internal links in HTML files..."
fix_links

echo "Adding .nojekyll file..."
add_nojekyll

echo "Injecting custom CSS into HTML files..."
inject_css

echo "Site build complete."
