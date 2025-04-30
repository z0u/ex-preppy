#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
# Treat unset variables as an error.
# Pipeline fails if any command fails.
set -euo pipefail

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
    find _site -name '*.ipynb' -exec uv run -- jupyter nbconvert --to html {} +
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

echo "Site build complete."
