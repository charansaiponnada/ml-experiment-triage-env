# ML Experiment Triage Environment - Technical Documentation

## Overview

This directory contains the LaTeX source files for the technical documentation of the ML Experiment Triage Environment project.

## Files

- `main.tex` - Main LaTeX document containing all sections
- `architecture.pdf` - System architecture diagram (generated from TikZ)
- `bibliography.bib` - BibTeX bibliography file

## Building the PDF

### Prerequisites

Ensure you have a LaTeX distribution installed:

**macOS:**
```bash
brew install --cask mactex
```

**Ubuntu/Debian:**
```bash
sudo apt-get install texlive-full
```

**Windows:**
Install MiKTeX from https://miktex.org/

### Build Commands

```bash
# Using pdflatex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Or using make
make

# Or using latexmk
latexmk -pdf main.tex
```

## Sections

1. **Introduction** - Background, problem statement, contributions
2. **System Architecture** - Component overview with diagrams
3. **Environment Specification** - Observation and action spaces
4. **Task Specifications** - Detailed task definitions
5. **Reward System** - Reward structure and calculation
6. **Implementation Details** - Code examples and explanations
7. **Deployment** - Docker and HuggingFace deployment
8. **API Reference** - REST endpoint documentation
9. **Evaluation Metrics** - Success criteria and scoring
10. **Future Work** - Potential improvements
11. **Appendices** - Additional reference material

## Generating Diagrams

Architecture diagrams are generated using TikZ. The main architecture figure is embedded directly in `main.tex`.

To generate standalone diagrams:

```bash
pdflatex architecture.tex
```

## Bibliography

The bibliography uses BibTeX. Add references to `bibliography.bib` and compile with:

```bash
bibtex main
pdflatex main.tex
```

## License

This documentation is part of the ML Experiment Triage Environment project.
