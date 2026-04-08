# Makefile for ML Experiment Triage Environment Documentation

TEXFILE = main
PDF = $(TEXFILE).pdf

.PHONY: all clean view install

all: $(PDF)

$(PDF): $(TEXFILE).tex
	@echo "Building documentation..."
	@pdflatex -interaction=nonstopmode $(TEXFILE)
	@bibtex $(TEXFILE) 2>/dev/null || true
	@pdflatex -interaction=nonstopmode $(TEXFILE)
	@pdflatex -interaction=nonstopmode $(TEXFILE)
	@echo "Done! PDF generated: $(PDF)"

clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(TEXFILE).aux $(TEXFILE).bbl $(TEXFILE).blg
	@rm -f $(TEXFILE).log $(TEXFILE).out $(TEXFILE).toc
	@rm -f $(TEXFILE).synctex.gz $(TEXFILE).fls $(TEXFILE).fdb_latexmk
	@rm -f $(PDF)
	@echo "Clean complete!"

view: $(PDF)
	@echo "Opening PDF..."
ifdef Darwin
	open $(PDF)
else ifneq ($(findstring CYGWIN,$(shell uname)),)
	cygstart $(PDF)
else ifneq ($(findstring MINGW,$(shell uname)),)
	start $(PDF)
else
	xdg-open $(PDF)
endif

install:
	@echo "Installing LaTeX distribution..."
ifdef Darwin
	brew install --cask mactex
else ifeq ($(shell which apt-get 2>/dev/null),)
	sudo apt-get install texlive-full
else ifeq ($(shell which yum 2>/dev/null),)
	sudo yum install texlive
endif

help:
	@echo "Available targets:"
	@echo "  all     - Build the PDF documentation (default)"
	@echo "  clean   - Remove all build artifacts"
	@echo "  view    - Build and open the PDF"
	@echo "  install - Install LaTeX distribution"
	@echo "  help    - Show this help message"
