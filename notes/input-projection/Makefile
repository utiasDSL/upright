PDF_FILE=input-projection.pdf
TEX_FILE=input-projection.tex
FLAGS=--bibtex-cmd=biber --latex-cmd pdflatex
LATEX_OUT_DIR=.latex.out

.PHONY: FORCE
$(PDF_FILE): FORCE
	@latexrun $(FLAGS) $(TEX_FILE) -O $(LATEX_OUT_DIR) -o $(PDF_FILE)

.PHONY: clean
clean:
# Remove the old PDF.
# Leading dash causes processing to proceed even if errors occur.
	@-rm -f $(PDF_FILE)
	@-rm -rf $(LATEX_OUT_DIR)
# Does not remove the $(LATEX_OUT_DIR) itself:
# @latexrun -O $(LATEX_OUT_DIR) --clean-all
