-include ../config/do.mk

DO_what=      RAGWORM: smallest brain I can imagine
DO_copyright= Copyright (c) 2023 Tim Menzies, BSD-2.
DO_repos=     . ../config ../data

install: ## load python3 packages (requires `pip3`)
	 pip3 install -qr requirements.txt

../data:
	(cd ..; git clone https://gist.github.com/d47b8699d9953eef14d516d6e54e742e.git data)

../config:
	(cd ..; git clone https://gist.github.com/42f78b8beec9e98434b55438f9983ecc.git config)

doc: ## generate documentation
	pdoc --html                     \
	--config show_source_code=True    \
	--config sort_identifiers=False     \
	--force -o docs --template-dir docs  \
	bayes2.py

tests: ## run test suite
	if ./less.py -ok;\
		then cp docs/pass.png docs/results.png; \
		else cp docs/fail.png docs/results.png; \
  fi

docs/fishn.html:
	pycco fishn.py
	cp docs/_pycco.css docs/pycco.css

less.md : less.py head.md
	cat head.md > $@
	gawk -f doc.awk $< >> $@

less.pdf : less.md
	pandoc -s --pdf-engine=xelatex --toc -N \
		        --listings -H listings-setup.tex \
		         -o $@ $<

docs/less.html: less.py
	 python3 -m pdoc -c sort_identifiers=False  \
		       --template-dir docs --force --html -o docs less.py
