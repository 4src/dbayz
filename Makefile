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
	tests.py lib.py ragworm.py

tests: ## run test suite
	python3 -B tests.py -g .
