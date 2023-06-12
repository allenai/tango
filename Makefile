.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-autobuild -b html --watch tango/ --watch examples/ docs/source/ docs/build/

.PHONY : run-checks
run-checks :
	isort --check .
	black --check .
	ruff check .
	mypy --check-untyped-defs .
	CUDA_VISIBLE_DEVICES='' pytest -v --color=yes --doctest-modules --ignore=tests/integrations --ignore=tango/integrations tests/ tango/
	CUDA_VISIBLE_DEVICES='' pytest -v --color=yes --doctest-modules tango/integrations/torch tests/integrations/torch
	CUDA_VISIBLE_DEVICES='' pytest -v --color=yes --doctest-modules tango/integrations/transformers tests/integrations/transformers
