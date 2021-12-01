.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-autobuild -b html --watch tango/ --watch examples/ docs/source/ docs/build/

.PHONY : run-checks
run-checks :
	isort --check .
	black --check .
	flake8 .
	mypy .
	CUDA_VISIBLE_DEVICES='' pytest -v --color=yes --doctest-modules --ignore=tests/integrations --ignore=tango/integrations tests/ tango/
	CUDA_VISIBLE_DEVICES='' pytest -v --color=yes --doctest-modules tango/integrations/pytorch_lightning tests/integrations/pytorch_lightning
	CUDA_VISIBLE_DEVICES='' pytest -v --color=yes --doctest-modules tango/integrations/torch tests/integrations/torch
