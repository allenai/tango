.PHONY : docs
docs :
	sphinx-autobuild -b html --watch tango/ --watch examples --open-browser docs/source/ docs/build/
