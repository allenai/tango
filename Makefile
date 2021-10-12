.PHONY : docs
docs :
	sphinx-autobuild -b html --watch tango/ --open-browser docs/source/ docs/build/
