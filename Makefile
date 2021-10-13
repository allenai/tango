.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-autobuild -b html --watch tango/ --watch examples/ --open-browser docs/source/ docs/build/
