.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-autobuild -b html --watch tango/ --watch examples/ docs/source/ docs/build/
