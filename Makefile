prefix = "./built"
opts = ""
SHELL:=/bin/bash
build:
	mkdir build; cd build; cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=${prefix} ${opts} ..

clean:
	rm -rf build

gitreqs:
	git submodule update --recursive --remote --init

.PHONY:
	build clean gitreqs