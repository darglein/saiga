#!/bin/sh

#adds a suppression entry after every error in the outputfile
#--gen-suppressions=all
#--leak-check=full --show-reachable=yes
valgrind --tool=memcheck --leak-check=full --gen-suppressions=all --log-file=valgrind_output.txt --suppressions=valgrind.supp $1
