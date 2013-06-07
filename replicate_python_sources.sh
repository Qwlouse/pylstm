#!/bin/bash

# $0 : name of command
# $1 : current binary dir

echo "current working dir:" `pwd`
echo "program name" $0
echo "destination" $1

if [ -f setup.py ];
then
    cp setup.py "$1/"
fi

rsync -am --include='*.py' -f 'hide,! */' src/pylstm "$1"
rsync -am --include='*.py' -f 'hide,! */' test/pylstm/* "$1/pylstm/test"
