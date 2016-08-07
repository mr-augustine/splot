#!/usr/bin/python

"""
file: splot.py
created: 20160806
author(s): mr-augustine

Statevars Plot (splot) produces plots for data in a given data file according to
the file's user-defined format specification.
"""
import getopt
import os.path
import sys

from statevars import Statevars
from data import Data
from plot import Plot

HELP_TEXT = 'usage: ' + sys.argv[0] + ' -i <input file> -s <statevars def> -o <output plot list>'

def main(argv):
    input_filename = ''
    statevars_spec_filename = ''
    plot_types_list_filename = ''
    plot_spec_filename = ''

    try:
        options, args = getopt.getopt(argv, "hi:s:o:", ["input=", "statevars=", "plots="])
    except getopt.GetoptError:
        print HELP_TEXT
        sys.exit(2)

    # Reject any invocation that doesn't have the exact number of arguments
    if len(argv) != 6:
        print HELP_TEXT
        sys.exit(2)

    for (option, argument) in options:
        if option in ('-i', '--input'):
            print 'input file: ' + argument
            input_filename = argument

        elif option in ('-s', '--statevars'):
            print 'statevars definition file: ' + argument
            statevars_spec_filename = argument

        elif option in ('-o', '--plots'):
            print 'output spec: ' + argument
            plot_types_list_filename = argument

        elif option in ('-p', '--plot-defs'):
            print 'plot definition file: ' + argument
            plot_spec_filename = argument

        else:
            print HELP_TEXT

        if not os.path.isfile(argument):
            print 'File: ' + argument + ' does not exist. Exiting...'
            sys.exit(2)

    # Parse the statevars file
    myStatevar = Statevars(statevars_spec_filename)

    # Use the parsed statevars to interpet the data in the data file
    myData = Data(input_filename, myStatevar)

    # Use the parsed data to create the plots specified by the user
    plotter = Plot(plot_types_list_filename, myData)

if __name__ == '__main__':
    main(sys.argv[1:])
