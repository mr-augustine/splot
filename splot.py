#!/usr/bin/python

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

    myStatevar = Statevars(statevars_spec_filename)
    # print myStatevar.get_varlist()
    #
    # print 'datatype: ' + Statevars.get_field_datatype(myStatevar.get_varlist()[0])
    # print 'count: ' + str(Statevars.get_field_count(myStatevar.get_varlist()[0]))
    # print 'name: ' + Statevars.get_field_name(myStatevar.get_varlist()[0])
    # print 'expected value: ' + Statevars.get_field_exp_val(myStatevar.get_varlist()[0])
    #
    # print 'has_exp_val: ' + str(Statevars.has_exp_val(myStatevar.get_varlist()[0]))

    myData = Data(input_filename, myStatevar)

    # for frame in myData.get_data():
        # print frame['gps_hdop']
    #print myData.get_all('gps_hdop')

    plotter = Plot(plot_types_list_filename, myData)

# print 'Number of arguments: ', len(sys.argv), 'arguments.'
# print 'Argument list: ', str(sys.argv)


if __name__ == '__main__':
    main(sys.argv[1:])
