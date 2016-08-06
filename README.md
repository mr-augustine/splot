# splot
Statevars Plot (splot) graphs status variable data collected by a robot

You must have matplotlib installed.

Sample invocation:
./splot.py -i K00061.DAT -s statevars_sgco.txt -o plots.txt

where:
K00061.DAT is the input data file
statevars_sgco.txt defines the names, order, and datatypes for the statevars used in K00061.DAT
plots.txt lists the names of the plots that the application should create
