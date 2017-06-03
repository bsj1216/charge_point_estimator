import numpy as np
import pandas as pd
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file

t = range(10)
a = [i + 1 for i in range(10)]
b = [i + 3 for i in range(10)]

pl = figure(title="Visualization test")
pl.grid.grid_line_alpha = 0.3
pl.xaxis.axis_label = 'Event index'
pl.yaxis.axis_label = 'Energy consumption'

pl.line(t, a,color='#A6CEE3', legend='a')
pl.line(t, b,color='#B2DF8A', legend='b')
pl.legend.location = "top_left"

output_file("test.html", title="prediction example")

show(gridplot([[pl]],plot_width=400, plot_height=400))