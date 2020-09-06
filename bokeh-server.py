from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler

from bokeh.plotting import Figure
from bokeh.embed import json_item
from bokeh.models import CustomJS
from bokeh.sampledata.autompg import autompg

# My imports
import numpy as np
import pandas as pd
import json
import time
from common import calculateMercatoProjectionFromLongAndLat
from bokeh.layouts import column
from bokeh.tile_providers import get_provider, Vendors  # I know this shows an error in your IDE but don't delete it
from bokeh.models import (HoverTool, Slider, Button)
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models.annotations import Title
from bokeh.io import curdoc, output_file, show
from sklearn import cluster


def hello_world():
    return 'Hello, World!'


def plot2():
    # copy/pasted from Bokeh 'JavaScript Callbacks' - used as an example
    # https://bokeh.pydata.org/en/latest/docs/user_guide/interaction/callbacks.html

    x = [x*0.005 for x in range(0, 200)]
    y = x

    source = ColumnDataSource(data=dict(x=x, y=y))

    plot = Figure(plot_width=400, plot_height=400)
    plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

    callback = CustomJS(args=dict(source=source), code="""
        var data = source.data;
        var f = cb_obj.value
        var x = data['x']
        var y = data['y']
        for (var i = 0; i < x.length; i++) {
            y[i] = Math.pow(x[i], f)
        }
        source.change.emit();
    """)

    slider = Slider(start=0.1, end=4, value=1, step=.1, title="power")
    slider.js_on_change('value', callback)
    layout = column(slider, plot)

    return json.dumps(json_item(layout, "myplot"))


def plot1():
    # copy/pasted from Bokeh Getting Started Guide - used as an example
    grouped = autompg.groupby("yr")
    mpg = grouped.mpg
    avg, std = mpg.mean(), mpg.std()
    years = list(grouped.groups)
    american = autompg[autompg["origin"]==1]
    japanese = autompg[autompg["origin"]==3]

    p = Figure(title="MPG by Year (Japan and US)")

    p.vbar(x=years, bottom=avg-std, top=avg+std, width=0.8,
           fill_alpha=0.2, line_color=None, legend="MPG 1 stddev")

    p.circle(x=japanese["yr"], y=japanese["mpg"], size=10, alpha=0.5,
             color="red", legend="Japanese")

    p.triangle(x=american["yr"], y=american["mpg"], size=10, alpha=0.3,
               color="blue", legend="American")

    p.legend.location = "top_left"
    return json.dumps(json_item(p, "myplot"))

def makina(my_dict):
    # TODO: make the makina function a function that accepts a dataframe
    # and it returns a plot.


    # Coloring the clusters
    colors = np.array(
        [x for x in ('#00f', '#0f0', '#f00', '#0ff', '#f0f', '#ff0')])
    colors = np.hstack([colors] * 20)

    # Bind a column or several columns for the model.
    columns = my_dict[['merged_mrx', 'merged_mry', 'merged_lat', 'merged_lng']].iloc[: , :].values.flatten()
    _1column = my_dict[['merged_mrx']].iloc[:,:].values.flatten()

    # obtain the values for all seconds for those 2 columns
    _2columns = my_dict[['merged_mrx', 'merged_mry']].iloc[:,:].values

    # Choose the first 300 seconds
    _2columns = _2columns[0:299]

    # we now have a multi-dimensional array. In order to flatten it
    # we need to use ravel().
    # Source: https://stackoverflow.com/questions/33711985/flattening-a-list-of-numpy-arrays/33718947
    for i in range(len(_2columns)):
        _2columns[i] = np.concatenate(_2columns[i]).ravel()


    # Dashboard Panel for ML Algorithm
    # Initialize the ML algorithm
    dbscan = cluster.DBSCAN(eps=20)
    means = cluster.KMeans(n_clusters=2)

    # Bind the algorithm to a variable to be referenced later.
    algorithm = dbscan

    # The first 300 records only have 1 value in the cell
    # let's try to predict a centroid for all of them.
    # HINT: (rows,columns)
    means.fit(_1column[0:299].reshape(-1,1))
    dbscan.fit(_1column[0:299].reshape(-1,1))

    xInMerc, yInMerc = calculateMercatoProjectionFromLongAndLat(
        6.8272388, 50.9578353)

    X = _2columns
    algorithm.fit(_2columns)
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)

    p = figure(
        x_range=(int(xInMerc)+13500, int(xInMerc)+15000), y_range=(int(yInMerc)-11000, int(yInMerc)+5000),
        x_axis_type="mercator", y_axis_type="mercator", x_axis_label='Longtitude', y_axis_label='Latitude',
        output_backend="webgl", title=algorithm.__class__.__name__,
        plot_width=500, plot_height=500
    )

    # Use CARTODB as our map
    tile_provider = get_provider(Vendors.CARTODBPOSITRON)

    # Add a map to the background of the figure
    p.add_tile(tile_provider)

    # Scatter the predicted values and color the predicted values
    p.scatter(X[:, 0], X[:, 1], alpha=0.8, color=colors[y_pred].tolist())

    # Output the clustering result in an html file.
    output_file("./debug/clustering_dbscan.html", title="DBSCAN for first 300 Seconds")

    # Call a function so that it opens our browser and automatically
    # opens the html file from the previous line.
    show(p)


def demo():
    # This function is used as a demonstration purpose, it uses the outputted parquet file from the
    # whatever resulted from the interpolator function and only uses the parquet file.
    path_to_demo = './data/demo.parquet'
    # path_to_demo2 = 'D:\\Dev\\Projects\\Python\\visualization-bokeh\\data\\demo.parquet'
    df_demo = pd.read_parquet(path_to_demo)
    return df_demo


def make_document(doc):

    # XXX: Changing the Debug value to True will require more than 3 minutes to finish executing
    # my_sources = process_data2(DEBUG=False)
    my_sources = demo().to_dict('index')

    # Call Machine Learning Algorithm
    makina(demo())
    source2 = ColumnDataSource(data=my_sources[[1561946460][0]])

    # Bind only the keys of my_sources dict as
    # a list into a variable. We will need this
    # in the Slider object.
    ticks = list(my_sources.keys())

    # Base coordinates for the plot
    xInMerc, yInMerc = calculateMercatoProjectionFromLongAndLat(
        6.8272388, 50.9578353)
    curdoc().theme = 'dark_minimal'
    plot = figure(x_range=(int(xInMerc)+13500, int(xInMerc)+15000), y_range=(int(yInMerc)-11000, int(yInMerc)+5000),
                  x_axis_type="mercator", y_axis_type="mercator", x_axis_label='Longtitude', y_axis_label='Latitude', toolbar_location="right", plot_width=1500, plot_height=800)

    tile_provider = get_provider(Vendors.CARTODBPOSITRON)

    plot.add_tile(tile_provider)

    title = Title(text=str(
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ticks[0]))), align="center")
    plot.add_layout(title, "below")

    # Add the title of the figure
    t = Title()
    t.text = 'Trip Line in Cologne'
    plot.title = t

    # Make a square glyph that uses the coordinates coming from source2
    plot.square(
        x='merged_mrx',
        y='merged_mry',
        source=source2,
        size=10,
        fill_color="#74add1",
    )

    plot.add_tools(HoverTool(tooltips=[('Longtitude', '@merged_lng'), ('Latitude', '@merged_lat')],
                             show_arrow=False, point_policy='follow_mouse'))

    def animate_update():
        # year = slider.value + 1
        tick = slider.value + 1
        if tick > int(ticks[-1]):
            tick = int(ticks[0])
        slider.value = tick

    def slider_update(attrname, old, new):
        """
        This function is called every 200ms to change the
        label that shows the date and time which appears in the figure.
        """
        tick = slider.value
        title.text = str(time.strftime(
            '%Y-%m-%d %H:%M:%S', time.localtime(tick)))
        source2.data = my_sources[tick]

    # Initialize the slider object to begin from the beginning of the list
    # TODO: Step is going to be different, because the distance each unique tick is not the same
    slider = Slider(start=ticks[0], end=ticks[-1],
                    value=ticks[0], step=1, title="", width=1500, show_value=False)
    slider.on_change('value', slider_update)

    callback_id = None

    def animate():
        """
        This function is called when we press the play button, it adds a callback
        that keeps being called every 200ms. When it is stopped, the callback
        will be stopped.
        """
        global callback_id
        if button.label == '► Play':
            button.label = '❚❚ Pause'
            callback_id = doc.add_periodic_callback(animate_update, 100)
        else:
            button.label = '► Play'
            doc.remove_periodic_callback(callback_id)

    # We declare the button, it's dimensions and label.
    button = Button(label='► Play', width=60)

    # call the function animate() above when it is pressed
    button.on_click(animate)

    # Bind the button, slider and plot to the doc which is
    # needed in our bokeh server.

    doc.add_root(plot)
    doc.add_root(slider)
    doc.add_root(button)

    print("Server running on http://localhost:5000")


# Run the server
apps = {'/': Application(FunctionHandler(make_document))}

server = Server(apps, port=5000)
server.start()
server.run_until_shutdown()