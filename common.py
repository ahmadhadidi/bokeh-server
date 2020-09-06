# This is a common py file which contains all of the functions that I will be
# using in the project.
# import masterarbeit.blueprints.enums as enums


# We obtain the values of a specifc column from a CSV file.
def getValuesof1ColumnInACSV(path, column_name):
    import pandas as pd
    import sys
    dataframe = pd.read_csv(path, sep=",")

    try:
        rows = dataframe[column_name]
        return rows
        pass
    except Exception as e:
        print("something went wrong!")
        e = sys.exc_info()[0]
        print('>>> Exception Header: {0}'.format(e))
# return rows


# We obtain the values of as many columns as we want from a CSV file.
def getValuesofNColumnsInACSV(path, column_names):
    import pandas as pd
    import sys

    dataframe = pd.read_csv(path, sep=",")

    try:
        # We abuse the fact that lists are passed by surrounding elements with
        # square brackets so we don't add another bracket [[column_names]]
        rows = dataframe[column_names]
        return rows
        pass
    except Exception as e:
        print("something went wrong!")
        e = sys.exc_info()[0]
        print('>>> Exception Header: {0}'.format(e))


# This function communicates with the Google Directions API and 
# writes the responses to a file
def saveDirectionsToJSON(count):
    import sys
    h = {}
    for x in range(count):
        try:
            inc_id, id, start_lat, start_lng, end_lat, end_lng, start_Seconds = bikeSharingCoordinatesExtractor(
                x)
            result = googleRequest(start_lat, start_lng, end_lat, end_lng,
                                   enums.Keys.GOOGLE_DIRECTIONS_API_KEY.value)
            h[str(x+1)] = result
            print("ID: {0} \nJson Response: {1}".format(x, result))
        except Exception as e:
            print("something went wrong!")
            e = sys.exc_info()[0]
            print('>>> Exception Header: {0}'.format(e))
            break

    writeToJsonFile(h)

    print("Finished")


# Makes a request to the Directions API and returns the result
def googleRequest(origin_lat, origin_lng, destination_lat, destination_lng, API_KEY):
    import json
    import requests

    # We prepare the URL to request it from Google
    url = 'https://maps.googleapis.com/maps/api/directions/json?origin='+str(origin_lat)+',' + \
        str(origin_lng)+'+&destination='+str(destination_lat) + \
        ','+str(destination_lng)+'&mode=bicycling&key='+API_KEY

    # Request the URL
    r = requests.get(url)

    # If the response from the server was "OK" then we continue, otherwise we stop.
    if r.status_code != requests.codes.ok:
        r.raise_for_status()

    # Convert the response to something that Python understands
    results = json.loads(r.text)
    return results


# Writes the JSON object to a file
def writeToJsonFile(data):
    import json

    with open('json_file.json', 'w') as outfile:
        json.dump(data, outfile, indent=4, separators=(',', ': '))


# Gets the length of a dataframe
def bikeSharingLength():
    import pandas as pd

    # Load the bikesharing data into a dataframe
    dataframe = pd.read_csv(
        enums.DatasetPaths.BIKESHARING_DATASET_PATH.value, sep=",")

    return len(dataframe)


# Obtain 1 record of the Start and End trip lat/lng from the bikesharing Dataframe
def bikeSharingCoordinatesExtractor(rowNumber):
    import pandas as pd
    import numpy as np

    # Load the bikesharing data into a dataframe
    dataframe = pd.read_csv(enums.DatasetPaths.BIKESHARING_DATASET_PATH.value, sep=",")

    # Create a new column in the dataframe with the index of the dataset
    # Note: this will help us in case a direction failed to be obtained
    dataframe['inc_id'] = dataframe.index

    # Convert start datetime to a datetime column
    dataframe['start_datetime'] = pd.to_datetime(
        dataframe['start_datetime'])

    # Create a new column and save in it the datetime in Epoch
    dataframe['start_Seconds'] = ((dataframe['start_datetime'].astype(
        np.int64) // 10**9))

    print(dataframe[['start_Seconds', 'start_datetime']].iloc[rowNumber])

    # Assign the values to variables and return them
    inc_id, id, start_lat, start_lng, end_lat, end_lng, start_Seconds = dataframe[[
        'inc_id', 'id', 'start_lat', 'start_lng', 'end_lat', 'end_lng', 'start_Seconds']].iloc[rowNumber]

    return inc_id, id, start_lat, start_lng, end_lat, end_lng, start_Seconds


# Open a JSON file and return it as a Pandas Dataframe
def openJSONFileAndReturnDataFrame(path):
    import pandas as pd

    # Bind the json file to our code.
    # Example: './/masterarbeit//blueprints//torque//data//data.json'

    with open(path, 'r') as f:
        json_file = f.read()

    # Parse JSON into an object with attributes corresponding to dict keys.
    # x = json.loads(json_text, object_hook=lambda d: namedtuple(
    #     'X', d.keys())(*d.values()))

    # x2 = json.loads(json_text, object_hook=lambda d: namedtuple(
    #     d.keys())(*d.values()))
    df_json = pd.DataFrame.from_records(json_file)
    return df_json


# Calculates the mercato projection value from longtitudes and latitudes
# This conversion is required when we want to use the CARTO
# service provider in Bokeh.
# Example x,y = calculateMercatoProjectionFromLongAndLat(1.123, 1.123)
def calculateMercatoProjectionFromLongAndLat(longtitude, latitude):
    # Import the needed classes
    from pyproj import Proj, transform
    # longitude first, latitude second.
    xInMerc, yInMerc = transform(Proj(init='epsg:4326'), Proj(init='epsg:3857'),
                                 longtitude, latitude)
    # print("xInMerc is: {0} \nyInMerc is: {1}".format(xInMerc, yInMerc))

    return (xInMerc, yInMerc)


# Bokeh Styling
def plot_styler(p):
    p.title.text_font_size = enums.BokehStyles.chart_title_font_size.value
    p.title.text_font = enums.BokehStyles.chart_font.value
    p.title.align = enums.BokehStyles.chart_title_alignment.value
    p.title.text_font_style = enums.BokehStyles.chart_font_style_title.value
    p.y_range.start = 0
    p.x_range.range_padding = enums.BokehStyles.chart_inner_left_padding.value
    p.xaxis.axis_label_text_font = enums.BokehStyles.chart_font.value
    p.xaxis.major_label_text_font = enums.BokehStyles.chart_font.value
    p.xaxis.axis_label_standoff = enums.BokehStyles.default_padding.value
    p.xaxis.axis_label_text_font_size = enums.BokehStyles.axis_label_size.value
    p.xaxis.major_label_text_font_size = enums.BokehStyles.axis_ticks_size.value
    p.yaxis.axis_label_text_font = enums.BokehStyles.chart_font.value
    p.yaxis.major_label_text_font = enums.BokehStyles.chart_font.value
    p.yaxis.axis_label_text_font_size = enums.BokehStyles.axis_label_size.value
    p.yaxis.major_label_text_font_size = enums.BokehStyles.axis_ticks_size.value
    p.yaxis.axis_label_standoff = enums.BokehStyles.default_padding.value
    p.toolbar.logo = None
    p.toolbar_location = None


'''Provides utility functions for encoding and decoding linestrings using the
Google encoded polyline algorithm.
https://gist.github.com/signed0/2031157
'''


def encode_coords(coords):
    '''Encodes a polyline using Google's polyline algorithm

    See http://code.google.com/apis/maps/documentation/polylinealgorithm.html
    for more information.

    :param coords: Coordinates to transform (list of tuples in order: latitude,
    longitude).
    :type coords: list
    :returns: Google-encoded polyline string.
    :rtype: string
    '''

    result = []

    prev_lat = 0
    prev_lng = 0

    for x, y in coords:
        lat, lng = int(y * 1e5), int(x * 1e5)

        d_lat = _encode_value(lat - prev_lat)
        d_lng = _encode_value(lng - prev_lng)

        prev_lat, prev_lng = lat, lng

        result.append(d_lat)
        result.append(d_lng)

    return ''.join(c for r in result for c in r)


def _split_into_chunks(value):
    while value >= 32:  # 2^5, while there are at least 5 bits

        # first & with 2^5-1, zeros out all the bits other than the first five
        # then OR with 0x20 if another bit chunk follows
        yield (value & 31) | 0x20
        value >>= 5
    yield value


def _encode_value(value):
    # Step 2 & 4
    value = ~(value << 1) if value < 0 else (value << 1)

    # Step 5 - 8
    chunks = _split_into_chunks(value)

    # Step 9-10
    return (chr(chunk + 63) for chunk in chunks)


def decode(point_str):
    '''Decodes a polyline that has been encoded using Google's algorithm
    http://code.google.com/apis/maps/documentation/polylinealgorithm.html

    This is a generic method that returns a list of (latitude, longitude) 
    tuples.

    :param point_str: Encoded polyline string.
    :type point_str: string
    :returns: List of 2-tuples where each tuple is (latitude, longitude)
    :rtype: list

    '''

    # sone coordinate offset is represented by 4 to 5 binary chunks
    coord_chunks = [[]]
    for char in point_str:

        # convert each character to decimal from ascii
        value = ord(char) - 63

        # values that have a chunk following have an extra 1 on the left
        split_after = not (value & 0x20)
        value &= 0x1F

        coord_chunks[-1].append(value)

        if split_after:
            coord_chunks.append([])

    del coord_chunks[-1]

    coords = []

    for coord_chunk in coord_chunks:
        coord = 0

        for i, chunk in enumerate(coord_chunk):
            coord |= chunk << (i * 5)

        # There is a 1 on the right if the coord is negative
        if coord & 0x1:
            coord = ~coord  # invert
        coord >>= 1
        coord /= 100000.0

        coords.append(coord)

    # convert the 1 dimensional list to a 2 dimensional list and offsets to
    # actual values
    points = []
    prev_x = 0
    prev_y = 0
    for i in range(0, len(coords) - 1, 2):
        if coords[i] == 0 and coords[i + 1] == 0:
            continue

        prev_x += coords[i + 1]
        prev_y += coords[i]
        # a round to 6 digits ensures that the floats are the same as when
        # they were encoded
        points.append((round(prev_x, 6), round(prev_y, 6)))

    return points
    