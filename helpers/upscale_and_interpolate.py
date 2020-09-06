import pandas as pd
import json
import numpy as np
from common import decode, calculateMercatoProjectionFromLongAndLat


# This function upscales and interpolates datapoints for the animation of Bokeh to work.
def process_data2(DEBUG):
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 99999)
    pd.set_option('display.float_format', lambda x: '%.6f' % x) # Suppresses scientific notation of floats z.b. 1.298e+06

    """### Dataset Path Binding"""

    path_json_directions_file = './data/100directions.json'
    path_bikehsharing_data = './data/bikesharing_data.csv'

    """### Binding to Dataframes"""

    df_bikesharing = pd.read_csv(path_bikehsharing_data)

    # Step 2.2: Load JSON file
    # since the dataset is in a JSON file, then we need
    # to open it using a JSON function.
    with open(path_json_directions_file, 'r') as f:
        google_directions_json = json.load(f)

    """### We take the first n numbers so that they could be matched with their respective directions"""

    df_bikesharing = df_bikesharing.head(len(google_directions_json))

    # We have 100

    """### Obtain trip duration from Google Directions"""

    df_bikesharing['g_directions_duration'] = 0

    """### Obtain duration from JSON reponses and save them into a new column"""

    for x in range(len(google_directions_json)):
        df_bikesharing['g_directions_duration'].at[x] = google_directions_json[str(x+1)
        ]['routes'][0]['legs'][0]['duration']['value']

    """### Convert start/end time in the CSV to datetime objects"""

    df_bikesharing['start_datetime'] = pd.to_datetime(
        df_bikesharing['start_datetime'])
    df_bikesharing['end_datetime'] = pd.to_datetime(
        df_bikesharing['end_datetime'])

    """### Change Datetime columns to Epoch"""

    df_bikesharing['start_seconds'] = (
        (df_bikesharing['start_datetime'].astype(np.int64) // 10**9))
    df_bikesharing['end_seconds'] = (
        (df_bikesharing['end_datetime'].astype(np.int64) // 10**9))

    """## Step 5: Add trip start time from ODK to Google directions trip duration

    ### We calculate the end trip time by adding the start_seconds and the duration from Google Directions together in a new column
    """

    df_bikesharing['g_end_seconds'] = df_bikesharing['g_directions_duration'] + \
                                      df_bikesharing['start_seconds']

    """## Prepare to decode the polyline and save its mercato coords in a cell

    ### We create a new column for the polyline_code from Google Directions
    """

    df_bikesharing['polyline_code'] = 'a'

    """### We fill the cells of the column from the JSON response for each trip"""

    for x in range(len(google_directions_json)):
        df_bikesharing['polyline_code'].at[x] = google_directions_json[str(x+1)
        ]['routes'][0]['overview_polyline']['points']

    """## Step 7: We decode the lng/lat coords from the polyline and put them into a separate dataframe for further processing.

    ### Step 7.1: Define lists that will be populated in the loop
    """

    lat_coords = list()
    lng_coords = list()
    trip_no = list()
    tick_list = list()
    step_list = list()

    """### Step 7.2: Populate our lists by decoding the polylines of each trip and by appending the trip number from a simple iterator "i"."""
    for x in range(len(google_directions_json)):
        decodedCoordsList = decode(
            google_directions_json[str(x+1)]['routes'][0]['overview_polyline']['points'])

        gDirectionsDuration = df_bikesharing['g_directions_duration'].at[x]
        startSecondsinEpoch = df_bikesharing['start_seconds'].at[x]
        global step
        step = round(gDirectionsDuration/len(decodedCoordsList))

        for i in range(len(decodedCoordsList)):
            lat_coords.append(decodedCoordsList[i][1])
            lng_coords.append(decodedCoordsList[i][0])
            trip_no.append(x)
            tick_list.append((step*i)+startSecondsinEpoch)
            step_list.append(step)

    """### Step 7.3: Create a dataframe object to hold the decoded coordinates and other data in step 7.2"""

    df_decoded_polylines = pd.DataFrame(
        {'tick': tick_list, 'lat': lat_coords, 'lng': lng_coords, 'trip_no': trip_no, 'step_to_next': step_list})

    """### Step 7.4: Calculate the mercato x,y coords from the lng,lat lists and save them into columns"""

    df_decoded_polylines['mercato_x'], df_decoded_polylines['mercato_y'] = calculateMercatoProjectionFromLongAndLat(
        df_decoded_polylines['lng'].to_list(), df_decoded_polylines['lat'].to_list())

    df_decoded_polylines

    """### We remove the e+09 values from the tick by setting the column 'tick' into int

    [Convert from epoch to datetime](https://stackoverflow.com/questions/16517240/pandas-using-unix-epoch-timestamp-as-datetime-index)
    """

    df_decoded_polylines = df_decoded_polylines.astype({'tick': 'datetime64[s]'})

    """### Create a multi-index to have trip_ids inside seconds"""

    #df_decoded_polylines.set_index(['tick', 'trip_no'], inplace=True)
    #df_decoded_polylines.set_index(['tick', 'trip_no'], inplace=True)

    double_index = df_decoded_polylines.set_index(['tick', 'trip_no'])
    double_index.sort_index(inplace=True)

    #df_decoded_polylines.reindex(pd.date_range(df_bikesharing['start_seconds'].head(1).values[0],end=df_bikesharing['g_end_seconds'].tail(1).values[0], periods=1))
    double_index

    """### We sort the index for the hierarchy to take effect
    [Section: The Multi-index of a pandas DataFrame](https://www.datacamp.com/community/tutorials/pandas-multi-index)
    * Finds out duplicate values: ```df_decoded_polylines[df_decoded_polylines.index.duplicated()]```

    * [Reindex Multi-Index](https://stackoverflow.com/questions/53286882/pandas-reindex-a-multiindex-dataframe)
    """

    df_decoded_polylines.sort_index(inplace=True)

    double_index.to_html('./debug/double_index.html')
    double_index.loc['2019-07-01 02:06:30']

    """### Fill the missing seconds

    *Journal*: Trying the question on Github
    * [Source](https://github.com/pandas-dev/pandas/issues/28313)

    > This works, checkout the output

    > Bug: does not interp my columns

    * [Resampling and Interpolating](https://machinelearningmastery.com/resample-interpolate-time-series-data-python/)

    * [How to select rows in a DataFrame between two values, in Python Pandas?](https://stackoverflow.com/questions/31617845/how-to-select-rows-in-a-dataframe-between-two-values-in-python-pandas)
    """

    # Fill the missing seconds
    print('[+] 198: Filling the missing seconds')
    github_interpd = double_index.reset_index('trip_no').groupby('trip_no', group_keys=False).resample('S').pad().reset_index().set_index(['tick','trip_no'])
    print('[-] Filling the missing seconds')
    # github_interpd['mercato_x'].loc[:,0]

    # Sort the index
    github_interpd.sort_index(inplace=True)
    # github_interpd.to_html('github_i.html')

    # Output the dataframe
    github_interpd

    """### Remove duplicate coordinates because the function padded them.

    #### Create a column to detect duplicates later on and set it to NaN
    """

    # Create a new column for detecting duplicates and set it to NaN
    github_interpd['duplicate'] = np.nan

    """#### [Unstack](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.unstack.html) the dataframe

    > We used ```-1``` for the level because we want to unstack the ```trip_ids``` ***not*** the ```timestamp```.
    """
    print('[+] 222: Unstacking..')
    g = github_interpd.unstack(level=-1, fill_value=None)
    print('[-] Finished unstacking')

    """#### Convert the index to a UNIX Timestamp"""

    # Convert index to UNIX Timestamp
    print('[+] 222: Converting to Unix Timestamp..')
    g.index = g.index.map(np.int64) // 10**9  # Pandarallel
    print('[-] Done with conversion')

    # Checkout our progress so far
    if DEBUG is True: g.to_html('./debug/unstack_w_epoch.html')

    """#### Create child columns inside our ```duplicate``` column"""

    # Create a column for detecting duplicate values and set its type to bool
    print('[+] 239: Creating a boolean column')
    g[['duplicate'][:]].astype(bool)
    print('[-] Creating a boolean column - Done')

    """#### We ascertain which coordinates are duplicates for all trips
    > We only need to do this for one coordinate because the coordinates are in pairs.

    > *Journal: I tried in many ways to find an alternative that doesn't use a for-loop but I couldn't.*
    """

    # Output the result of the duplicated() function into each trip
    for i in range(len(g['duplicate'].columns)):
        g['duplicate', i] = g['lat', i].duplicated().values

    """#### Set the duplicate values in each coordinate to NaN if they were a duplicate"""

    # Set the duplicate values in each coordinate to NaN if they were a duplicate
    g['lat'] = g['lat'].where((g['duplicate'] == False), np.NaN)
    g['lng'] = g['lng'].where((g['duplicate'] == False), np.NaN)
    g['mercato_x'] = g['mercato_x'].where((g['duplicate'] == False), np.NaN)
    g['mercato_y'] = g['mercato_y'].where((g['duplicate'] == False), np.NaN)

    """#### Interpolate the value between those two sets of coordinates linearly
    > *Journal: I limited the interpolation to 1 minute because if we didn't specify it then the trip would just remain on the map until the end of the slider.*

    >> Hint: if you are dealing with nested columns, then you need to talk to it via
    ```df['COL_NAME', <INDEX OR CHILD COLUMN NAME>]``` e.g. ```df['parent', 0:10]``` _**not**_ using ```df['parent'][0]``` as this would be understood as row number.
    """

    # Interpolate the empty values between each coordinate (accurate up to 1 minute)
    print('[+] 264: Interpolating...')
    g['lat'] = g['lat'].interpolate(limit=60)
    g['lng'] = g['lng'].interpolate(limit=60)
    g['mercato_x'] = g['mercato_x'].interpolate(limit=60)
    g['mercato_y'] = g['mercato_y'].interpolate(limit=60)
    print('[-] Interpolation finished')

    """#### Checkout our result!"""

    # Output the dataframe into an HTML file.
    if (DEBUG is True): g.to_html('./debug/interpolated.html')

    """# TEST: TO DICTIONARY AND DATASOURCE

    * [Convert A Level in a Multi-index to a another datatype](https://stackoverflow.com/questions/34417970/pandas-convert-index-type-in-multiindex-dataframe)

    * [Dataframe with multi-index to_dict()](https://stackoverflow.com/questions/39067831/dataframe-with-multiindex-to-dict)

    * [Multi-index from List of Lists with irregular length](https://stackoverflow.com/questions/58940018/multiindex-from-lists-of-lists-with-irregular-length)
    * [DataFrame with MultiIndex to dict](https://stackoverflow.com/questions/24988131/nested-dictionary-to-multiindex-dataframe-where-dictionary-keys-are-column-label)
    * [Nested dictionary to multiindex dataframe where dictionary keys are column labels](https://stackoverflow.com/questions/24988131/nested-dictionary-to-multiindex-dataframe-where-dictionary-keys-are-column-label)
    * [Python Dictionary Comprehension Tutorial](https://www.datacamp.com/community/tutorials/python-dictionary-comprehension)


    >> Hint: If you want to join two dataframes horizontally, then use ```result = pd.concat([g_duplicated, g_lat], axis=1)```

    ### Merge all of the columns into a list
    Resource: [Merge multiple column values into one column in python pandas
    ](https://stackoverflow.com/questions/33098383/merge-multiple-column-values-into-one-column-in-python-pandas)
    """

    # Merge all columns into a list
    # https://stackoverflow.com/questions/33098383/merge-multiple-column-values-into-one-column-in-python-pandas

    """#### Create A New Dataframe From The Coordinates For Each Trip"""

    # create a new dataframe from the coordinates for each trip
    tmp_mrx = g['mercato_x']
    tmp_mry = g['mercato_y']
    tmp_lat = g['lat']
    tmp_lng = g['lng']

    """#### Drop NA Values & Save Them As Lists In Their Cells"""

    # DropNA values and join the coordinates for each trip in each second as a string
    # TODO: try to implement this by using only lists.
    print('[+] 310: Applying...')
    tmp_mrx['merged_mrx'] = tmp_mrx[tmp_mrx.columns[0:]].apply(lambda x:','.join(x.dropna().astype(str)), axis=1)
    tmp_mry['merged_mry'] = tmp_mry[tmp_mry.columns[0:]].apply(lambda x:','.join(x.dropna().astype(str)), axis=1)
    tmp_lat['merged_lat'] = tmp_lat[tmp_lat.columns[0:]].apply(lambda x:','.join(x.dropna().astype(str)), axis=1)
    tmp_lng['merged_lng'] = tmp_lng[tmp_lng.columns[0:]].apply(lambda x:','.join(x.dropna().astype(str)), axis=1)

    # split the resulting string into a list of floats
    tmp_mrx['merged_mrx'] = tmp_mrx.merged_mrx.apply(lambda s: [float(x.strip(' []')) for x in s.split(',')])
    tmp_mry['merged_mry'] = tmp_mry.merged_mry.apply(lambda s: [float(x.strip(' []')) for x in s.split(',')])
    tmp_lat['merged_lat'] = tmp_lat.merged_lat.apply(lambda s: [float(x.strip(' []')) for x in s.split(',')])
    tmp_lng['merged_lng'] = tmp_lng.merged_lng.apply(lambda s: [float(x.strip(' []')) for x in s.split(',')])
    print('[-]: Finished applying...')
    print('----------------------------------------------------------------------')


    """#### Checkout Split Values For 1 Coordinate"""

    # checkout our progress for one of the variables so far
    if (DEBUG is True): tmp_lat.to_html('./debug/merged.html')

    """#### Merge Columns Into The Master Dataframe"""

    # Merge those columns into our master dataframe
    g['merged_mrx'] = tmp_mrx['merged_mrx']
    g['merged_mry'] = tmp_mry['merged_mry']
    g['merged_lat'] = tmp_lat['merged_lat']
    g['merged_lng'] = tmp_lng['merged_lng']

    #print(type(tmp_lat['merged_lat'])) #series

    """### Prepare for Visualization"""

    # We prepare a new dataframe for the visualization
    visualization = g
    visualization

    """#### We Drop Extra Columns That We Don't Need"""

    # We drop the extra columns as we only need the coords for each trip as a list
    print('[+] 353: Removing Unnecessary Columns...')
    visualization = visualization.drop(['lat', 'lng', 'mercato_x', 'mercato_y', 'duplicate', 'step_to_next'], axis=1)
    visualization

    """#### We drop the child level from the multi-level column index
    Resource: [Pandas: drop a level from a multi-level column index?](https://stackoverflow.com/questions/22233488/pandas-drop-a-level-from-a-multi-level-column-index)
    """

    # We drop the child level from the multi-level column index
    # https://stackoverflow.com/questions/22233488/pandas-drop-a-level-from-a-multi-level-column-index
    visualization['new_index'] = visualization.index
    visualization.columns = visualization.columns.droplevel(1)
    visualization.set_index('new_index', inplace=True)
    if DEBUG is True: visualization.to_html('visualization.html')

    """#### We create a dict with the key as the index of our dataframe"""

    # We create a dict with the key as the index of our dataframe
    ready_dict = visualization.to_dict('index')

    print("[+] Done with the upscaling and interpolation function")
    if DEBUG is True:
        visualization.to_parquet('./data/demo.parquet', engine='pyarrow')
        print('[+] Outputted the result of the function into >> ./data/demo.parquet for later demonstration purposes')

    print("[+][+][+][+][+] We Rowdy! [+][+][+][+][+][+]")
    return ready_dict