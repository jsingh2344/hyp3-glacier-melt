import pandas as pd
import os
from pathlib import Path

def selectglaciersrgitable(glac_no=None, rgi_regionsO1=None, rgi_regionsO2='all', rgi_glac_number='all',
                           rgi_fp=None, rgi_cols_drop=None,
                           include_landterm=True, include_laketerm=True, include_tidewater=True,
                           glac_no_skip=None, min_glac_area_km2=0):
    """
    Select all glaciers to be used in the model run according to the regions and glacier numbers defined by the RGI
    glacier inventory. This function returns the rgi table associated with all of these glaciers.

    glac_no : list of strings
        list of strings of RGI glacier numbers (e.g., ['1.00001', '13.00001'])
    rgi_regionsO1 : list of integers
        list of integers of RGI order 1 regions (e.g., [1, 13])
    rgi_regionsO2 : list of integers or 'all'
        list of integers of RGI order 2 regions or simply 'all' for all the order 2 regions
    rgi_glac_number : list of strings
        list of RGI glacier numbers without the region (e.g., ['00001', '00002'])

    Output: Pandas DataFrame of the glacier statistics for each glacier in the model run
    (rows = GlacNo, columns = glacier statistics)
    """

    rgi_fp = Path(rgi_fp)
    if glac_no is not None:
        glac_no_byregion = {}
        rgi_regionsO1 = [int(i.split('.')[0]) for i in glac_no]
        rgi_regionsO1 = list(set(rgi_regionsO1))
        for region in rgi_regionsO1:
            glac_no_byregion[region] = []
        for i in glac_no:
            region = i.split('.')[0]
            glac_no_only = i.split('.')[1]
            glac_no_byregion[int(region)].append(glac_no_only)

        for region in rgi_regionsO1:
            glac_no_byregion[region] = sorted(glac_no_byregion[region])

    # Create an empty dataframe
    rgi_regionsO1 = sorted(rgi_regionsO1)
    glacier_table = pd.DataFrame()
    for region in rgi_regionsO1:

        if glac_no is not None:
            rgi_glac_number = glac_no_byregion[region]

        for i in os.listdir(rgi_fp):
            if str(region).zfill(2) in i:
                rgi_fp_reg = rgi_fp / i

        for i in os.listdir(rgi_fp_reg):
            if i.endswith('attributes.csv'):
                rgi_fn = i
                
        try:
            csv_regionO1 = pd.read_csv(rgi_fp_reg / rgi_fn)
        except:
            csv_regionO1 = pd.read_csv(rgi_fp_reg / rgi_fn, encoding='latin1')

        # Populate glacer_table with the glaciers of interest
        if rgi_regionsO2 == 'all' and rgi_glac_number == 'all':
            print("All glaciers within region(s) %s are included in this model run." % (region))
            if glacier_table.empty:
                glacier_table = csv_regionO1
            else:
                glacier_table = pd.concat([glacier_table, csv_regionO1], axis=0)
        elif rgi_regionsO2 != 'all' and rgi_glac_number == 'all':
            print("All glaciers within subregion(s) %s in region %s are included in this model run." %
                  (rgi_regionsO2, region))
            for regionO2 in rgi_regionsO2:
                regionO2_str = str(region).zfill(2) + '-' + str(regionO2).zfill(2)
                if glacier_table.empty:
                    glacier_table = csv_regionO1.loc[csv_regionO1['o2region'] == regionO2_str]
                else:
                    glacier_table = (pd.concat([glacier_table, csv_regionO1.loc[csv_regionO1['o2region'] ==
                                                                                regionO2_str]], axis=0))
        else:
            if len(rgi_glac_number) < 20:
                print("%s glaciers in region %s are included: %s" % (len(rgi_glac_number), region, rgi_glac_number))
            else:
                print("%s glaciers in region %s are included" % (len(rgi_glac_number), region))
                
            rgiid_subset = ['RGI2000-v7.0-G-' + str(region).zfill(2) + '-' + x.zfill(5) for x in rgi_glac_number]
            rgiid_all = list(csv_regionO1.rgi_id.values)
            rgi_idx = [rgiid_all.index(x) for x in rgiid_subset if x in rgiid_all]
            if glacier_table.empty:
                glacier_table = csv_regionO1.loc[rgi_idx]
            else:
                glacier_table = (pd.concat([glacier_table, csv_regionO1.loc[rgi_idx]],
                                           axis=0))

    glacier_table = glacier_table.copy()
    # reset the index so that it is in sequential order (0, 1, 2, etc.)
    glacier_table.reset_index(inplace=True)

    # drop connectivity 2 for Greenland and Antarctica
    glacier_table = glacier_table.loc[glacier_table['conn_lvl'].isin([0,1])]
    glacier_table.reset_index(drop=True, inplace=True)

    # add a simple glacier number column
    glacier_table['glacno'] = [int(x.split('-')[-1]) for x in glacier_table.rgi_id.values]

    # drop columns of data that is not being used
    glacier_table.drop(rgi_cols_drop, axis=1, inplace=True)

    # Longitude between 0-360deg (no negative)
    glacier_table['cenlon_360'] = glacier_table['cenlon']
    glacier_table.loc[glacier_table['cenlon'] < 0, 'cenlon_360'] = (
            360 + glacier_table.loc[glacier_table['cenlon'] < 0, 'cenlon_360'])
    # Subset glaciers based on their terminus type
    termtype_values = []
    if include_landterm:
        termtype_values.append(0)
        # assume dry calving, regenerated, and not assigned are land-terminating
        termtype_values.append(3)
        termtype_values.append(4)
        termtype_values.append(9)
    if include_tidewater:
        termtype_values.append(1)
        # assume shelf-terminating glaciers are tidewater
        termtype_values.append(5)
    if include_laketerm:
        termtype_values.append(2)
    glacier_table = glacier_table.loc[glacier_table['term_type'].isin(termtype_values)]
    glacier_table.reset_index(inplace=True, drop=True)

    # Remove glaciers below threshold
    glacier_table = glacier_table.loc[glacier_table['area_km2'] > min_glac_area_km2,:]
    glacier_table.reset_index(inplace=True, drop=True)

    # Remove glaciers that are meant to be skipped
    if glac_no_skip is not None:
        glac_no_all = list(glacier_table['glacno'])
        glac_no_unique = [x for x in glac_no_all if x not in glac_no_skip]
        unique_idx = [glac_no_all.index(x) for x in glac_no_unique]
        glacier_table = glacier_table.loc[unique_idx,:]
        glacier_table.reset_index(inplace=True, drop=True)

    # print("This study is focusing on %s glaciers in region %s" % (glacier_table.shape[0], rgi_regionsO1))
    return glacier_table