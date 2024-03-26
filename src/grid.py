import numpy as np
import math
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Polygon
from tqdm import tqdm



class Grid():

    RADIUS_EQUATOR = 6378.137 # km

    def __init__(self,dist,latitude_range=(-85,85),longitude_range=(-180,180),utm_definition='bottomleft'):
        self.dist = dist
        self.latitude_range = latitude_range
        self.longitude_range = longitude_range
        self.utm_definition = utm_definition
        self.rows,self.lats = self.get_rows()
        self.points, self.points_by_row = self.get_points()
        
    def get_rows(self):

        # Define set of latitudes to use, based on the grid distance
        arc_pole_to_pole = math.pi * self.RADIUS_EQUATOR
        num_divisions_in_hemisphere = math.ceil(arc_pole_to_pole / self.dist)

        latitudes = np.linspace(-90, 90, num_divisions_in_hemisphere+1)[:-1]
        latitudes = np.mod(latitudes, 180) - 90

        # order should be from south to north
        latitudes = np.sort(latitudes)

        zeroth_row = np.searchsorted(latitudes,0)

        # From 0U-NU and 1D-ND
        rows = [None] * len(latitudes)
        rows[zeroth_row:] = [f'{i}U' for i in range(len(latitudes)-zeroth_row)]
        rows[:zeroth_row] = [f'{abs(i-zeroth_row)}D' for i in range(zeroth_row)]

        # bound to range
        idxs = (latitudes>=self.latitude_range[0]) * (latitudes<=self.latitude_range[1])
        rows,latitudes = np.array(rows), np.array(latitudes)
        rows,latitudes = rows[idxs],latitudes[idxs]

        return rows,latitudes

    def get_circumference_at_latitude(self,lat):

        # Circumference of the cross-section of a sphere at a given latitude

        radius_at_lat = self.RADIUS_EQUATOR * math.cos(lat * math.pi / 180)
        circumference = 2 * math.pi * radius_at_lat

        return circumference

    def subdivide_circumference(self,lat,return_cols=False):
        # Provide a list of longitudes that subdivide the circumference of the earth at a given latitude
        # into equal parts as close as possible to dist

        circumference = self.get_circumference_at_latitude(lat)
        num_divisions = math.ceil(circumference / self.dist)
        longitudes = np.linspace(-180,180, num_divisions+1)[:-1]
        longitudes = np.mod(longitudes, 360) - 180
        longitudes = np.sort(longitudes)


        if return_cols:
            cols = [None] * len(longitudes)
            zeroth_idx = np.where(longitudes==0)[0][0]
            cols[zeroth_idx:] = [f'{i}R' for i in range(len(longitudes)-zeroth_idx)]
            cols[:zeroth_idx] = [f'{abs(i-zeroth_idx)}L' for i in range(zeroth_idx)]
            return np.array(cols),np.array(longitudes)

        return np.array(longitudes)

    def get_points(self):
        
        r_idx = 0
        points_by_row = [None]*len(self.rows)
        for r,lat in zip(self.rows,self.lats):
            point_names,grid_row_names,grid_col_names,grid_row_idx,grid_col_idx,grid_lats,grid_lons,utm_zones,epsgs = [],[],[],[],[],[],[],[],[]
            cols,lons = self.subdivide_circumference(lat,return_cols=True)

            cols,lons = self.filter_longitude(cols,lons)
            c_idx = 0
            for c,lon in zip(cols,lons):
                point_names.append(f'{r}_{c}')
                grid_row_names.append(r)
                grid_col_names.append(c)
                grid_row_idx.append(r_idx)
                grid_col_idx.append(c_idx)
                grid_lats.append(lat)
                grid_lons.append(lon)
                if self.utm_definition == 'bottomleft':
                    utm_zones.append(get_utm_zone_from_latlng([lat,lon]))
                elif self.utm_definition == 'center':
                    center_lat = lat + (1000*self.dist/2)/111_120
                    center_lon = lon + (1000*self.dist/2)/(111_120*math.cos(center_lat*math.pi/180))
                    utm_zones.append(get_utm_zone_from_latlng([center_lat,center_lon]))
                else:
                    raise ValueError(f'Invalid utm_definition {self.utm_definition}')
                epsgs.append(f'EPSG:{utm_zones[-1]}')

                c_idx += 1
            points_by_row[r_idx] = gpd.GeoDataFrame({
                'name':point_names,
                'row':grid_row_names,
                'col':grid_col_names,
                'row_idx':grid_row_idx,
                'col_idx':grid_col_idx,
                'utm_zone':utm_zones,
                'epsg':epsgs
            },geometry=gpd.points_from_xy(grid_lons,grid_lats))
            r_idx += 1
        points = gpd.GeoDataFrame(pd.concat(points_by_row))
        # points.reset_index(inplace=True,drop=True)
        return points, points_by_row

    def group_points_by_row(self):
        # Make list of different gdfs for each row
        points_by_row = [None]*len(self.rows)
        for i,row in enumerate(self.rows):
            points_by_row[i] = self.points[self.points.row==row]
        return points_by_row

    def filter_longitude(self,cols,lons):
        idxs = (lons>=self.longitude_range[0]) * (lons<=self.longitude_range[1])
        cols,lons = cols[idxs],lons[idxs]
        return cols,lons

    def latlon2rowcol(self,lats,lons,return_idx=False,integer=False):
        """
        Convert latitude and longitude to row and column number from the grid
        """
        # Always take bottom left corner of grid cell
        rows = np.searchsorted(self.lats,lats)-1

        # Get the possible points of the grid cells at the given latitude
        possible_points = [self.points_by_row[row] for row in rows]

        # For each point, find the rightmost point that is still to the left of the given longitude
        cols = [poss_points.iloc[np.searchsorted(poss_points.geometry.x,lon)-1].col for poss_points,lon in zip(possible_points,lons)]
        rows = self.rows[rows].tolist()

        outputs = [rows, cols]
        if return_idx:
            # Get the table index for self.points with each row,col pair in rows, cols
            idx = [self.points[(self.points.row==row) & (self.points.col==col)].index.values[0] for row,col in zip(rows,cols)]
            outputs.append(idx)

        # return raw numbers
        if integer:
            outputs[0] = [int(el[:-1]) if el[-1] == 'U' else -int(el[:-1]) for el in outputs[0]]
            outputs[1] = [int(el[:-1]) if el[-1] == 'R' else -int(el[:-1]) for el in outputs[1]]
            
        return outputs

    def rowcol2latlon(self,rows,cols):
        point_geoms = [self.points.loc[(self.points.row==row) & (self.points.col==col),'geometry'].values[0] for row,col in zip(rows,cols)]
        lats = [point.y for point in point_geoms]
        lons = [point.x for point in point_geoms]
        return lats,lons

    def get_bounded_footprint(self,point,buffer_ratio=0):
        # Gets the polygon footprint of the grid cell for a given point, bounded by the other grid points' cells.
        # Grid point defined as bottom-left corner of polygon. Buffer ratio is the ratio of the grid cell's width/height to buffer by.

        bottom,left = point.geometry.y,point.geometry.x
        row = point.row
        row_idx = point.row_idx
        col_idx = point.col_idx
        next_row_idx = row_idx+1
        next_col_idx = col_idx+1

        if next_row_idx >= len(self.lats): # If at top row, use difference between top and second-to-top row for height
            height = (self.lats[row_idx] - self.lats[row_idx-1])
            top = self.lats[row_idx] + height
        else:
            top = self.lats[next_row_idx]
        
        max_col = len(self.points_by_row[row].col_idx)-1
        if next_col_idx > max_col: # If at rightmost column, use difference between rightmost and second-to-rightmost column for width
            width = (self.points_by_row[row].iloc[col_idx].geometry.x - self.points_by_row[row].iloc[col_idx-1].geometry.x)
            right = self.points_by_row[row].iloc[col_idx].geometry.x + width
        else:
            right = self.points_by_row[row].iloc[next_col_idx].geometry.x

        # Buffer the polygon by the ratio of the grid cell's width/height
        width = right - left
        height = top - bottom

        buffer_horizontal = width * buffer_ratio
        buffer_vertical = height * buffer_ratio

        new_left = left - buffer_horizontal
        new_right = right + buffer_horizontal

        new_bottom = bottom - buffer_vertical
        new_top = top + buffer_vertical

        bbox = Polygon([(new_left,new_bottom),(new_left,new_top),(new_right,new_top),(new_right,new_bottom)])

        return bbox
    

def get_utm_zone_from_latlng(latlng):
    """
    Get the UTM ZONE from a latlng list.

    Parameters
    ----------
    latlng : List[Union[int, float]]
        The latlng list to get the UTM ZONE from.

    return_epsg : bool, optional
        Whether or not to return the EPSG code instead of the WKT, by default False

    Returns
    -------
    str
        The WKT or EPSG code.
    """
    assert isinstance(latlng, (list, np.ndarray)), "latlng must be in the form of a list."

    zone = math.floor(((latlng[1] + 180) / 6) + 1)
    n_or_s = "S" if latlng[0] < 0 else "N"

    false_northing = "10000000" if n_or_s == "S" else "0"
    central_meridian = str(zone * 6 - 183)
    epsg = f"32{'7' if n_or_s == 'S' else '6'}{str(zone)}"

    return epsg


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dist = 100
    grid = Grid(dist,latitude_range=(10,70),longitude_range=(-30,60))

    from pprint import pprint

    test_lons = np.random.uniform(-20,50,size=(1000))
    test_lats = np.random.uniform(12,68,size=(1000))

    test_rows,test_cols = grid.latlon2rowcol(test_lats,test_lons)
    test_lats2,test_lons2 = grid.rowcol2latlon(test_rows,test_cols)

    print(test_lons[:10])
    print(test_lats[:10])
    print(test_rows[:10])
    print(test_cols[:10])

    # Make line segments from the points to their corresponding grid points
    lines = []
    for i in range(len(test_lats)):
        lines.append([(test_lons[i],test_lats[i]),(test_lons2[i],test_lats2[i])])

    lines = gpd.GeoDataFrame(geometry=gpd.GeoSeries([LineString(line) for line in lines])) 

    lines.to_file(f'testlines_{dist}km.geojson',driver='GeoJSON')
    grid.points.to_file(f'testgrid_{dist}km.geojson',driver='GeoJSON')
