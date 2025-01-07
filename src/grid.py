import numpy as np
import os
import math
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Polygon, box
from tqdm import tqdm
import re



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
                'grid_cell':point_names,
                'row':grid_row_names,
                'col':grid_col_names,
                'row_idx':grid_row_idx,
                'col_idx':grid_col_idx,
                'utm_zone':utm_zones,
                'utm_crs':epsgs
            },geometry=gpd.points_from_xy(grid_lons,grid_lats), crs='EPSG:4326')
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
        row_idx = point.row_idx
        col_idx = point.col_idx
        next_row_idx = row_idx+1
        next_col_idx = col_idx+1

        if next_row_idx >= len(self.lats): # If at top row, use difference between top and second-to-top row for height
            height = (self.lats[row_idx] - self.lats[row_idx-1])
            top = self.lats[row_idx] + height
        else:
            top = self.lats[next_row_idx]
        
        max_col = len(self.points_by_row[row_idx].col_idx)-1
        if next_col_idx > max_col: # If at rightmost column, use difference between rightmost and second-to-rightmost column for width
            width = (self.points_by_row[row_idx].iloc[col_idx].geometry.x - self.points_by_row[row_idx].iloc[col_idx-1].geometry.x)
            right = self.points_by_row[row_idx].iloc[col_idx].geometry.x + width
        else:
            right = self.points_by_row[row_idx].iloc[next_col_idx].geometry.x

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
        

        
    def generate_product_outlines_for_utm_zone(self, utm_zone, shift=340, pixel_size=10, raster_width=1068, raster_height=1068, 
                                                output_file=None, driver="ESRI Shapefile", get_footprints=False):
        """
        Generate the Major-TOM grid and product outlines for a given UTM zone.

        Args:
            utm_zone (str): UTM zone (e.g., "EPSG:32633").
            shift (int): Shift applied to the bottom-left corner (default is 340 meters).
            pixel_size (int): Pixel size in meters (default is 10).
            raster_width (int): Raster width in pixels (default is 1068).
            raster_height (int): Raster height in pixels (default is 1068).
            output_file (str): Path to save the resulting shapefile or GeoJSON.
            driver (str): File driver (e.g., "ESRI Shapefile", "GeoJSON").
            get_footprints (bool): Whether to include the `utm_footprint` column.

        Returns:
            GeoDataFrame: Product outlines for the specified UTM zone.
        """
        grid_points_utm = self.points[self.points['utm_crs'] == utm_zone]
        product_outlines = generate_product_outlines(
            grid_points_utm,
            shift=shift,
            pixel_size=pixel_size,
            raster_width=raster_width,
            raster_height=raster_height,
            get_footprints=get_footprints,
        )

        if output_file:
            output_file = ensure_file_extension(output_file, driver)
            product_outlines.to_file(output_file, driver=driver)
        return product_outlines
     
   

    def get_product_outline_for_cell(self, grid_cell=None, lat_lon=None, shift=340, pixel_size=10, raster_width=1068, raster_height=1068):
        """
        Retrieve the product outline for a single grid cell.

        Args:
            grid_cell (str): The grid cell name (optional).
            lat_lon (tuple): A tuple of (latitude, longitude) (optional).
            shift (int): Shift applied to the bottom-left corner (default is 340 meters).
            pixel_size (int): Pixel size in meters (default is 10).
            raster_width (int): Raster width in pixels (default is 1068).
            raster_height (int): Raster height in pixels (default is 1068).

        Returns:
            shapely.geometry.Polygon: Product outline for the specified grid cell.
        """
        if grid_cell:
            grid_row = self.points[self.points['grid_cell'] == grid_cell]
        elif lat_lon:
            lat, lon = lat_lon
            grid_row_idx = self.latlon2rowcol([lat], [lon], return_idx=True)[-1][0]
            grid_row = self.points.iloc[[grid_row_idx]]
        else:
            raise ValueError("Either 'grid_cell' or 'lat_lon' must be provided.")

        return generate_product_outlines(grid_row, shift=shift, pixel_size=pixel_size, raster_width=raster_width, raster_height=raster_height).iloc[0].geometry
        
    def generate_global_product_outlines_by_utm(self, output_folder, shift=340, pixel_size=10, raster_width=1068, raster_height=1068, 
                                                naming_convention="zone", driver="ESRI Shapefile", get_footprints=False):
        """
        Generate global product outlines grouped by UTM zones.

        Args:
            output_folder (str): Directory to save the shapefiles.
            shift (int): Shift applied to the bottom-left corner (default is 340 meters).
            pixel_size (int): Pixel size in meters (default is 10).
            raster_width (int): Raster width in pixels (default is 1068).
            raster_height (int): Raster height in pixels (default is 1068).
            naming_convention (str): Naming convention for filenames ("epsg" or "zone").
                                     - "epsg": Use EPSG code in the filename.
                                     - "zone": Use UTM zone with hemisphere in the filename.

        Returns:
            None
        """
        os.makedirs(output_folder, exist_ok=True)

        for utm_zone in self.points['utm_crs'].unique():
            print("Processing utm_zone... ", utm_zone)
            # Determine naming convention
            if naming_convention == "zone":
                if "EPSG:326" in utm_zone:
                    zone_number = int(utm_zone.split(":326")[1])  # UTM Zone = EPSG - 32600
                    hemisphere = "N"  # Northern Hemisphere
                elif "EPSG:327" in utm_zone:
                    zone_number = int(utm_zone.split(":327")[1])  # UTM Zone = EPSG - 32700
                    hemisphere = "S"  # Southern Hemisphere
                else:
                    print(f"Skipping EPSG code: {utm_zone} (not a valid UTM zone)")
                    continue
                output_filename = f"raster_outlines_UTM_zone_{zone_number}{hemisphere}.shp"
            else:
                # Default to using EPSG in the filename
                output_filename = f"raster_outlines_{utm_zone.replace(':', '_')}.shp"

            # Construct full path to output file
            output_file = os.path.join(output_folder, output_filename)

            # Generate the product outlines for the current UTM zone
            self.generate_product_outlines_for_utm_zone(
                utm_zone, 
                shift=shift, 
                pixel_size=pixel_size, 
                raster_width=raster_width, 
                raster_height=raster_height, 
                output_file=output_file,
                driver=driver,
                get_footprints=get_footprints
            )
            
    def filter_gridpoints_from_metadata(self, metadata):
        """
        Filter grid_points using the metadata.parquet file.

        Args:
            metadata: Dataframe with the contents of the metadata file (Parquet format) containing a 'grid_cell' column.
        Returns:
            Filtered grid points that match the metadata rows
        """

        if 'grid_cell' not in metadata.columns:
            raise ValueError("The metadata file must contain a 'grid_cell' column.")
        
        unique_grid_cells = metadata['grid_cell'].unique()
        print(f"Found {len(unique_grid_cells)} unique grid cells in metadata.")

        # Filter the grid points to include only the matching grid cells
        filtered_grid_points = self.points[self.points['grid_cell'].isin(unique_grid_cells)]
        print(f"Filtered grid points to {len(filtered_grid_points)} matching records.")
        
        return filtered_grid_points


                
def generate_product_outlines(grid_points_gdf, shift=340, pixel_size=10, raster_width=1068, raster_height=1068, get_footprints=False):
    """
    Optimized generation of product outlines from grid points by processing all points in a UTM zone together.

    Args:
        grid_points_gdf (GeoDataFrame): Grid points GeoDataFrame for a single UTM zone.
        shift (int): Distance in meters to shift the bottom-left point (default is 340m).
        pixel_size (int): Pixel size in meters (e.g., 10m for Sentinel-2).
        raster_width (int): Raster width in pixels (e.g., 1068).
        raster_height (int): Raster height in pixels (e.g., 1068).

    Returns:
        GeoDataFrame: GeoDataFrame with raster outlines as polygons.
    """
    # Reproject all points in the GeoDataFrame to the UTM CRS
    utm_crs = grid_points_gdf["utm_crs"].iloc[0]  # Get the CRS (all points in this group share the same CRS)
    reprojected_gdf = grid_points_gdf.to_crs(utm_crs)
    
    # Shift the bottom-left points
    reprojected_gdf["shifted_x"] = reprojected_gdf.geometry.x - shift
    reprojected_gdf["shifted_y"] = reprojected_gdf.geometry.y - shift

    # Vectorized bounding box creation
    reprojected_gdf["geometry"] = reprojected_gdf.apply(
        lambda row: box(
            row["shifted_x"],
            row["shifted_y"],
            row["shifted_x"] + raster_width * pixel_size,
            row["shifted_y"] + raster_height * pixel_size,
        ),
        axis=1,
    )
    if get_footprints: 
        reprojected_gdf["utm_footprint"] = reprojected_gdf["geometry"].astype('string')
        export_columns = ["grid_cell", "utm_crs", "geometry", "utm_footprint"]
    else:
        export_columns = ["grid_cell", "utm_crs", "geometry"]
    # Drop temporary columns and return the GeoDataFrame
    return reprojected_gdf[export_columns]

def merge_utm_files_to_wgs84(utm_folder, output_file, ext=".shp", driver="ESRI Shapefile"):
    """
    Merge UTM zone shapefiles into a single WGS84 file.

    Args:
        utm_folder (str): Directory containing UTM zone shapefiles.
        output_file (str): Path to save the merged WGS84 file.

    Returns:
        None
    """
    all_gdfs = []
    for file in os.listdir(utm_folder):
        if file.endswith(ext):
            print("processing file: ", file)
            gdf = gpd.read_file(os.path.join(utm_folder, file))
            gdf = gdf.to_crs("EPSG:4326")  # Reproject to WGS84
            all_gdfs.append(gdf)

    merged_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
    output_file = ensure_file_extension(output_file, driver)
    merged_gdf.to_file(output_file, driver=driver)
 
 
def ensure_file_extension(output_file, driver):
    """
    Ensure the correct file extension for the given driver.

    Args:
        output_file (str): The output file path.
        driver (str): The file driver (e.g., "ESRI Shapefile", "GeoJSON").

    Returns:
        str: The file path with the correct extension.
    """
    extension_map = {
        "ESRI Shapefile": ".shp",
        "GeoJSON": ".geojson",
        "GPKG": ".gpkg",
    }
    desired_extension = extension_map.get(driver, "")
    if not desired_extension:
        raise ValueError(f"Unsupported driver: {driver}")

    base, ext = os.path.splitext(output_file)
    if ext.lower() != desired_extension:
        output_file = f"{base}{desired_extension}"
    return output_file



def get_utm_zone_from_latlng(latlng):
    """
    Get the UTM zone from a latlng list and return the corresponding EPSG code.

    Parameters
    ----------
    latlng : List[Union[int, float]]
        The latlng list to get the UTM zone from.

    Returns
    -------
    str
        The EPSG code for the UTM zone.
    """
    assert isinstance(latlng, (list, tuple)), "latlng must be in the form of a list or tuple."

    longitude = latlng[1]
    latitude = latlng[0]

    zone_number = (math.floor((longitude + 180) / 6)) % 60 + 1

    # Special zones for Svalbard and Norway
    if latitude >= 56.0 and latitude < 64.0 and longitude >= 3.0 and longitude < 12.0:
        zone_number = 32
    elif latitude >= 72.0 and latitude < 84.0:
        if longitude >= 0.0 and longitude < 9.0:
            zone_number = 31
        elif longitude >= 9.0 and longitude < 21.0:
            zone_number = 33
        elif longitude >= 21.0 and longitude < 33.0:
            zone_number = 35
        elif longitude >= 33.0 and longitude < 42.0:
            zone_number = 37

    # Determine the hemisphere and construct the EPSG code
    if latitude < 0:
        epsg_code = f"327{zone_number:02d}"
    else:
        epsg_code = f"326{zone_number:02d}"
    if not re.match(r"32[6-7](0[1-9]|[1-5][0-9]|60)",epsg_code):
        print(f"latlng: {latlng}, epsg_code: {epsg_code}")
        raise ValueError(f"out of bound latlng resulted in incorrect EPSG code for the point")
    
    return epsg_code


if __name__ == '__main__':

    assert get_utm_zone_from_latlng([-1,-174.34]) == "32701"
    assert get_utm_zone_from_latlng([48,-4]) == "32630"
    assert get_utm_zone_from_latlng([78,13]) == "32633"
    assert get_utm_zone_from_latlng([-34,19.7]) == "32734"
    assert get_utm_zone_from_latlng([-36,175.7]) == "32760"


    dist = 100
    grid = Grid(dist)

    np.random.seed(0)
    test_lons = np.random.uniform(-20,20,size=(1000)) % 180 # Checks edge-case of crossing 180th meridian
    test_lats = np.random.uniform(-20,68,size=(1000))

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
    
    # Test 2: Single Grid Cell Outline
    grid_cell_name = "1U_17R"  # Replace with a valid grid cell name
    outline = grid.get_product_outline_for_cell(grid_cell=grid_cell_name)
    print(f"Product outline for grid cell {grid_cell_name}: {outline}")

    lat_lon = (45.0, 13.0)  # Replace with a valid lat/lon pair
    outline = grid.get_product_outline_for_cell(lat_lon=lat_lon)
    print(f"Product outline for lat/lon {lat_lon}: {outline}")
