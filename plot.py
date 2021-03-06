"""
file: plot.py
created: 20160806
author(s): mr-augustine

The Plot class is responsible for plotting the statevars data according to
the list of plot names specified in the given file.

Lines in the plot file would look similar to this:
main_loop_counter
#odometer_ticks
compass_vs_gps_heading

In the example above, the main_loop_counter plot would be created in Figure 1,
and the compass_vs_gps_heading plot would be created in Figure 2. The
odometer_ticks plot will not be created since that line is commented out.
"""
from math import radians, degrees, cos, sin, asin, sqrt, atan2, fabs
import matplotlib.pyplot as plt
import numpy as np
import sys

from data import Data

class Plot:

    SUPPORTED_PLOT_TYPES = ['compass_vs_gps_heading',
                            'compass_vs_gps_vs_nav_heading',
                            'control_heading_desired',
                            'control_steering_pwm_vs_commanded',
                            'control_xtrack_error',
                            'control_xtrack_error_rate',
                            'control_xtrack_error_sum',
                            'desired_vs_actual_headings',
                            'gps_coordinates',
                            'gps_cumulative_dist',
                            'gps_ground_course_deg',
                            'gps_ground_speed_kt',
                            'gps_ground_speed_mph',
                            'gps_hdop',
                            'gps_pdop',
                            'gps_vdop',
                            'gps_satcount',
                            'heading_deg',
                            'main_loop_counter',
                            'meters_per_gps_update',
                            'mobility_motor_pwm',
                            'nav_distance_to_waypoint',
                            'nav_heading_deg',
                            'nav_positions',
                            'nav_relative_bearings',
                            'nav_speed',
                            'odometer_ticks',
                            'odometer_ticks_per_iteration',
                            'odometer_timestamp',
                            'pitch_deg',
                            'position_estimates',
                            'roll_deg',
                            'simulate_nav_positions',
                            'simulate_xtrack_error',
                            'status',
                            'ticks_per_gps_update',
                            'ticks_per_meter_per_gps_update']

    _plots = {}
    _data = None

    def __init__(self, filename, data):
        self._parse_plot_list_file(filename)

        self._data = data

        for plot_name in self._plots.keys():
            self._prepare_plot(plot_name)

        plt.show()

    def _parse_plot_list_file(self, filename):
        """ Parses the specified file for supported plot types and associates
        each plot with a figure number.
        """

        with open(filename) as f:
            lines = f.readlines()

        figure_number = 1
        for line in lines:
            line = str.strip(line)

            # Ignore empty lines and comment lines.
            if line == '' or line[0] == '#':
                continue

            # Assign figure numbers to each plot.
            if line in self.SUPPORTED_PLOT_TYPES:
                self._plots[line] = figure_number
                figure_number += 1
            else:
                print '[!] Unsupported plot type: ' + line

        print self._plots

    def _prepare_plot(self, plot_name):
        """ Orchestrates preparation for the specified plot type """

        print 'Preparing plot ' + plot_name + '...'

        if plot_name == 'compass_vs_gps_heading':
            self._prepare_compass_vs_gps_heading_plot()
        elif plot_name == 'compass_vs_gps_vs_nav_heading':
            self._prepare_compass_vs_gps_vs_nav_heading_plot()
        elif plot_name == 'control_heading_desired':
            self._prepare_control_heading_desired_plot()
        elif plot_name == 'control_steering_pwm_vs_commanded':
            self._prepare_control_steering_pwm_vs_commanded_plot()
        elif plot_name == 'control_xtrack_error':
            self._prepare_control_xtrack_error_plot()
        elif plot_name == 'control_xtrack_error_rate':
            self._prepare_control_xtrack_error_rate_plot()
        elif plot_name == 'control_xtrack_error_sum':
            self._prepare_control_xtrack_error_sum_plot()
        elif plot_name == 'desired_vs_actual_headings':
            self._prepare_desired_vs_actual_headings_plot()
        elif plot_name == 'gps_coordinates':
            self._prepare_gps_coordinates_plot()
        elif plot_name == 'gps_cumulative_dist':
            self._prepare_gps_cumulative_dist_plot()
        elif plot_name == 'gps_ground_course_deg':
            self._prepare_gps_ground_course_deg_plot()
        elif plot_name == 'gps_ground_speed_kt':
            self._prepare_gps_ground_speed_kt_plot()
        elif plot_name == 'gps_ground_speed_mph':
            self._prepare_gps_ground_speed_mph_plot()
        elif plot_name == 'gps_hdop':
            self._prepare_gps_hdop_plot()
        elif plot_name == 'gps_pdop':
            self._prepare_gps_pdop_plot()
        elif plot_name == 'gps_vdop':
            self._prepare_gps_vdop_plot()
        elif plot_name == 'gps_satcount':
            self._prepare_gps_satcount_plot()
        elif plot_name == 'heading_deg':
            self._prepare_heading_deg_plot()
        elif plot_name == 'main_loop_counter':
            self._prepare_main_loop_counter_plot()
        elif plot_name == 'meters_per_gps_update':
            self._prepare_meters_per_gps_update_plot()
        elif plot_name == 'mobility_motor_pwm':
            self._prepare_mobility_motor_pwm_plot()
        elif plot_name == 'nav_distance_to_waypoint':
            self._prepare_nav_distance_to_waypoint_plot()
        elif plot_name == 'nav_heading_deg':
            self._prepare_nav_heading_deg_plot()
        elif plot_name == 'nav_relative_bearings':
            self._prepare_nav_relative_bearings_plot()
        elif plot_name == 'nav_positions':
            self._prepare_nav_positions_plot()
        elif plot_name == 'nav_speed':
            self._prepare_nav_speed_plot()
        elif plot_name == 'odometer_ticks':
            self._prepare_odometer_ticks_plot()
        elif plot_name == 'odometer_ticks_per_iteration':
            self._prepare_odometer_ticks_per_iteration_plot()
        elif plot_name == 'odometer_timestamp':
            self._prepare_odometer_timestamp_plot()
        elif plot_name == 'pitch_deg':
            self._prepare_pitch_deg_plot()
        elif plot_name == 'position_estimates':
            self._prepare_position_estimates_plot()
        elif plot_name == 'roll_deg':
            self._prepare_roll_deg_plot()
        elif plot_name == 'simulate_nav_positions':
            self._prepare_simulate_nav_positions_plot()
        elif plot_name == 'simulate_xtrack_error':
            self._prepare_simulate_xtrack_error_plot()
        elif plot_name == 'status':
            self._prepare_status_plot()
        elif plot_name == 'ticks_per_gps_update':
            self._prepare_ticks_per_gps_update_plot()
        elif plot_name == 'ticks_per_meter_per_gps_update':
            self._prepare_ticks_per_meter_per_gps_update_plot()

    def _calculate_dist_between_gps_coords(self, coord1, coord2):
        """ Returns the distance between two GPS coordinates in meters
        using the Haversine formula
        """

        lat1 = coord1[0]
        long1 = coord1[1]
        lat2 = coord2[0]
        long2 = coord2[1]

        long1, lat1, long2, lat2 = map(radians, [long1, lat1, long2, lat2])

        dlong = long2 - long1
        dlat = lat2 - lat1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlong/2)**2
        c = 2 * asin(sqrt(a))

        meters = 6371000 * c

        return meters

    def _calculate_dist_per_interval(self, coordinates, interval_indexes):
        """ Returns a list of the distance traveled between each
        GPS update interval in meters
        """

        # coordinates is a list of tuples (latitude, longitude)
        distances = []

        prev_position = coordinates[interval_indexes[0]]

        for index in range(0, len(interval_indexes)):
            new_position = coordinates[interval_indexes[index]]
            distance = self._calculate_dist_between_gps_coords(prev_position, new_position)

            distances.append(distance)
            prev_position = new_position

        return distances

    def gcs_to_n_vector(self,lat,longitude):
        phi = radians(lat)
        theta = radians(longitude)
        x = cos(phi) * cos(theta)
        y = cos(phi) * sin(theta)
        z = sin(phi)
        return [x,y,z]

    def n_vector_to_gcs(self,nvector):
        xsquared = nvector[0]**2
        ysquared = nvector[1]**2
        d = sqrt((xsquared+ysquared))

        lat = atan2(nvector[2],d)
        glong = atan2(nvector[1],nvector[0])
        return (degrees(lat),degrees(glong))

    def calculate_using_vectors(self,latitude,longitude,distance,heading):
        r_earth = 6371e3
        north_vector = [0,0,1]
        initial_location = (latitude,longitude)

        initial_vector = self.gcs_to_n_vector(initial_location[0],initial_location[1])
        theta_bearing = radians(heading)

        #Get the vector east of initial location a
        de = np.cross(north_vector,initial_vector)
        #North vector at initial location
        dn = np.cross(initial_vector,de)

        #We need to compute the direction
        direction = (np.dot(dn,cos(theta_bearing))) + (np.dot(de,sin(theta_bearing)))

        xv = np.dot(initial_vector,cos(distance/r_earth))
        yv = np.dot(direction,sin(distance/r_earth))
        arrival_vector = xv + yv

        return self.n_vector_to_gcs(arrival_vector)

    def _calculate_gps_position(self, latitude, longitude, distance, heading):
        """ Calculates a new GPS coordinate given a starting position, distance
        traveled, and heading. Returns a tuple representing the calculated
        position in decimal degrees.

        Assumes the GPS coordinates are specified as decimal degrees; distance
        is specified in meters; and heading is specified in decimal degrees.
        The heading is assumed to have already been  corrected for magnetic
        declination (if needed).
        """

        earth_radius_m = 6371393.0

        lat_rad = radians(latitude)
        long_rad = radians(longitude)
        heading_rad = radians(heading)

        est_lat = asin( sin(lat_rad) * \
            cos(distance/earth_radius_m) + \
            cos(lat_rad) * \
            sin(distance/earth_radius_m) * \
            cos(heading_rad) )

        est_long = long_rad + \
            atan2( sin(heading_rad) * \
            sin(distance/earth_radius_m) * \
            cos(lat_rad), \
            cos(distance/earth_radius_m) -
            sin(lat_rad) *
            sin(est_lat) )

        new_lat = degrees(est_lat)
        new_long = degrees(est_long)

        return (new_lat, new_long)

    def _calculate_gps_position_float32(self, latitude, longitude, distance, heading):
        """ Calculates a new GPS coordinate given a starting position, distance
        traveled, and heading. Returns a tuple representing the calculated
        position in decimal degrees.

        Assumes the GPS coordinates are specified as decimal degrees; distance
        is specified in meters; and heading is specified in decimal degrees.
        The heading is assumed to have already been  corrected for magnetic
        declination (if needed).
        """

        earth_radius_m = 6371393.0

        lat_rad = radians(np.float32(latitude))
        long_rad = radians(np.float32(longitude))
        heading_rad = radians(np.float32(heading))

        est_lat = asin( sin(np.float32(lat_rad)) * \
            cos(np.float32(distance)/np.float32(earth_radius_m)) + \
            cos(np.float32(lat_rad)) * \
            sin(np.float32(distance)/np.float32(earth_radius_m)) * \
            cos(np.float32(heading_rad)) )

        est_long = np.float32(long_rad) + \
            atan2( sin(np.float32(heading_rad)) * \
            sin(np.float32(distance)/np.float32(earth_radius_m)) * \
            cos(np.float32(lat_rad)), \
            cos(np.float32(distance)/np.float32(earth_radius_m)) -
            sin(np.float32(lat_rad)) *
            sin(np.float32(est_lat)) )

        new_lat = degrees(np.float32(est_lat))
        new_long = degrees(np.float32(est_long))

        return (np.float32(new_lat), np.float32(new_long))

    def _calculate_mid_angle(self, heading_1, heading_2):
        """ Returns the angle that is halfway between the specified angles """

        if heading_1 > heading_2:
            temp = heading_1
            heading_1 = heading_2
            heading_2 = temp

        if heading_2 - heading_1 > 180.0:
            heading_2 = heading_2 - 360.0

        mid_angle = (heading_2 + heading_1) / 2.0

        if mid_angle < 0.0:
            mid_angle = mid_angle + 360.0

        return mid_angle

    def _calculate_mid_angle_float32(self, heading_1, heading_2):
        """ Returns the angle that is halfway between the specified angles """

        if heading_1 > heading_2:
            temp = heading_1
            heading_1 = heading_2
            heading_2 = temp

        if heading_2 - heading_1 > 180.0:
            heading_2 = heading_2 - np.float32(360.0)

        mid_angle = (heading_2 + heading_1) / np.float32(2.0)

        if mid_angle < 0.0:
            mid_angle = mid_angle + np.float32(360.0)

        return np.float32(mid_angle)

    def _calculate_relative_bearing(self, desired, actual):
        """ Calculates the relative bearing between a desired heading and
        and actual heading
        """

        diff = desired - actual

        if diff > 180.0:
            return (diff - 360.0)
        elif diff < -180.0:
            return (diff + 360.0)

        return diff

    def _calculate_ticks_per_interval(self, ticks, interval_indexes):
        """ Returns a list of the total number of ticks counted between each
        GPS update interval
        """

        delta_ticks = []

        prev_tick_count = ticks[interval_indexes[0]]

        for index in range(0, len(interval_indexes)):
            new_tick_count = ticks[interval_indexes[index]]
            tick_diff = new_tick_count - prev_tick_count

            delta_ticks.append(tick_diff)
            prev_tick_count = new_tick_count

        return delta_ticks

    def _find_indexes_for_nonzero_values(self, collection):
        """ Returns a list of indexes that correspond with the non-zero values
        in the specified collection
        """

        indexes = []

        for index in range(0, len(collection)):
            if collection[index] != 0:
                indexes.append(index)

        return indexes

    def _find_start_index(self, collection):
        """ Returns the index of the first non-zero occurrence in the specified
        collection
        """

        start_index = 0

        for index in range(0, len(collection)):
            if collection[index] != 0:
                start_index = index
                break;

        return start_index

    def _fill_zeroed_values(self, collection):
        """ Traverses the values in the collection and overwrites each zeroed
        value with the non-zero value that preceded it.

        This method was created to transform sparse data which are updated
        less frequently than others. For example, GPS-derived data are
        collected once per second, while pitch data are collected 40 times
        per second.
        """

        old_value = collection[0]

        for index in range(0, len(collection)):
            if collection[index] == 0:
                collection[index] = old_value
            elif collection[index] != old_value:
                old_value = collection[index]

    def _prepare_compass_vs_gps_heading_plot(self):
        """ Plots the compass headings and GPS headings on the same plot """

        compass_y_values = np.asarray(self._data.get_all('heading_deg'))
        gps_heading_y_values = np.asarray(self._data.get_all('gps_ground_course_deg'))
        x_values = np.arange(0, len(compass_y_values), 1)

        self._fill_zeroed_values(gps_heading_y_values)

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['compass_vs_gps_heading']) + \
            ' - Compass Heading vs GPS Heading')
        plt.xlabel('iteration')
        plt.ylabel('heading')
        plt.title('Compass Heading (Magnetic) and GPS Heading (True) in Degrees')
        plt.grid()
        plt.plot(x_values, compass_y_values)
        plt.plot(x_values, gps_heading_y_values)

    def _prepare_compass_vs_gps_vs_nav_heading_plot(self):
        """ Plots the compass headings, gps headings and navigation heading
        on the same plot
        """

        compass_headings = np.asarray(self._data.get_all('heading_deg'))
        gps_headings = np.asarray(self._data.get_all('gps_ground_course_deg'))
        nav_headings = np.asarray(self._data.get_all('nav_heading_deg'))
        iterations = np.asarray(self._data.get_all('main_loop_counter'))

        nav_gps_hdg = np.asarray(self._data.get_all('nav_gps_heading'))

        self._fill_zeroed_values(gps_headings)

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['compass_vs_gps_vs_nav_heading']) + \
            ' - Compass vs GPS vs Navigation Heading')
        plt.xlabel('iteration')
        plt.ylabel('heading (deg)')
        plt.title('Compass vs GPS vs Navigation Heading in Degrees')
        plt.grid()
        plt.plot(iterations, compass_headings)
        plt.plot(iterations, gps_headings)
        plt.plot(iterations, nav_headings)
        plt.plot(iterations, nav_gps_hdg)

    def _prepare_control_heading_desired_plot(self):
        """ Plots the target headings """

        target_headings = np.asarray(self._data.get_all('control_heading_desired'))
        iterations = np.asarray(self._data.get_all('main_loop_counter'))

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['control_heading_desired']) + \
            ' - Desired Heading')
        plt.title('Desired Heading')
        plt.xlabel('iteration')
        plt.ylabel('heading (deg)')
        plt.grid()
        plt.plot(iterations, target_headings)

    def _prepare_control_steering_pwm_vs_commanded_plot(self):
        """ Plots the control (calculated) steering PWM values versus the
        commanded steering PWM values on the same plot
        """

        control_pwm = np.asarray(self._data.get_all('control_steering_pwm'))
        commanded_pwm = np.asarray(self._data.get_all('mobility_steering_pwm'))
        iterations = np.asarray(self._data.get_all('main_loop_counter'))

        # Scale the pwm values by four to get microseconds
        for index in range(0, len(iterations)):
            control_pwm[index] = control_pwm[index] * 4
            commanded_pwm[index] = commanded_pwm[index] * 4

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['control_steering_pwm_vs_commanded']) + \
            ' - Control Steering vs Commanded Steering PWM')
        plt.title('Control Steering vs Commanded Steering PWM')
        plt.xlabel('iteration')
        plt.ylabel('pulse width (us)')
        plt.grid()
        control, = plt.plot(iterations, control_pwm, label='Control')
        commanded, = plt.plot(iterations, commanded_pwm, label='Commanded')
        plt.legend(handles=[control, commanded])

    def _prepare_control_xtrack_error_plot(self):
        """ Plots the cross track error """

        xtrack_errors = np.asarray(self._data.get_all('control_xtrack_error'))
        iterations = np.asarray(self._data.get_all('main_loop_counter'))

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['control_xtrack_error']) + \
            ' - Cross Track Error')
        plt.title('Cross-Track Error')
        plt.xlabel('iteration')
        plt.ylabel('error (deg)')
        plt.grid()
        plt.plot(iterations, xtrack_errors)

    def _prepare_control_xtrack_error_rate_plot(self):
        """ Plots the cross track error rate """

        xtrack_error_rates = np.asarray(self._data.get_all('control_xtrack_error_rate'))
        iterations = np.asarray(self._data.get_all('main_loop_counter'))

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['control_xtrack_error_rate']) + \
            ' - Cross Track Error Rate')
        plt.title('Cross Track Error Rate')
        plt.xlabel('iteration')
        plt.ylabel('error rate (deg/computation cycle)')
        plt.grid()
        plt.plot(iterations, xtrack_error_rates)

    def _prepare_control_xtrack_error_sum_plot(self):
        """ Plots the cross track error sum """

        xtrack_error_sums = np.array(self._data.get_all('control_xtrack_error_sum'))
        iterations = np.asarray(self._data.get_all('main_loop_counter'))

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['control_xtrack_error_sum']) +\
            ' - Cross Track Error Sum')
        plt.title('Cross Track Error Sum')
        plt.xlabel('iteration')
        plt.ylabel('sum (deg)')
        plt.grid()
        plt.plot(iterations, xtrack_error_sums)

    def _prepare_desired_vs_actual_headings_plot(self):
        """ Plots the desired headings and the nav headings on the same plot """

        target_headings = np.asarray(self._data.get_all('control_heading_desired'))
        nav_headings = np.asarray(self._data.get_all('nav_heading_deg'))
        compass_headings = np.asarray(self._data.get_all('heading_deg'))
        iterations = np.asarray(self._data.get_all('main_loop_counter'))

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['desired_vs_actual_headings']) + \
            ' - Desired vs Actual Headings')
        plt.title('Desired vs Actual Headings')
        plt.xlabel('iteration')
        plt.ylabel('heading (deg)')
        plt.grid()
        desired, = plt.plot(iterations, target_headings, label='Desired')
        actual, = plt.plot(iterations, nav_headings, label='Actual')
        compass, = plt.plot(iterations, compass_headings, label='Compass')
        plt.legend(handles = [desired, actual, compass])

    def _prepare_gps_coordinates_plot(self):
        """ Plots the GPS coordinates on square axes """

        latitudes = np.asarray(self._data.get_all('gps_latitude'))
        longitudes = np.asarray(self._data.get_all('gps_longitude'))

        # Find the first non-zero index, then discard all previous data points
        start_index = self._find_start_index(latitudes)
        latitudes = latitudes[start_index:]
        longitudes = longitudes[start_index:]

        # Fill in zeroed values to avoid plotting the (0 N, 0 W) coordinate
        # TODO: Just copy the non-zero values into a new list since we don't
        # need to keep track of which iteration the position was recorded for
        self._fill_zeroed_values(latitudes)
        self._fill_zeroed_values(longitudes)

        # Calculate total distance driven during this runningcoords = []
        coords = []
        for index in range(0, len(latitudes)):
            coords.append((latitudes[index], longitudes[index]))

        distances = self._calculate_dist_per_interval(coords, range(0, len(latitudes)))
        total_distance = sum(distances)

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['gps_coordinates']) + ' - GPS Coordinates')
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.title('GPS Coordinates' + \
            '\ntotal distance: ' + "{:.2f}".format(total_distance) + ' meters')
        plt.axis('equal')

        # Find the ranges of the latitude and longitudes to reshape the axes
        min_long = min(longitudes)
        max_long = max(longitudes)
        min_lat = min(latitudes)
        max_lat = max(latitudes)
        long_buffer = abs(max_long - min_long) * 0.1
        lat_buffer = abs(max_lat - min_lat) * 0.1

        # Reshape the axes dimensions to place whitespace padding on all sides
        plt.xlim([min(longitudes) - long_buffer, max(longitudes) + long_buffer])
        plt.ylim([min(latitudes) - lat_buffer, max(latitudes) + lat_buffer])

        plt.ticklabel_format(style='plain', useOffset=False)
        plt.grid()
        plt.scatter(longitudes, latitudes, color='g', marker='o')

    def _prepare_gps_ground_course_deg_plot(self):
        """ Plots the true course over the ground """

        y_values = np.asarray(self._data.get_all('gps_ground_course_deg'))
        x_values = np.asarray(self._data.get_all('main_loop_counter'))

        self._fill_zeroed_values(y_values)

        plt.figure(self._plots['gps_ground_course_deg'])
        plt.xlabel('iteration')
        plt.ylabel('course (deg)')
        plt.title('GPS Ground Course in Degrees (True)')
        plt.ylim([min(y_values), max(y_values) + 1])
        plt.grid()
        plt.plot(x_values, y_values)

    def _prepare_gps_ground_speed_kt_plot(self):
        """ Plots the ground speed in knots """

        y_values = np.asarray(self._data.get_all('gps_ground_speed_kt'))
        x_values = np.asarray(self._data.get_all('main_loop_counter'))

        self._fill_zeroed_values(y_values)

        plt.figure(self._plots['gps_ground_speed_kt'])
        plt.xlabel('iteration')
        plt.ylabel('speed (knots)')
        plt.title('GPS Ground Speed in Knots')
        plt.ylim([min(y_values), max(y_values) + 1])
        plt.grid()
        plt.plot(x_values, y_values)

    def _prepare_gps_ground_speed_mph_plot(self):
        """ Plots the ground speed in miles per hour """

        y_values = np.asarray(self._data.get_all('gps_ground_speed_kt'))
        x_values = np.arange(0, len(y_values), 1)

        # Convert knots to miles per hour
        y_values = y_values * 1.150779

        self._fill_zeroed_values(y_values)

        #plt.figure(self._plots['gps_ground_speed_mph'])
        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['gps_ground_speed_mph']) + \
            ' - Ground Speed in Miles per Hour')
        plt.xlabel('iteration')
        plt.ylabel('speed (mph)')
        plt.title('GPS Ground Speed in Miles per Hour' + \
            '\nmax: ' + "{:.2f}".format(max(y_values)) + \
            '  mean: ' + "{:.2f}".format(sum(y_values)/(len(y_values) + 0.0)))
        plt.ylim([min(y_values), max(y_values) + 1])
        plt.grid()
        plt.plot(x_values, y_values)

    def _prepare_gps_hdop_plot(self):
        """ Plots the horizontal dilution of precision """

        y_values = np.asarray(self._data.get_all('gps_hdop'))
        x_values = np.asarray(self._data.get_all('main_loop_counter'))

        self._fill_zeroed_values(y_values)

        plt.figure(self._plots['gps_hdop'])
        plt.xlabel('iteration')
        plt.ylabel('hdop')
        plt.title('GPS Horizontal Dilution of Precision')
        plt.ylim([min(y_values), max(y_values) + 1])
        plt.grid()
        plt.plot(x_values, y_values)

    def _prepare_gps_pdop_plot(self):
        """ Plots the position dilution of precision """

        y_values = np.asarray(self._data.get_all('gps_pdop'))
        x_values = np.asarray(self._data.get_all('main_loop_counter'))

        self._fill_zeroed_values(y_values)

        plt.figure(self._plots['gps_pdop'])
        plt.xlabel('iteration')
        plt.ylabel('pdop')
        plt.title('GPS Position Dilution of Precision')
        plt.ylim([min(y_values), max(y_values) + 1])
        plt.grid()
        plt.plot(x_values, y_values)

    def _prepare_gps_vdop_plot(self):
        """ Plots the vertical dilution of precision """

        y_values = np.asarray(self._data.get_all('gps_vdop'))
        x_values = np.asarray(self._data.get_all('main_loop_counter'))

        self._fill_zeroed_values(y_values)

        plt.figure(self._plots['gps_vdop'])
        plt.xlabel('iteration')
        plt.ylabel('vdop')
        plt.title('GPS Vertical Dilution of Precision')
        plt.ylim([min(y_values), max(y_values) + 1])
        plt.grid()
        plt.plot(x_values, y_values)

    def _prepare_gps_satcount_plot(self):
        """ Plots the GPS satellite count """

        y_values = np.asarray(self._data.get_all('gps_satcount'))
        x_values = np.asarray(self._data.get_all('main_loop_counter'))

        self._fill_zeroed_values(y_values)

        plt.figure(self._plots['gps_satcount'])
        plt.xlabel('iteration')
        plt.ylabel('number of satellites')
        plt.title('GPS Number of Satellites')
        plt.ylim([min(y_values), max(y_values) + 1])
        plt.grid()
        plt.plot(x_values, y_values)

    def _prepare_heading_deg_plot(self):
        """ Plots the magnetic heading """

        y_values = np.asarray(self._data.get_all('heading_deg'))
        x_values = np.asarray(self._data.get_all('main_loop_counter'))

        plt.figure(self._plots['heading_deg'])
        plt.xlabel('iteration')
        plt.ylabel('heading (deg)')
        plt.title('Compass Heading in Degrees (Magnetic)')
        plt.grid()
        plt.plot(x_values, y_values)

    def _prepare_main_loop_counter_plot(self):
        """ Plots the main loop counter """

        y_values = np.asarray(self._data.get_all('main_loop_counter'))
        x_values = np.arange(0, len(y_values), 1)

        plt.figure(self._plots['main_loop_counter'])
        plt.xlabel('iteration')
        plt.ylabel('counter')
        plt.title('Main Loop Counter\ntotal: ' + str(len(y_values)))
        plt.grid()
        plt.plot(x_values, y_values)

    def _prepare_meters_per_gps_update_plot(self):
        """ Plots the distance traveled during each GPS update interval """

        latitudes = np.asarray(self._data.get_all('gps_latitude'))
        longitudes = np.asarray(self._data.get_all('gps_longitude'))

        update_indexes = self._find_indexes_for_nonzero_values(latitudes)

        coords = []

        for index in range(0, len(latitudes)):
            coords.append((latitudes[index], longitudes[index]))

        distances = self._calculate_dist_per_interval(coords, update_indexes)

        # DEBUG
        print 'update indexes: ' + str(update_indexes)
        print 'distances: ' + str(distances)

        plt.figure(self._plots['meters_per_gps_update'])
        plt.xlabel('update iteration')
        plt.ylabel('distance (meters)')
        plt.title('Displacement Per GPS Update Interval')
        plt.grid()
        plt.plot(update_indexes, distances)

    def _prepare_mobility_motor_pwm_plot(self):
        """ Plots the commanded motor PWM values """

        motor_pwms = np.asarray(self._data.get_all('mobility_motor_pwm'))
        iterations = np.asarray(self._data.get_all('main_loop_counter'))

        # Multiply by 4 to get the value in microseconds
        for index in range(0, len(motor_pwms)):
            motor_pwms[index] = motor_pwms[index] * 4

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['mobility_motor_pwm']) + \
            ' - Mobility Motor PWM')
        plt.xlabel('iterations')
        plt.ylabel('pulse width (us)')
        plt.title('Mobility Motor PWM')
        plt.grid()
        plt.plot(iterations, motor_pwms)

    def _prepare_nav_distance_to_waypoint_plot(self):
        """ Plots the calculated distance to the current waypoint in meters """

        distances = np.asarray(self._data.get_all('nav_distance_to_waypt_m'))
        iterations = np.asarray(self._data.get_all('main_loop_counter'))

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['nav_distance_to_waypoint']) + \
            ' - Navigation Distances to Waypoints')
        plt.xlabel('iteration')
        plt.ylabel('distance (m)')
        plt.title('Navigation Distances to Waypoints in Meters')
        plt.grid()
        plt.plot(iterations, distances)

    def _prepare_nav_heading_deg_plot(self):
        """ Plots the heading used for navigation calculations """

        headings = np.asarray(self._data.get_all('nav_heading_deg'))
        iterations = np.asarray(self._data.get_all('main_loop_counter'))

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['nav_heading_deg']) + \
            ' - Navigation Headings')
        plt.xlabel('iteration')
        plt.ylabel('heading (deg)')
        plt.title('Navigation Headings')
        plt.grid()
        plt.plot(iterations, headings)

    def _prepare_nav_positions_plot(self):
        """ Plots the estimated robot positions between the measured GPS
        positions and the waypoints """

        meas_latitudes = np.asarray(self._data.get_all('gps_lat_ddeg'))
        meas_longitudes = np.asarray(self._data.get_all('gps_long_ddeg'))
        est_latitudes = np.asarray(self._data.get_all('nav_latitude'))
        est_longitudes = np.asarray(self._data.get_all('nav_longitude'))
        waypt_latitudes = np.asarray(self._data.get_all('nav_waypt_latitude'))
        waypt_longitudes = np.asarray(self._data.get_all('nav_waypt_longitude'))
        iterations = np.asarray(self._data.get_all('main_loop_counter'))

        latitude_indexes = []
        latitude_indexes = self._find_indexes_for_nonzero_values(meas_latitudes)

        latitudes = []
        longitudes = []

        for index in latitude_indexes:
            latitudes.append(meas_latitudes[index])
            longitudes.append(meas_longitudes[index])

        # Find the ranges of the latitude and longitudes to reshape the axes
        min_long = min(longitudes)
        max_long = max(longitudes)
        min_lat = min(latitudes)
        max_lat = max(latitudes)
        long_buffer = abs(max_long - min_long) * 0.1
        lat_buffer = abs(max_lat - min_lat) * 0.1

        # Reshape the axes dimensions to place whitespace padding on all sides
        plt.xlim([min(longitudes) - long_buffer, max(longitudes) + long_buffer])
        plt.ylim([min(latitudes) - lat_buffer, max(latitudes) + lat_buffer])

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['nav_positions']) + \
            ' - Navigation Positions')
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.title('Navigation Positions \nGPS-measured, Calculated, and Waypoints')
        plt.grid()
        plt.axis('equal')
        plt.ticklabel_format(style='plain', useOffset=False)

        #plt.plot(range(0, len(est_longitudes)), est_longitudes)

        plt.scatter(longitudes, latitudes, color='g', marker='o')
        plt.scatter(est_longitudes, est_latitudes, color='b', marker='x')
        plt.scatter(waypt_longitudes, waypt_latitudes, color='r', marker='*')

        # print 'type(meas_latitudes[0]): ' + str(type(meas_latitudes[0].astype(np.float32)))
        # print 'Waypoint Coords\n--------------\n'
        # for index in range(0, 30):
        #     print '(' + "{:.8f}".format(waypt_latitudes[index]) + ', ' + "{:.8f}".format(waypt_longitudes[index]) + ')'

    def _prepare_nav_relative_bearings_plot(self):
        """ Plots the relative bearing to the waypoints """

        rel_bearings = np.asarray(self._data.get_all('nav_rel_bearing_deg'))
        iterations = np.asarray(self._data.get_all('main_loop_counter'))
        nav_headings = np.asarray(self._data.get_all('nav_heading_deg'))

        # The code snippet below was used to fix the miscalculated
        # relative bearings. It's fixed in the robot code now. We don't need
        # this snippet anymore.
        # corrected_rel_bearings = []
        #
        # for index in range(0, len(iterations)):
        #     gps_coord_angle = rel_bearings[index] + nav_headings[index]
        #
        #     diff = gps_coord_angle - nav_headings[index]
        #
        #     if (diff > 180.0):
        #         corrected_rel_bearings.append(diff - 360.0)
        #     elif (diff < -180.0):
        #         corrected_rel_bearings.append(diff + 360.0)
        #     else:
        #         corrected_rel_bearings.append(diff)

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['nav_relative_bearings']) + \
            ' - Relative Bearings to Waypoints')
        plt.xlabel('iteration')
        plt.ylabel('bearing (deg)')
        plt.xlim([0, len(iterations) + len(iterations) * 0.05])
        plt.ylim([-190, 190])
        plt.title('Relative Bearings to Waypoints')
        plt.grid()
        plt.plot(iterations, rel_bearings)
        # plt.plot(iterations, corrected_rel_bearings)

    def _prepare_nav_speed_plot(self):
        """ Plots the odometer-based speed value """

        speed = np.asarray(self._data.get_all('nav_speed'))
        iterations = np.asarray(self._data.get_all('main_loop_counter'))

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['nav_speed']) + \
            ' - Navigation Speed (meters per sec)')
        plt.xlabel('iteration')
        plt.ylabel('speed (meters per sec)')
        plt.title('Navigation Speed in Meters Per Second')
        plt.grid()
        plt.plot(iterations, speed)

    def _prepare_odometer_ticks_plot(self):
        """ Plots the odometer ticks """

        y_values = np.asarray(self._data.get_all('odometer_ticks'))
        x_values = np.asarray(self._data.get_all('main_loop_counter'))

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['odometer_ticks']) + \
            ' - Cumulative Odometer Ticks')
        plt.xlabel('iteration')
        plt.ylabel('ticks')
        plt.title('Odometer Ticks' + \
            '\ntotal: ' + str(max(y_values)))
        plt.grid()
        plt.plot(x_values, y_values)

    def _prepare_odometer_timestamp_plot(self):
        """ Plots the timestamps for the odometer readings """

        timestamps = np.asarray(self._data.get_all('odometer_timestamp'))
        iterations = np.asarray(self._data.get_all('main_loop_counter'))

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['odometer_timestamp']) + \
            ' - Odometer Timestamps')
        plt.xlabel('iteration')
        plt.ylabel('timestamp (ticks)')
        plt.title('Odometer Timestamps in Loop-Time Ticks')
        plt.grid()
        plt.plot(iterations, timestamps)

    def _prepare_pitch_deg_plot(self):
        """ Plots the pitch """

        y_values = np.asarray(self._data.get_all('pitch_deg'))
        x_values = np.asarray(self._data.get_all('main_loop_counter'))

        plt.figure(self._plots['pitch_deg'])
        plt.xlabel('iteration')
        plt.ylabel('pitch (deg)')
        plt.title('Pitch in Degrees')
        plt.grid()
        plt.plot(x_values, y_values)

    def _prepare_position_estimates_plot(self):
        """ Plots the estimated positions between GPS updates """

        latitudes = np.asarray(self._data.get_all('gps_latitude'))
        longitudes = np.asarray(self._data.get_all('gps_longitude'))
        ticks = np.asarray(self._data.get_all('odometer_ticks'))
        headings = np.asarray(self._data.get_all('heading_deg'))
        headings_raw = np.asarray(self._data.get_all('heading_raw'))
        gps_headings = np.asarray(self._data.get_all('gps_ground_course_deg'))
        iterations = np.asarray(self._data.get_all('main_loop_counter'))
        declination_deg = 4.0

        ticks_per_meter = 7.6
        earth_radius_m = 6371393.0

        # Fill zeroed gps_headings
        self._fill_zeroed_values(gps_headings)

        # Find the indexes of the non-zero GPS coordinates
        update_indexes = self._find_indexes_for_nonzero_values(latitudes)

        prev_ticks = ticks[0]
        ticks_per_iter = []
        for iteration in range(0, len(iterations)):
            ticks_per_iter.append(ticks[iteration] - prev_ticks)

            prev_ticks = ticks[iteration]

        lats = []
        longs = []

        # Remove zeroed values
        for i in update_indexes:
            lats.append(latitudes[i])
            longs.append(longitudes[i])
            # print "coordinate: " + str(latitudes[i]) + ", " + str(longitudes[i])

        # Stores the lats and longs for Compass heading-based position estimates
        est_lats = []
        est_longs = []

        # Stores the lats and longs for GPS heading-based position estimations
        alt_est_lats = []
        alt_est_longs = []

        # Stores the lats and longs for position estimates based on compass
        # headings that have a fudge factor added to them
        fudge_est_lats = []
        fudge_est_longs = []

        nvector_est_lats = []
        nvector_est_longs = []

        print "update_indexes: " + str(update_indexes)

        #for index in range(0, len(lats)):
        for index in range(10, 13):
            known_lat = lats[index]
            known_long = longs[index]

            alt_known_lat = lats[index]
            alt_known_long = longs[index]

            fudge_known_lat = lats[index]
            fudge_known_long = longs[index]

            nvector_known_lat = lats[index]
            nvector_known_long = longs[index]

            print "iteration: " + str(update_indexes[index])
            print "gps coord: " + str(lats[index]) + ", " + str(longs[index])

            i = update_indexes[index]

            while (i < max(update_indexes) and i < update_indexes[index + 1]):
                distance = (ticks_per_iter[i] + 0.0) / ticks_per_meter

                # After looking at some of the data, I suspect that the compass
                # is a little bit off. Here we'll add in a fudge factor to the
                # raw heading. Remember: raw headings are integers in tenths of
                # degrees
                fudge = 150  # +15.0 degrees

                compass_heading = headings[i] + declination_deg
                gps_heading = gps_headings[i]
                # fudged_compass_heading = ((headings_raw[i] + 40 + fudge) % 3600) / 10.0

                # Forget the fudge. We'll try using the mid angle between the
                # compass and GPS headings
                fudged_compass_heading = self._calculate_mid_angle(compass_heading, gps_heading)

                print "compass: " + str(compass_heading) + "; " + "gps: " + str(gps_heading) + "; " + "mid: " + str(fudged_compass_heading)

                (fudge_lat, fudge_long) = self._calculate_gps_position(fudge_known_lat, fudge_known_long, distance, fudged_compass_heading)
                (est_lat, est_long) = self._calculate_gps_position(known_lat, known_long, distance, compass_heading)
                (alt_est_lat, alt_est_long) = self._calculate_gps_position(alt_known_lat, alt_known_long, distance, gps_heading)
                (nvector_est_lat, nvector_est_long) = self.calculate_using_vectors(nvector_known_lat, nvector_known_long, distance, compass_heading)

                if (distance > 0.0):
                    est_lats.append(est_lat)
                    est_longs.append(est_long)

                    alt_est_lats.append(alt_est_lat)
                    alt_est_longs.append(alt_est_long)

                    fudge_est_lats.append(fudge_lat)
                    fudge_est_longs.append(fudge_long)

                    nvector_est_lats.append(nvector_est_lat)
                    nvector_est_longs.append(nvector_est_long)

                    known_lat = est_lat
                    known_long = est_long

                    alt_known_lat = alt_est_lat
                    alt_known_long = alt_est_long

                    fudge_known_lat = fudge_lat
                    fudge_known_long = fudge_long

                    nvector_known_lat = nvector_est_lat
                    nvector_known_long = nvector_est_long

                print "ticks[" + str(i) + "]: " + str(ticks_per_iter[i]) + "; distance: " + str(distance) + "; heading: " + str(headings[i]) + \
                    "; est coord: " + str(est_lat) + ", " + str(est_long)

                i = i + 1

                if i >= max(update_indexes):
                    break

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['position_estimates']) + ' - GPS Position Estimates')
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.title('GPS Coordinates and Position Estimates' + \
            '\nusing Heading and Odometry')
        plt.axis('equal')

        # Find the ranges of the latitude and longitudes to reshape the axes
        min_long = min(longs)
        max_long = max(longs)
        min_lat = min(lats)
        max_lat = max(lats)
        long_buffer = abs(max_long - min_long) * 0.1
        lat_buffer = abs(max_lat - min_lat) * 0.1

        # Reshape the axes dimensions to place whitespace padding on all sides
        plt.xlim([min(longs) - long_buffer, max(longs) + long_buffer])
        plt.ylim([min(lats) - lat_buffer, max(lats) + lat_buffer])

        plt.ticklabel_format(style='plain', useOffset=False)
        plt.grid()
        plt.scatter(longs, lats, color='g', marker='o')
        plt.scatter(est_longs, est_lats, color='r', marker='x')
        plt.scatter(alt_est_longs, alt_est_lats, color='b', marker='x')
        plt.scatter(fudge_est_longs, fudge_est_lats, color='k', marker='x')
        #plt.scatter(nvector_est_longs, nvector_est_lats, color='c', marker='v')

        cep_lats = []
        cep_longs = []
        #for j in range(0, len(longs)):
        for j in range(10, 13):
            for deg in range(0, 360):
                (la, lo) = self._calculate_gps_position(lats[j], longs[j], 6.16, deg)
                cep_lats.append(la)
                cep_longs.append(lo)

        plt.scatter(cep_longs, cep_lats, color='m', marker='.')
        print "len(est_lats): " + str(len(est_lats))
        #print "est_lats: " + str(est_lats)
        #print "est_longs: " + str(est_longs)

    def _prepare_roll_deg_plot(self):
        """ Plots the roll """

        y_values = np.asarray(self._data.get_all('roll_deg'))
        x_values = np.asarray(self._data.get_all('main_loop_counter'))

        plt.figure(self._plots['roll_deg'])
        plt.xlabel('iteration')
        plt.ylabel('roll (deg)')
        plt.title('Roll in Degrees')
        plt.grid()
        plt.plot(x_values, y_values)

    def _prepare_simulate_nav_positions_plot(self):
        """ Plots a simulation of the navigation positions by splitting the
        degrees and minutes calculations
        """

        ticks_per_meter = np.float32(7.6)
        earth_radius_m = np.float32(6371393.0)
        declination_deg = np.float32(4.0)

        gpgga_sentences = np.asarray(self._data.get_all('sentence0'))
        latitudes = np.asarray(self._data.get_all('gps_latitude'))
        gps_headings = np.asarray(self._data.get_all('gps_ground_course_deg'))
        # gps_headings = np.asarray(self._data.get_all('gps_true_hdg_deg'))

        update_indexes = self._find_indexes_for_nonzero_values(latitudes)
        self._fill_zeroed_values(gps_headings)

        lat_deg = []
        lat_dec_deg = []
        long_deg = []
        long_dec_deg = []

        lat_dd_64 = []
        long_dd_64 = []

        for index in range(0, len(update_indexes)):
            fields = self._get_lat_long_fields(gpgga_sentences[update_indexes[index]])
            (lat, lat_dd, lon, lon_dd) = self._get_lat_long_values(fields)

            lat_dd_64_x = np.float(fields[0][2:9]) / 60.0
            long_dd_64_y = np.float(fields[2][3:10]) / 60.0

            lat_deg.append(lat)
            lat_dec_deg.append(lat_dd)
            long_deg.append(lon)
            long_dec_deg.append(lon_dd)

            lat_dd_64.append(lat_dd_64_x)
            long_dd_64.append(long_dd_64_y)

        headings = np.asarray(self._data.get_all('heading_deg'))
        ticks = np.asarray(self._data.get_all('odometer_ticks'))
        ticks_per_iter = self._calculate_ticks_per_iteration(ticks)
        iterations = np.asarray(self._data.get_all('main_loop_counter'))

        est_lats = []
        est_longs = []

        gps_lats = []
        gps_longs = []

        mix_lats = []
        mix_longs = []

        for index in range(0, len(lat_deg)):
            known_lat = lat_dec_deg[index]
            known_long = long_dec_deg[index]

            known_gps_lat = lat_dec_deg[index]
            known_gps_long = long_dec_deg[index]

            known_mix_lat = lat_dec_deg[index]
            known_mix_long = long_dec_deg[index]

            print "iteration: " + str(update_indexes[index])
            print "gps coord (dd):" + str(lat_dec_deg[index]) + ", " + str(long_dec_deg[index])

            i = update_indexes[index]

            while (i < max(update_indexes) and i < update_indexes[index + 1]):
                distance = np.float32(ticks_per_iter[i]) / np.float32(ticks_per_meter)

                compass_heading = np.float32(headings[i]) + np.float32(declination_deg)
                gps_heading = np.float32(gps_headings[i])

                mix_hdg = self._calculate_mid_angle_float32(compass_heading, gps_heading)

                # print "compass: " + str(compass_heading) + "; gps: " + str(gps_heading) + "; mix:" + str(mix_hdg)

                (est_lat, est_long) = self._calculate_gps_position_float32(known_lat, known_long, distance, compass_heading)
                (gps_lat, gps_long) = self._calculate_gps_position_float32(known_gps_lat, known_gps_long, distance, gps_heading)
                (mix_lat, mix_long) = self._calculate_gps_position_float32(known_mix_lat, known_mix_long, distance, mix_hdg)

                if (distance > 0.0):
                    est_lats.append(est_lat)
                    est_longs.append(est_long)

                    gps_lats.append(gps_lat)
                    gps_longs.append(gps_long)

                    mix_lats.append(mix_lat)
                    mix_longs.append(mix_long)

                    known_lat = est_lat
                    known_long = est_long

                    known_gps_lat = gps_lat
                    known_gps_long = gps_long

                    known_mix_lat = mix_lat
                    known_mix_long = mix_long

                i = i + 1

                if i >= max(update_indexes):
                    break;

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['simulate_nav_positions']) + \
            ' - Simulated Nav Position (float32)')
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.title('GPS Coordinates and Position Estimates' + \
            '\noff the Coast of Africa!')
        plt.axis('equal')

        plt.ticklabel_format(style='plain', useOffset=False)
        plt.grid()
        plt.scatter(long_dec_deg, lat_dec_deg, color='g', marker='o')
        #plt.scatter(long_dd_64, lat_dd_64, color='b', marker='^')
        plt.scatter(est_longs, est_lats, color='r', marker='x')
        plt.scatter(gps_longs, gps_lats, color='b', marker='x')
        plt.scatter(mix_longs, mix_lats, color='k', marker='x')

    def _prepare_simulate_xtrack_error_plot(self):
        """ Simulates the Cross-Track error """

        desired_headings = np.asarray(self._data.get_all('control_heading_desired'))
        nav_headings = np.asarray(self._data.get_all('nav_heading_deg'))
        iterations = np.asarray(self._data.get_all('main_loop_counter'))

        xtrack_errors = []

        for index in range(0, len(iterations)):
            xtrack_errors.append(self._calculate_relative_bearing(desired_headings[index], nav_headings[index]))

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['simulate_xtrack_error']) + \
            ' - Simulated Cross-Track Error')
        plt.xlabel('iterations')
        plt.ylabel('error (deg)')
        plt.title('Simulated Cross-Track Error')
        plt.grid()
        desired, = plt.plot(iterations, desired_headings, label='Desired')
        actual, = plt.plot(iterations, nav_headings, label='Actual')
        error, = plt.plot(iterations, xtrack_errors, label='Error')
        plt.legend(handles = [desired, actual, error])

    def _get_lat_long_fields(self, sentence):
        """ Returns the latitude, latitude hemisphere, longitude, and
        longitude hemisphere substrings from the specified sentence
        """

        tokens = sentence.split(',')

        return (tokens[2:6])

    def _get_lat_long_values(self, tokens):
        """ Returns the latitude degrees, latitude decimal degrees,
        longitude degrees, and longitude decimal degrees for the specified
        tokens. Assumes the tokens are arranged as:
        (ddmm.mmmm, N/S, dddmm.mmmm, E/W)
        """

        lat_deg = np.float32(tokens[0][0:2])
        lat_dec_deg = np.float32(tokens[0][2:9]) / np.float32(60)
        if (tokens[1] == 'S'):
            lat_deg = lat_deg * -1

        long_deg = np.float32(tokens[2][0:3])
        long_dec_deg = np.float32(tokens[2][3:10]) / np.float32(60)
        if (tokens[3] == 'W'):
            long_deg = long_deg * -1
            long_dec_deg = long_dec_deg * -1

        return (lat_deg, lat_dec_deg, long_deg, long_dec_deg)

    def _calculate_ticks_per_iteration(self, ticks):
        prev_ticks = ticks[0]
        ticks_per_iter = []

        for iteration in range(0, len(ticks)):
            ticks_per_iter.append(ticks[iteration] - prev_ticks)

            prev_ticks = ticks[iteration]

        return ticks_per_iter

    def _prepare_status_plot(self):
        """ Plots the status bits """

        status_values = self._data.get_all('status')
        iterations = self._data.get_all('main_loop_counter')

        y_values = []
        x_values = []

        # Calculate how often the main loop was running late
        main_loop_late_count = 0

        for iteration in iterations:
            #for value in status_values:
            value = status_values[iteration]

            for bit in range(0, 32):
                if value & (1 << bit):
                    y_values.append(bit)
                    x_values.append(iteration)

                    if bit == 12:
                        main_loop_late_count += 1

        x = np.asarray(x_values)
        y = np.asarray(y_values)

        late_percentage = (main_loop_late_count * 100.0) / len(status_values)

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['status']) + ' - Status Bits')
        plt.xlabel('iteration')
        plt.ylabel('status')
        plt.title('Status Bits (ON only)\n[late ' + str(main_loop_late_count) + \
                ' out of ' + str(len(status_values)) + \
                ' iterations (' + "{:.2f}".format(late_percentage) + '%)]')
        plt.xlim([min(iterations) - 10, max(iterations) + 10])
        plt.ylim([-1, 33])
        plt.grid()
        plt.scatter(x, y, color='r', marker='o')

    def _prepare_ticks_per_gps_update_plot(self):
        """ Plots the number of ticks that occurred since the previous
        GPS update
        """

        latitudes = np.asarray(self._data.get_all('gps_latitude'))
        longitudes = np.asarray(self._data.get_all('gps_longitude'))
        ticks = np.asarray(self._data.get_all('odometer_ticks'))

        update_indexes = self._find_indexes_for_nonzero_values(latitudes)
        tick_deltas = self._calculate_ticks_per_interval(ticks, update_indexes)

        # DEBUG
        print 'update indexes: ' + str(update_indexes)
        print 'delta ticks: ' + str(tick_deltas)

        plt.figure(self._plots['ticks_per_gps_update'])
        plt.xlabel('update iteration')
        plt.ylabel('ticks')
        plt.title('Ticks Per GPS Update Interval\nAverage: ' + \
                "{:.2f}".format(sum(tick_deltas)/(len(tick_deltas) + 0.0)))
        plt.grid()
        plt.plot(update_indexes, tick_deltas)

    def _prepare_odometer_ticks_per_iteration_plot(self):
        """ Plots the number of ticks that were counted during each iteration"""

        ticks = np.asarray(self._data.get_all('odometer_ticks'))
        iterations = np.asarray(self._data.get_all('main_loop_counter'))

        prev_ticks = ticks[0]
        ticks_per_iter = []
        for iteration in range(0, len(iterations)):
            ticks_per_iter.append(ticks[iteration] - prev_ticks)

            prev_ticks = ticks[iteration]

        max_ticks = max(ticks_per_iter)
        avg_ticks_per_iter = sum(ticks_per_iter) / (len(ticks) + 0.0)
        stats = {}

        for i in range(0, max_ticks + 1):
            stats[i] = 0

        for tick_count in ticks_per_iter:
            stats[tick_count] = stats[tick_count] + 1

        title_stats = ''
        for tick_count in sorted(stats.keys()):
            title_stats += str(tick_count) + ': ' + str(stats[tick_count]) + \
            ' (' + "{:.2f}".format(stats[tick_count] * 100.0/len(ticks)) + '%) '

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['odometer_ticks_per_iteration']) + \
            ' - Odometer Ticks Per Iteration')
        plt.xlabel('iteration')
        plt.ylabel('ticks')
        plt.title('Ticks Per Iteration' + '\n[max: ' + str(max_ticks) + \
            '; mean: ' + "{:.2f}".format(avg_ticks_per_iter) + ']' + \
            '\n' + title_stats)
        plt.grid()
        plt.plot(iterations, ticks_per_iter)

    def _prepare_ticks_per_meter_per_gps_update_plot(self):
        """ Plots the number of ticks that were counted during each
        GPS update interval
        """

        latitudes = np.asarray(self._data.get_all('gps_latitude'))
        longitudes = np.asarray(self._data.get_all('gps_longitude'))
        ticks = np.asarray(self._data.get_all('odometer_ticks'))

        update_indexes = self._find_indexes_for_nonzero_values(latitudes)

        coords = []

        for index in range(0, len(latitudes)):
            coords.append((latitudes[index], longitudes[index]))

        distances = self._calculate_dist_per_interval(coords, update_indexes)
        tick_deltas = self._calculate_ticks_per_interval(ticks, update_indexes)

        ticks_per_meter = []

        for index in range(0, len(update_indexes)):
            interval_ticks = tick_deltas[index]
            interval_meters = distances[index]

            if interval_meters != 0:
                interval_ticks_per_meter = interval_ticks/interval_meters
            else:
                interval_ticks_per_meter = 0.0

            ticks_per_meter.append(interval_ticks_per_meter)

        # Calculate the average ticks per interval using only non-zero ticks
        non_zero_ticks_per_meter = []

        for index in range(0, len(ticks_per_meter)):
            tpm = ticks_per_meter[index]
            if tpm > 0.0:
                non_zero_ticks_per_meter.append(tpm)

        avg_ticks_per_meter = sum(non_zero_ticks_per_meter)/len(non_zero_ticks_per_meter)
        total_distance = sum(distances)
        total_ticks = np.amax(ticks)

        plt.figure().canvas.set_window_title('Figure ' + \
            str(self._plots['ticks_per_meter_per_gps_update']) + \
            ' - Ticks Per Meter Per GPS Update Interval')
        plt.xlabel('interval index')
        plt.ylabel('ticks per meter')
        plt.title('Ticks Per Meter Per GPS Update Interval\nInterval Average: ' + \
                "{:.4f}".format(avg_ticks_per_meter) + ' ticks/meter\n' + \
                'Course Ticks / Course Distance: ' + \
                "{:.4f}".format(total_ticks/total_distance) + ' ticks/meter')
        plt.grid()
        plt.plot(update_indexes, ticks_per_meter)
