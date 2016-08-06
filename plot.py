import matplotlib.pyplot as plt
import numpy as np
import sys

from data import Data

class Plot:
    'Class responsible for plotting the data'

    SUPPORTED_PLOT_TYPES = ['main_loop_counter',
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
                            'odometer_ticks',
                            'pitch_deg',
                            'roll_deg',
                            'status',
                            'compass_vs_gps_heading',
                            'ticks_per_gps_update']

    _plots = {}
    _data = None

    def __init__(self, filename, data):
        self.__parse_plot_list_file(filename)

        self._data = data

        for plot_name in self._plots.keys():
            self.__prepare_plot(plot_name)
            #self._plots[plot_name] = self.__prepare_plot(plot_name)

        #for plot_name in self._plots.keys():
            #self._plots[plot_name].show()
            #print type(self._plots[plot_name])

        plt.show()

    def __parse_plot_list_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()

        figure_number = 1
        for line in lines:
            line = str.strip(line)

            if line == '' or line[0] == '#':
                continue

            if line in self.SUPPORTED_PLOT_TYPES:
                self._plots[line] = figure_number
                #self._plots[line] = None
                figure_number += 1
            else:
                print '[!] Unsupported plot type: ' + line

        print self._plots

    def __prepare_plot(self, plot_name):
        print 'Preparing plot ' + plot_name + '...'

        if plot_name == 'main_loop_counter':
            self.__prepare_main_loop_counter_plot()
        elif plot_name == 'odometer_ticks':
            self.__prepare_odometer_ticks_plot()
        elif plot_name == 'gps_coordinates':
            self.__prepare_gps_coordinates_plot()
        elif plot_name == 'gps_cumulative_dist':
            self.__prepare_gps_cumulative_dist_plot()
        elif plot_name == 'gps_ground_course_deg':
            self.__prepare_gps_ground_course_deg_plot()
        elif plot_name == 'gps_ground_speed_kt':
            self.__prepare_gps_ground_speed_kt()
        elif plot_name == 'gps_ground_speed_mph':
            self.__prepare_gps_ground_speed_mph()
        elif plot_name == 'gps_hdop':
            self.__prepare_gps_hdop_plot()
        elif plot_name == 'gps_pdop':
            self.__prepare_gps_pdop_plot()
        elif plot_name == 'gps_vdop':
            self.__prepare_gps_vdop_plot()
        elif plot_name == 'gps_satcount':
            self.__prepare_gps_satcount_plot()
        elif plot_name == 'heading_deg':
            self.__prepare_heading_deg_plot()
        elif plot_name == 'pitch_deg':
            self.__prepare_pitch_deg_plot()
        elif plot_name == 'roll_deg':
            self.__prepare_roll_deg_plot()
        elif plot_name == 'status':
            self.__prepare_status_plot()
        elif plot_name == 'ticks_per_gps_update':
            self.__prepare_ticks_per_gps_update_plot()
        elif plot_name == 'compass_vs_gps_heading':
            self.__prepare_compass_vs_gps_heading_plot()

    def __fill_zeroed_values(self, collection):
        old_value = collection[0]

        for index in range(0, len(collection)):
            if collection[index] == 0:
                collection[index] = old_value
            elif collection[index] != old_value:
                old_value = collection[index]

    def __prepare_gps_coordinates_plot(self):
        latitudes = np.asarray(self._data.get_all('gps_latitude'))
        longitudes = np.asarray(self._data.get_all('gps_longitude'))
        iterations = np.asarray(self._data.get_all('main_loop_counter'))

        start_index = 0
        for index in range(0, len(latitudes)):
            if latitudes[index] != 0:
                start_index = index
                break;

        latitudes = latitudes[start_index:]
        longitudes = longitudes[start_index:]

        # Fill in zeroed values
        self.__fill_zeroed_values(latitudes)
        self.__fill_zeroed_values(longitudes)

        print "start_index " + str(start_index)
        print "latitudes"
        print latitudes
        print "longitudes"
        print longitudes

        plt.figure(self._plots['gps_coordinates'])
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.title('GPS Coordinates')
        plt.axis('equal')

        min_long = min(longitudes)
        max_long = max(longitudes)
        min_lat = min(latitudes)
        max_lat = max(latitudes)
        long_buffer = abs(max_long - min_long) * 0.1
        lat_buffer = abs(max_lat - min_lat) * 0.1

        plt.xlim([min(longitudes) - long_buffer, max(longitudes) + long_buffer])
        plt.ylim([min(latitudes) - lat_buffer, max(latitudes) + lat_buffer])

        plt.ticklabel_format(style='plain', useOffset=False)
        plt.scatter(longitudes, latitudes, color='g', marker='o')

    def __prepare_gps_cumulative_dist_plot(self):
        latitudes = np.asarray(self._data.get_all('gps_latitude'))
        longitudes = np.asarray(self._data.get_all('gps_longitude'))
        iterations = np.asarray(self._data.get_all('main_loop_counter'))

        earth_radus_m = 6371000
        lat_rad = math.radians(latitudes)
        long_rad = math.radians(longitudes)

        # Continue here

    def __prepare_gps_ground_course_deg_plot(self):
        y_values = np.asarray(self._data.get_all('gps_ground_course_deg'))
        x_values = np.asarray(self._data.get_all('main_loop_counter'))

        # Fill in zeroed values
        self.__fill_zeroed_values(y_values)

        plt.figure(self._plots['gps_ground_course_deg'])
        plt.xlabel('iteration')
        plt.ylabel('course (deg)')
        plt.title('GPS Ground Course in Degrees')
        plt.ylim([min(y_values), max(y_values) + 1])
        plt.plot(x_values, y_values)

    def __prepare_gps_ground_speed_kt(self):
        y_values = np.asarray(self._data.get_all('gps_ground_speed_kt'))
        x_values = np.asarray(self._data.get_all('main_loop_counter'))

        # Fill in zeroed values
        self.__fill_zeroed_values(y_values)

        plt.figure(self._plots['gps_ground_speed_kt'])
        plt.xlabel('iteration')
        plt.ylabel('speed (knots)')
        plt.title('GPS Ground Speed in Knots')
        plt.ylim([min(y_values), max(y_values) + 1])
        plt.plot(x_values, y_values)

    def __prepare_gps_ground_speed_mph(self):
        y_values = np.asarray(self._data.get_all('gps_ground_speed_kt'))
        x_values = np.arange(0, len(y_values), 1)

        # Convert knots to miles per hour
        y_values = y_values * 1.150779

        self.__fill_zeroed_values(y_values)

        plt.figure(self._plots['gps_ground_speed_mph'])
        plt.xlabel('iteration')
        plt.ylabel('speed (mph)')
        plt.title('GPS Ground Speed in Miles per Hour')
        plt.ylim([min(y_values), max(y_values) + 1])
        plt.plot(x_values, y_values)

    def __prepare_gps_hdop_plot(self):
        y_values = np.asarray(self._data.get_all('gps_hdop'))
        x_values = np.asarray(self._data.get_all('main_loop_counter'))

        self.__fill_zeroed_values(y_values)

        plt.figure(self._plots['gps_hdop'])
        plt.xlabel('iteration')
        plt.ylabel('hdop')
        plt.title('GPS Horizontal Dilution of Precision')
        plt.ylim([min(y_values), max(y_values) + 1])
        plt.grid()
        plt.plot(x_values, y_values)

    def __prepare_gps_pdop_plot(self):
        y_values = np.asarray(self._data.get_all('gps_pdop'))
        x_values = np.asarray(self._data.get_all('main_loop_counter'))

        self.__fill_zeroed_values(y_values)

        plt.figure(self._plots['gps_pdop'])
        plt.xlabel('iteration')
        plt.ylabel('pdop')
        plt.title('GPS Position Dilution of Precision')
        plt.ylim([min(y_values), max(y_values) + 1])
        plt.grid()
        plt.plot(x_values, y_values)

    def __prepare_gps_vdop_plot(self):
        y_values = np.asarray(self._data.get_all('gps_vdop'))
        x_values = np.asarray(self._data.get_all('main_loop_counter'))

        self.__fill_zeroed_values(y_values)

        plt.figure(self._plots['gps_vdop'])
        plt.xlabel('iteration')
        plt.ylabel('vdop')
        plt.title('GPS Vertical Dilution of Precision')
        plt.ylim([min(y_values), max(y_values) + 1])
        plt.grid()
        plt.plot(x_values, y_values)

    def __prepare_gps_satcount_plot(self):
        y_values = np.asarray(self._data.get_all('gps_satcount'))
        x_values = np.arange(0, len(y_values), 1)

        # Fill in zeroed values
        old_value = y_values[0]
        for index in range(0, len(y_values)):
            if y_values[index] == 0:
                y_values[index] = old_value
            elif y_values[index] != old_value:
                old_value = y_values[index]

        plt.figure(self._plots['gps_satcount'])
        plt.xlabel('iteration')
        plt.ylabel('number of satellites')
        plt.title('GPS Number of Satellites')
        plt.ylim([min(y_values), max(y_values) + 1])
        plt.plot(x_values, y_values)

    def __prepare_heading_deg_plot(self):
        y_values = np.asarray(self._data.get_all('heading_deg'))
        x_values = np.arange(0, len(y_values), 1)

        plt.figure(self._plots['heading_deg'])
        plt.xlabel('iteration')
        plt.ylabel('heading (deg)')
        plt.title('Compass Heading in Degrees')
        plt.plot(x_values, y_values)

    def __prepare_main_loop_counter_plot(self):
        y_values = np.asarray(self._data.get_all('main_loop_counter'))
        x_values = np.arange(0, len(y_values), 1)

        plt.figure(self._plots['main_loop_counter'])
        plt.xlabel('iteration')
        plt.ylabel('counter')
        plt.title('Main Loop Counter\ntotal: ' + str(len(y_values)))
        plt.plot(x_values, y_values)

    def __prepare_odometer_ticks_plot(self):
        y_values = np.asarray(self._data.get_all('odometer_ticks'))
        x_values = np.arange(0, len(y_values), 1)

        plt.figure(self._plots['odometer_ticks'])
        plt.xlabel('iteration')
        plt.ylabel('ticks')
        plt.title('Odometer Ticks')
        plt.grid()
        plt.plot(x_values, y_values)

    def __prepare_pitch_deg_plot(self):
        y_values = np.asarray(self._data.get_all('pitch_deg'))
        x_values = np.arange(0, len(y_values), 1)

        plt.figure(self._plots['pitch_deg'])
        plt.xlabel('iteration')
        plt.ylabel('pitch (deg)')
        plt.title('Pitch in Degrees')
        plt.grid()
        plt.plot(x_values, y_values)

    def __prepare_roll_deg_plot(self):
        y_values = np.asarray(self._data.get_all('roll_deg'))
        x_values = np.arange(0, len(y_values), 1)

        plt.figure(self._plots['roll_deg'])
        plt.xlabel('iteration')
        plt.ylabel('roll (deg)')
        plt.title('Roll in Degrees')
        plt.grid()
        plt.plot(x_values, y_values)

    def __prepare_status_plot(self):
        status_values = self._data.get_all('status')
        y_values = []
        x_values = []

        print 'sizeof(values): ' + str(sys.getsizeof(status_values[0]))
        print type(status_values[0])
        print sys.getsizeof(type(status_values[0])())

        main_loop_late_count = 0

        for iteration in range(0, len(status_values)):
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

        plt.figure(self._plots['status'])
        plt.xlabel('iteration')
        plt.ylabel('status')
        plt.title('Status Bits (ON only)\n[late ' + str(main_loop_late_count) + \
                ' out of ' + str(len(status_values)) + \
                ' iterations (' + "{:.2f}".format(late_percentage) + '%)]')
        plt.xlim([min(x) - (0.1 * max(x)), max(x) + 0.1 * max(x)])
        plt.grid()
        plt.scatter(x, y, color='r', marker='o')

    def __prepare_compass_vs_gps_heading_plot(self):
        compass_y_values = np.asarray(self._data.get_all('heading_deg'))
        gps_heading_y_values = np.asarray(self._data.get_all('gps_ground_course_deg'))
        x_values = np.arange(0, len(compass_y_values), 1)

        # Fill in zeroed values
        self.__fill_zeroed_values(gps_heading_y_values)

        plt.figure(self._plots['compass_vs_gps_heading'])
        plt.xlabel('iteration')
        plt.ylabel('heading')
        plt.title('Compass and GPS Headings in Degrees')
        plt.grid()
        plt.plot(x_values, compass_y_values)
        plt.plot(x_values, gps_heading_y_values)

    def __prepare_ticks_per_gps_update_plot(self):
        latitudes = np.asarray(self._data.get_all('gps_latitude'))
        longitudes = np.asarray(self._data.get_all('gps_longitude'))
        ticks = np.asarray(self._data.get_all('odometer_ticks'))

        start_index = self.__find_start_index(latitudes)

        print 'latitudes start index: ' + str(self.__find_start_index(latitudes))
        print 'longitudes start index: ' + str(self.__find_start_index(longitudes))

        update_indexes = self.__find_indexes_for_nonzero_values(latitudes)
        print 'update indexes: ' + str(update_indexes)

        tick_deltas = self.__calculate_ticks_per_interval(ticks, update_indexes)
        print 'delta ticks: ' + str(tick_deltas)

        plt.figure(self._plots['ticks_per_gps_update'])
        plt.xlabel('update iteration')
        plt.ylabel('ticks')
        plt.title('Ticks Per GPS Update Inteval\nAverage: ' + "{:.2f}".format(sum(tick_deltas)/(len(tick_deltas) + 0.0)))
        plt.grid()
        plt.plot(update_indexes, tick_deltas)

    def __calculate_ticks_per_interval(self, ticks, interval_indexes):
        delta_ticks = []

        prev_tick_count = ticks[interval_indexes[0]]

        for index in range(0, len(interval_indexes)):
            new_tick_count = ticks[interval_indexes[index]]
            tick_diff = new_tick_count - prev_tick_count

            delta_ticks.append(tick_diff)
            prev_tick_count = new_tick_count

        return delta_ticks

    def __find_indexes_for_nonzero_values(self, collection):
        indexes = []

        for index in range(0, len(collection)):
            if collection[index] != 0:
                indexes.append(index)

        return indexes

    def __find_start_index(self, collection):
        start_index = 0

        for index in range(0, len(collection)):
            if collection[index] != 0:
                start_index = index
                break;

        return start_index
