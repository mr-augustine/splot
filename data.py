import struct

from statevars import Statevars

class Data:
    """ Class responsible for reading and storing state variable data from the
    specified data file according to the specified state variable definition
    """

    _data_list = []

    def __init__(self, filename, statevars):
        datafile = open(filename, 'rb')

        statevars_index = 0
        num_fields = len(statevars.get_varlist())

        while True:
            next_dataframe = self._read_next_dataframe(datafile, statevars.get_varlist())
            if len(next_dataframe) == num_fields:
                self._data_list.append(next_dataframe)
            elif len(next_dataframe) == 0:
                print '[*] Finished!'
                print '    Read ' + str(statevars_index) + ' dataframes'
                break
            # TODO Actually create this error detection capability
            elif len(next_dataframe) != len(statevars.get_varlist()):
                print '[!] There was an error at dataframe + ' + str(statevars_index)
                break

            statevars_index += 1

        datafile.close()

    def _convert_bytes(self, bytes_to_convert, field_def):
        """ Returns the specified bytes as a value interpreted in accordance
        with the field definition
        """

        datatype = Statevars.get_field_datatype(field_def)
        format_string = 'c'

        if datatype == 'uint32_t':
            format_string = '<I'
        elif datatype == 'uint16_t':
            format_string = '<H'
        elif datatype == 'uint8_t':
            format_string = '<B'
        elif datatype == 'int32_t':
            format_string = '<i'
        elif datatype == 'int16_t':
            format_string = '<h'
        elif datatype == 'int8_t':
            format_string = '<b'
        elif datatype == 'float' or datatype == 'double':
            format_string = '<f'

        return struct.unpack_from(format_string, bytes_to_convert)[0]

    def get_all(self, field_name):
        """ Returns a list of all recorded values for the specified field """

        data = []

        for frame in self._data_list:
            data.append(frame[field_name])

        return data

    def get_data(self):
        """ Returns the data list of data frames """
        return self._data_list
    
    def _read_next_dataframe(self, datafile, statevars_def):
        """ Returns a dictionary that contains the field names and associated
        values for the next data frame in the data file
        """

        statevars_index = 0
        data_frame = {}

        for field in statevars_def:
            field_size = Statevars.get_datatype_size(Statevars.get_field_datatype(field))
            field_count = Statevars.get_field_count(field)
            field_name = Statevars.get_field_name(field)

            buff = datafile.read(field_size * field_count)

            if not buff:
                data_frame.clear()
                return data_frame

            # Little Endian conversion is handled in the convert_bytes() function
            if field_count == 1:
                converted_buff = self._convert_bytes(buff, field)
            # TODO test this code section with multiple, multi-byte values (e.g., int32[3])
            elif field_count > 1:
                # Handle arrays of multi-byte elements
                if field_size > 1:
                    new_buff = ()
                    for index in range(0, field_count):
                        new_buff.append(self._convert_bytes(buff[index], field))
                    converted_buff = new_buff
                # Handle arrays of single-byte elements (e.g., char[16])
                # TODO update this code to handle multiple, single-byte values
                # other than char (e.g., int8_t and uint8_t)
                elif field_size == 1:
                    converted_buff = str(buff)

            data_frame[field_name] = converted_buff

        return data_frame
