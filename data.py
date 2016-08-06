import struct

from statevars import Statevars

class Data:
    'Stores the data read from the datafile'

    _data_list = []

    def __init__(self, filename, statevars):
        datafile = open(filename, 'rb')

        statevars_index = 0
        num_fields = len(statevars.get_varlist())

        while True:
            next_dataframe = self.__read_next_dataframe(datafile, statevars.get_varlist())
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


    def __read_next_dataframe(self, datafile, statevars_def):
        statevars_index = 0
        out_hash = {}

        for field in statevars_def:
            field_size = Statevars.get_datatype_size(Statevars.get_field_datatype(field))
            field_count = Statevars.get_field_count(field)
            field_name = Statevars.get_field_name(field)

            buff = datafile.read(field_size * field_count)

            if not buff:
                out_hash.clear()
                return out_hash

            # Little Endian conversion is handled in the convert_bytes() function
            if field_count == 1:
                converted_buff = self.__convert_bytes(buff, field)
            # TODO test this code section with multiple, multi-byte values
            elif field_count > 1:
                if field_size > 1:
                    new_buff = ()
                    for index in range(0, field_count):
                        new_buff.append(self.__convert_bytes(buff[index], field))
                    converted_buff = new_buff
                elif field_size == 1:
                    converted_buff = str(buff)

            out_hash[field_name] = converted_buff

            # print 'field = ' + str(field)
            # print 'field size = ' + str(field_size)
            # print 'field count = ' + str(field_count)
            # print 'field value = ' + str(out_hash[field_name]) + '\n'

        return out_hash
            #print Statevars.get_field_name(field) + ": " + str(struct.unpack_from("<I", buff)[0])

    def __convert_bytes(self, bytes_to_convert, field_def):
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

    def get_data(self):
        return self._data_list

    def get_all(self, field_name):
        data = []

        for frame in self._data_list:
            data.append(frame[field_name])

        return data

