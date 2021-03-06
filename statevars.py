"""
file: statevars.py
created: 20160806
author(s): mr-augustine

The Statevars class is responsible for parsing the statevars definition file.

Lines in the statevars definition file would look similar to this:
uint32_t, main_loop_counter
float, gps_latitude
float, gps_longitude
char[84], gprmc_sentence
"""
class Statevars:

    FIELD_DELIMETER = ','
    FIELD_DATATYPE_INDEX = 0
    FIELD_COUNT_INDEX = 1
    FIELD_NAME_INDEX = 2
    FIELD_EXPECTED_VAL_INDEX = 3
    VAR_TYPE_SIZE_MAP = {'uint32_t':4,
                            'uint16_t':2,
                            'uint8_t':1,
                            'int32_t':4,
                            'int16_t':2,
                            'int8_t':1,
                            'char':1,
                            'float':4,
                            'double':4}

    _varlist = [];

    def __init__(self, filename):
        #Open the file
        #Parse the file
        #Populate the list of statevars (type, size, name)
        if not self._parse_statevars_file(filename):
            self._varlist = None

    @staticmethod
    def get_datatype_size(datatype):
        """ Returns the size of the specified datatype """

        if datatype in Statevars.VAR_TYPE_SIZE_MAP.keys():
            return Statevars.VAR_TYPE_SIZE_MAP[datatype]

        return 0

    @staticmethod
    def get_field_count(field_def):
        """ Returns the number of contiguous occurences for the
        specified field
        """

        return field_def[Statevars.FIELD_COUNT_INDEX]

    @staticmethod
    def get_field_datatype(field_def):
        """ Returns the datatype of the specified field """

        return field_def[Statevars.FIELD_DATATYPE_INDEX]

    @staticmethod
    def get_field_exp_val(field_def):
        """ Returns the expected value of the specified field """

        return field_def[Statevars.FIELD_EXPECTED_VAL_INDEX]

    @staticmethod
    def get_field_name(field_def):
        """ Returns the field name """

        return field_def[Statevars.FIELD_NAME_INDEX]

    @staticmethod
    def has_exp_val(field_def):
        """ Returns True if the specified field has an associated
        expected value
        """

        return Statevars.get_field_exp_val(field_def) != ''

    def _is_array(self, array_name):
        """ Returns 1 if the statevar field name is an array of the format:
        array_name[<array_size>]; Returns 0 otherwise
        """

        if (array_name.count('[') == 0) or (array_name.count(']') == 0):
            return 0

        # Verify that the array length is at least one digit wide
        left_bracket_pos = array_name.find('[')
        if left_bracket_pos < len(array_name) - 2 and  array_name.endswith(']'):
            array_element_count = array_name[left_bracket_pos + 1:len(array_name) - 1]

            if array_element_count.isdigit() and int(array_element_count > 0):
                return 1

        return 0

    def _parse_array(self, array_name):
        """ Returns a tuple that contains the name and size of the specified
        array
        """

        left_bracket_pos = array_name.find('[')
        array_element_count = array_name[left_bracket_pos + 1 : len(array_name) - 1]

        return (array_name[:left_bracket_pos], int(array_element_count))

    def _parse_datatype(self, datatype):
        """ Parses the datatype; Returns a tuple containing the datatype name
        and the count (number of contiguous values of that type)
        """

        if self._is_array(datatype):
            (datatype, count) = self._parse_array(datatype)
        else:
            count = 1

        if datatype not in self.VAR_TYPE_SIZE_MAP.keys():
            print 'Unrecognized data type: ' + datatype
            return ('', '')
        else:
            return (datatype, count)

    def _parse_line(self, line):
        """ Parses the specified line for statevar definition fields;
        Returns a tuple that contains the datatype, name and expected value
        """

        if line.count(self.FIELD_DELIMETER) == 1:
            (datatype, name) = line.split(self.FIELD_DELIMETER, 2)
            exp_val = ''
        elif line.count(self.FIELD_DELIMETER) == 2:
            (datatype, name, exp_val) = line.split(self.FIELD_DELIMETER, 3)
        else:
            (datatype, name, exp_val) = ('', '', '')
            print 'Malformed statevars definition: ' + line

        return (str.strip(datatype), str.strip(name), str.strip(exp_val))

    def _parse_statevars_file(self, filename):
        """ Parses the specified statevars file """

        with open(filename) as f:
            lines = f.readlines()

        line_number = 0
        for line in lines:
            line_number = line_number + 1

            line = str.strip(line)

            # Ignore commented lines
            if line[0] == '#':
                continue

            # datatype and name are mandatory for all lines
            (datatype, name, exp_val) = self._parse_line(line)

            if '' in (datatype, name):
                # There was an error
                print 'line number: ' + str(line_number)
                return 0

            (datatype, count) = self._parse_datatype(datatype)

            if '' in (datatype, count):
                return 0

            self._varlist += [(datatype, count, name, exp_val)]

        return 1

    def get_varlist(self):
        """ Returns the list of statevars variables """

        return self._varlist
