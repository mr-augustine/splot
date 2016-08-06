class Statevars:
    'Class responsible for interpreting the statevars definition file'

    DELIMITER = ','
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
        if not self.__parse_statevars_file(filename):
            self._varlist = None

    # You give it a filename and it parses its contents
    def __parse_statevars_file(self, filename):
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
            (datatype, name, exp_val) = self.__parse_line(line)

            if '' in (datatype, name):
                # There was an error
                print 'line number: ' + str(line_number)
                return 0

            (datatype, count) = self.__parse_datatype(datatype)

            if '' in (datatype, count):
                return 0

            self._varlist += [(datatype, count, name, exp_val)]

        return 1

    def __parse_line(self, line):
        if line.count(self.DELIMITER) == 1:
            (datatype, name) = line.split(self.DELIMITER, 2)
            exp_val = ''
        elif line.count(self.DELIMITER) == 2:
            (datatype, name, exp_val) = line.split(self.DELIMITER, 3)
        else:
            (datatype, name, exp_val) = ('', '', '')
            print 'Malformed statevars definition: ' + line

        return (str.strip(datatype), str.strip(name), str.strip(exp_val))

    def __parse_datatype(self, datatype):
        if self.__is_array(datatype):
            (datatype, count) = self.__parse_array(datatype)
        else:
            count = 1

        if datatype not in self.VAR_TYPE_SIZE_MAP.keys():
            print 'Unrecognized data type: ' + datatype
            return ('', '')
        else:
            return (datatype, count)

    def __is_array(self, array_name):
        if (array_name.count('[') == 0) or (array_name.count(']') == 0):
            return 0

        left_bracket_pos = array_name.find('[')
        if left_bracket_pos < len(array_name) - 2 and  array_name.endswith(']'):
            array_element_count = array_name[left_bracket_pos + 1:len(array_name) - 1]

            if array_element_count.isdigit() and int(array_element_count > 0):
                return 1

        return 0

    def __parse_array(self, array_name):
        left_bracket_pos = array_name.find('[')
        array_element_count = array_name[left_bracket_pos + 1 : len(array_name) - 1]

        return (array_name[:left_bracket_pos], int(array_element_count))

    def get_varlist(self):
        return self._varlist

    @staticmethod
    def get_field_datatype(field_def):
        return field_def[Statevars.FIELD_DATATYPE_INDEX]

    @staticmethod
    def get_field_count(field_def):
        return field_def[Statevars.FIELD_COUNT_INDEX]

    @staticmethod
    def get_field_name(field_def):
        return field_def[Statevars.FIELD_NAME_INDEX]

    @staticmethod
    def get_field_exp_val(field_def):
        return field_def[Statevars.FIELD_EXPECTED_VAL_INDEX]

    @staticmethod
    def get_datatype_size(datatype):
        if datatype in Statevars.VAR_TYPE_SIZE_MAP.keys():
            return Statevars.VAR_TYPE_SIZE_MAP[datatype]

        return 0

    @staticmethod
    def has_exp_val(field_def):
        return Statevars.get_field_exp_val(field_def) != ''
