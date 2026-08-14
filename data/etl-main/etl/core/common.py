class ColumnNotExists(Exception):
    """ Raise when Column does not exist """
    def __str__(self):
        return "Column does not exist"

class NoRecordFound(Exception):
    """ Raise when there is no record present """
    def __str__(self):
        return "No Data Found"
