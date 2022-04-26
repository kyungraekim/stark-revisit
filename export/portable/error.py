class ExportError(Exception):
    def __init__(self, attr):
        super(ExportError, self).__init__("{} is not supported in export".format(attr))
