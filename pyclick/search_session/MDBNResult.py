from pyclick.search_session import SearchResult

class MDBNResult(SearchResult):
    def __init__(self, search_result_id, click, sat):
        super(MDBNResult, self).__init__(search_result_id, click)

        if sat is None:
            # Not defined if click is not defined
            self.sat = None
        elif sat in [0, 1]:
            self.sat = sat
        else:
            raise RuntimeError("Invalid satisfaction value: %s" % sat)