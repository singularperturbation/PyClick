import json
from pyclick.search_session import SearchSession
from pyclick.search_session import MDBNResult

__author__ = 'Sloane Simmons'

class MDBNSession(SearchSession):
    '''
    A session type for the MDBN model that includes the observed satisfaction
    events.
    '''

    def __init__(self, query):
        super(MDBNSession, self).__init__(query)
        self.satisfaction_events = []
    
    def get_satisfaction_events(self):
        return [result.sat for result in self.web_results]
    
    @classmethod
    def from_JSON(cls, json_str):
        '''
        Extracts session from JSON using MDBNResult
        '''
        # Not calling super(MDBNSession, cls).from_JSON since have to handle
        # web results differently.

        session = cls("")
        session.__dict__ = json.loads(json_str)

        web_results = []

        for web_result_json in session.web_results:
            web_result = MDBNResult.from_JSON(web_result_json)
            
            web_results.append(web_result)
        
        session.web_results = web_results

        return session