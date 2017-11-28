'''
Modified DBN model for multiple satisfaction events.

Based fairly heavily on the DBN model.
'''

from enum import Enum

from pyclick.click_models.ClickModel import ClickModel
from pyclick.click_models.Inference import EMInference

from pyclick.click_models.Param import ParamMLE, ParamEM
from pyclick.click_models.ParamContainer import QueryDocumentParamContainer, SingleParamContainer

__author__ = 'Sloane Simmons'

class MDBN(ClickModel):
    '''
    A modification of the DBN click model.  Multiple 'satisfaction' events can
    occur, and satisfaction is explicitly observed.

    A satisfaction event here is a like or favorite for a particular product in
    search.
    '''

    param_names = Enum('MDBNParams', 'attr sat cont_sat cont_nosat exam car')
    '''
    Names of MDBN parameters.

    :attr: attractiveness parameter
    Probability of attracting click on document given examined.
    :sat: satisfactoriness parameter
    Also known as 'true relevance'.  Probability satisfies given clicked.
    :cont_sat: continuation given that current document satisfies
    :cont_nosat: continuation given that current document *does not* satisfy
    This is equivalent to the continuation parameter in the DBN model.
    :exam: examination probability
    Does the user examine this search result?
    :car: probability of click on or after $r$ given examination at $r$.
    Whether user clicks on current or any result below current result.
    '''
    
    def __init__(self, inference=EMInference()):
        self.params = {
            self.param_names.attr: QueryDocumentParamContainer(MDBNAttrEM),
            self.param_names.sat: QueryDocumentParamContainer(MDBNSatEM),
            self.param_names.cont_sat: SingleParamContainer(MDBNContSatEM),
            self.param_names.cont_nosat: SingleParamContainer(MDBNContNoSatEM)
        }
        self._inference = inference
    
    def get_session_params(self, search_session):
        '''
        Overrides ClickModel.get_session_params to get the examination and car
        parameters.
        '''
        session_params = super(MDBN, self).get_session_params(search_session)

        # TODO: Write _get_session_foo methods for exam, car
    
    def get_full_click_probs(self, search_session):
        pass
    
    def get_conditional_click_probs(self, search_session):
        pass
    
    def predict_relevance(self, query, search_result):
        pass
    
class MDBNAttrEM(ParamEM):
    pass

class MDBNSatEM(ParamEM):
    pass

class MDBNContSatEM(ParamEM):
    pass

class MDBNContNoSatEM(ParamEM):
    pass