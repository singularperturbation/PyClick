'''
Modified DBN model for multiple satisfaction events.

Based fairly heavily on the DBN model.
'''

from enum import Enum
import itertools
import math

from pyclick.click_models.ClickModel import ClickModel
from pyclick.click_models.Inference import EMInference

from pyclick.click_models.Param import ParamEM, ParamStatic
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
        # Session params is a list (by rank) of dictionaries that map to parameter
        # values at that rank.
        session_params = super(MDBN, self).get_session_params(search_session)

        session_exam = self._get_session_exam(search_session, session_params)
        session_car = self._get_session_clickafterrank(search_session, session_params)

        for rank, session_param in enumerate(session_params):
            # Strangely, these are not updated with EM like other params but fixed
            # until we have a new call of get_session_params.  I see examination prob
            # being recalculated ad-hoc where needed, but I think this is unavoidable.
            session_param[self.param_names.exam] = ParamStatic(session_exam[rank])
            session_param[self.param_names.car] = ParamStatic(session_car[rank])

        return session_params

    def _get_session_exam(self, search_session, session_params):
        '''
        Get E_{r} examination probability for each search result in session.

        :search_session: Observed search session
        :session_params: Current values of parameters for given search session.

        :returns: Examination probabilities by rank.
        '''
        session_exam = [1]

        for rank, session_param in enumerate(session_params):
            # Reminder:
            # cont_sat = gamma1
            # cont_nosat = gamma2

            attr = session_param[self.param_names.attr].value()
            sat = session_param[self.param_names.sat].value()
            cont_sat = session_param[self.param_names.cont_sat].value()
            cont_nosat = session_param[self.param_names.cont_nosat].value()

            exam = session_exam[rank]

            # This is the new update rule for E_{r}
            exam *= cont_nosat * ((1 - sat) * attr + (1 - attr)) + \
                cont_sat * attr * sat

            session_exam.append(exam)

        return session_exam

    def _get_session_clickafterrank(self, search_session, session_params):
        '''
        Calculate the probability of a click on current result or any result below.

        P(C_{>=r} | E_r = 1), where r is rank of current search result.

        :search_session: Observed search session.
        :session_params: Current values of parameters for search session.

        :returns: List of P(C_{>=r} | E_r = 1) for session.
        '''

        # Note: This is the same as DBN._get_session_clickafterrank (except have
        # to make sure that using cont_nosat).  Reason for this is that continuation
        # after click is irrelevant for predicting click afterrank, as this means will
        # have already have had to click.

        session_car = [0] * (len(search_session.web_results) + 1)

        for rank in range(len(search_session.web_results) - 1, -1, -1):
            attr = session_params[rank][self.param_names.attr].value()
            cont = session_params[rank][self.param_names.cont_nosat].value()
            car = session_car[rank + 1]

            car = attr + (1 - attr) * cont * car

            session_car[rank] = car

        return session_car

    def get_full_click_probs(self, search_session):
        '''
        Probability of click at rank.  Taken verbatim from DBN class.
        '''
        session_params = self.get_session_params(search_session)
        click_probs = []

        for rank, session_param in enumerate(session_params):
            attr = session_param[self.param_names.attr].value()
            exam = session_param[self.param_names.exam].value()

            click_probs.append(attr * exam)
        
        return click_probs
    
    def get_conditional_click_probs(self, search_session):
        session_params = self.get_session_params(search_session)
        return self._get_tail_clicks(search_session, 0, session_params)[0]
    
    def predict_relevance(self, query, search_result):
        attr = self.params[self.param_names.attr].get(query, search_result).value()
        sat = self.params[self.param_names.sat].get(query, search_result).value()

        return attr * sat
    
    @classmethod
    def _get_continuation_factor(cls, search_session, rank, session_params):
        '''
        Calculate the \Phi(x, y, z) continuation factor.  This should be very
        similar to the DBN classmethod.
        '''
        click = search_session.web_results[rank].click
        attr = session_params[rank][cls.param_names.attr].value()
        sat = session_params[rank][cls.param_names.sat].value()
        cont_sat = session_params[rank][cls.param_names.cont_sat].value()
        cont_nosat = session_params[rank][cls.param_names.cont_nosat].value()

        def factor(x, y, z):
            '''
            Calculates \Phi(x, y, z) for a particular rank, given state above.

            Can test as - when cont_nosat = 0.0, this should be the same as the
            DBN model.
            '''

            last_click = search_session.get_last_click_rank()

            log_prob = 0.0
            # Have 5 parts of this
            # 1) P(E_r = x | C_{<r}) - conditional examination given clicks, adapted 3.48
            # 2) P(C_r = c_r | E_r = x) - have formula
            # 3) P(S_r = y | C_r = c_r) - have formula
            # 4) P(E_{r+1} = z | E_r = x, S_r = y, C_r = c_r) - have formula
            # 5) P(C_{>=r+1} | E_{r+1} = z)

            # Calculate (2-4) of above
            if not click:
                # Satisfied but not clicked - impossible
                if y:
                    return 0.0

                if x:
                    # P(E_r = 1, S_r = 0, E_{r+1} = z, C_r = 0)
                    # (x, y, z) = (1, 0, z)
                    log_prob += math.log(cont_nosat if z else (1 - cont_nosat))
                elif z:
                    # No examination at r -> no examination at r + 1
                    return 0.0

                # P(E_r = 1, S_r = 0, E_{r+1} = 0, C_r = 0)
                # Adding probability of not being attracted to content given examined
                log_prob += math.log(1 - attr)
            else:
                # Clicked but not examined - impossible state
                if not x:
                    return 0.0

                if not y:
                    log_prob += math.log(1 - sat)
                    log_prob += math.log(cont_nosat if z else (1 - cont_nosat))
                else:
                    log_prob += math.log(sat)

                    # Continuing after satisfaction events is possible
                    if z:
                        log_prob += math.log(cont_sat)
                    else:
                        log_prob += math.log(cont_nosat)

                # Probability for examined and clicked
                log_prob += math.log(attr)

            # Compute part (5) P(C_{>r} | E_{r+1} = z)
            if not z:
                if search_session.get_last_click_rank() >= rank + 1:
                    # Clicks after this is impossible if we've stopped examining
                    # here.
                    return 0.0
            elif rank + 1 < len(search_session.web_results):
                log_prob += sum(math.log(p) for p in cls._get_tail_clicks(search_session,
                                                                          rank + 1,
                                                                          session_params)[0])


            # Part (1) P(E_r = 1 | \mathbf{C}_{<r})
            exam = cls._get_tail_clicks(search_session, 0, session_params)[1][rank]
            log_prob += math.log(exam if x else (1 - exam))

            return math.exp(log_prob)

        return factor


    @classmethod
    def _get_tail_clicks(cls, search_session, start_rank, session_params):
        '''
        Calculate P(C_r | C_{r-1}, ..., C_l, E_l = 1), P(E_r = 1 | C_{r-1}, ..., C_l, E_l = 1)
        for each r in [l, n) where l is start_rank.
        '''
        exam = 1.0
        click_probs = []
        exam_probs = [exam]

        for rank, result in enumerate(search_session.web_results[start_rank:]):
            attr = session_params[rank][cls.param_names.attr].value()
            sat = session_params[rank][cls.param_names.sat].value()
            cont_sat = session_params[rank][cls.param_names.cont_sat].value()
            cont_nosat = session_params[rank][cls.param_names.cont_nosat].value()

            clicked = result.click
            # Not sure if should use result.sat here, or inferred satisfactoriness
            # of the link ('sat').
            #
            # Think it makes sense here, since looking at Prob Click at rank r for
            # particular session and particular query.
            result_sat = result.sat

            if clicked:
                click_prob = attr * exam
                # exam = cont_nosat * (1 - sat) + cont_sat * sat

                if result_sat:
                    exam = cont_sat
                else:
                    exam = cont_nosat
            else:
                click_prob = 1 - attr * exam
                exam *= cont_nosat * (1 - attr) / click_prob

            click_probs.append(click_prob)
            exam_probs.append(exam)

        return click_probs, exam_probs


class MDBNAttrEM(ParamEM):
    '''
    Attractiveness parameter of the MDBN model.
    Infer using EM algorithm and update using derived $\alpha_{uq}$ update rule.

    This is the same as DBN only because the car value is calculated
    differently.
    '''
    
    def update(self, search_session, rank, session_params):
        if search_session.web_results.click:
            self._numerator += 1
        
        # If clicks after this result (and not clicked), then this is not
        # attractive, as it was examined but not clicked.
        elif rank >= search_session.get_last_click_rank():
            attr = session_params[rank][MDBN.param_names.attr].value()
            exam = session_params[rank][MDBN.param_names.exam].value()
            car = session_params[rank][MDBN.param_names.car].value()

            num = (1 - exam) * attr
            denom = 1 - exam * car

            self._numerator += num / denom

        self._denominator += 1


class MDBNSatEM(ParamEM):
    '''
    Document satisfactoriness parameter.
    This value can be directly computed by looking at the number of satisfaction
    events observed divided by the total number of clicks for a given document.

    I'm not sure if this is the correct way to implement this, as we're really
    doing MLE extimation.  Think this has to be, since EMInference.infer_params
    will be doing the updates, trying to pass in the old session_params as part of
    the update.
    '''
    def update(self, search_session, rank, session_params):
        if search_session.web_results[rank].click:
            if search_session.web_results[rank].sat:
                self._numerator += 1

            self._denominator += 1


class MDBNContSatEM(ParamEM):
    '''
    Continuation parameter when do *not* have satisfaction event for current rank.
    This is the same as the 'gamma1' parameter.
    Value inferred using EM.
    '''

    def update(self, search_session, rank, session_params):
        factor = MDBN._get_continuation_factor(search_session, rank, session_params)
        # \gamma_1 = P(E_r = 1, S_r = 1, E_{r+1} = z | C) / sum_{x,y,z}(\phi(x,y,z))
        # Where x, y, z and be 0 or 1
        exam_prob = lambda z: factor(1, 1, z) / sum(
            factor(*p) for p in itertools.product([0, 1], repeat=3))
        
        self._numerator += exam_prob(1)
        self._denominator += sum(exam_prob(z) for z in [0, 1])


class MDBNContNoSatEM(ParamEM):
    '''
    Continuation parameter when do *not* have satisfaction event for current rank.
    This is the same as the 'gamma2' / DBNContEM parameter.
    '''

    def update(self, search_session, rank, session_params):
        factor = MDBN._get_continuation_factor(search_session, rank, session_params)
        # \gamma_1 = P(E_r = 1, S_r = 1, E_{r+1} = z | C) / sum_{x,y,z}(\phi(x,y,z))
        # Where x, y, z and be 0 or 1

        exam_prob = lambda z: factor(1, 0, z) / sum(
            factor(*p) for p in itertools.product([0, 1], repeat=3))

        self._numerator += exam_prob(1)
        self._denominator += sum(exam_prob(z) for z in [0, 1])