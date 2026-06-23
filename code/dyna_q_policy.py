"""Rol C - Planning: Egitilmis Dyna-Q'yu act(obs) kontratina saran politika."""
import numpy as np

from dispatch_features import (decision_context, nearest_hub_distance,
                               ASSIGN, CHARGE)
from dyna_q_agent import DynaQAgent


class DynaQPolicy:
    """Egitilmis Q-tablosunu kullanarak act(obs) -> env aksiyon numarasi.
    Gorulmeyen durumlarda guvenli kurala duser (soc dusukse sarj, yoksa ata)."""

    def __init__(self, cfg, weights_path=None, charge_threshold=0.30):
        self.cfg = cfg
        self.charge_threshold = charge_threshold
        self.agent = DynaQAgent(seed=0)
        if weights_path is not None:
            self.agent.load(weights_path)
        self._hub_cache = {}     # grid -> hub_field (her episode'da bir kez hesapla)

    def _hub_field(self, grid):
        key = grid.tobytes()
        if key not in self._hub_cache:
            self._hub_cache = {key: nearest_hub_distance(grid)}   # tek grid sakla
        return self._hub_cache[key]

    def act(self, obs):
        hub_field = self._hub_field(obs["grid"])
        state, mask, a_act, c_act, n_act, info = decision_context(obs, self.cfg, hub_field)

        if state is None:                      # gercek karar yok -> bekle
            return n_act

        q = self.agent.Q.get(state)            # bu durumu egitimde gorduk mu?
        if q is None:
            macro = self._fallback(mask, info) # gorulmemis -> guvenli kural
        else:
            qq = np.where(mask, q, -np.inf)     # gecersiz aksiyonlari ele
            macro = int(np.argmax(qq))          # en iyi gecerli aksiyon (eps=0)

        if macro == CHARGE and c_act >= 0:
            return c_act
        return a_act        # varsayilan: ata (DUZELTME: eskiden yanlislikla noop donuyordu)

    def _fallback(self, mask, info):
        """Egitimde gorulmemis durum: greedy benzeri guvenli kural."""
        if mask[CHARGE] and info.get("soc", 1.0) < self.charge_threshold:
            return CHARGE
        return ASSIGN