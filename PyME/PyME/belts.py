import numpy as np

from MT22ta.Admir_Filan.ME_kraefte_kettentrieb_filanadm import Ft

_TROLLEY_DIAMETER_ = np.array([
    [0.00075, 0.0009, 0.001, 0.0016, 0.0018, 0.003, 0.0045, 0.008, 0.01, 0.015, 0.04, 0.06, 0.1, 0.12,  0.14, 0.17, 0.2, 0.25, 0.3, 0.4, 0.44],
    [63, 71, 80, 90, 100, 112, 125, 140, 160, 180, 200, 224, 250, 280, 315, 355, 400, 450, 500, 560, 630]
])

class Belt(object):
    def __init__(
        self,
        torque,
        num_of_rev,
        app_factor,
        trolley_diameter_drive,
        trolley_diameter_load
    ):
        self.torque = torque
        self.num_of_rev = num_of_rev
        self.app_factor = app_factor
        self.trolley_diameter_drive = trolley_diameter_drive
        self.trolley_diameter_load = trolley_diameter_load


class ToothBelt(Belt):
    def __init__(
        self,
        torque: float|int,
        num_of_rev: float|int,
        ratio: float|int,
        app_factor: float|int
    ):
        self.angular_velocity = num_of_rev / 60 * 2 * np.pi
        self.power = torque * self.angular_velocity * app_factor
        ptn = self.power/1000 / num_of_rev
        index = np.where(_TROLLEY_DIAMETER_[0] >= ptn)
        d1 = _TROLLEY_DIAMETER_[1, index[0]]
        d2 = d1 * ratio
        super().__init__(torque, num_of_rev, app_factor, d1, d2)

    def calculate_toothbelt_length(self):
        v = self.trolley_diameter_drive * np.pi * self.num_of_rev
        Ft = self.power / v
        dm = (self.trolley_diameter_drive + self.trolley_diameter_load) * 0.5
        e_ = (15 + dm + 2 * (self.trolley_diameter_drive + self.trolley_diameter_load)) / 2
        Ld_ = 2 * e_ + np.pi/2 * (self.trolley_diameter_drive + self.trolley_diameter_load) + (self.trolley_diameter_drive + self.trolley_diameter_load)**2 / (4 * e_)



class FlatBelt(Belt):
    def __init__(self):
        pass
