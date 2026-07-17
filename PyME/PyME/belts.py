import numpy as np

_TROLLEY_DIAMETER_ = np.array([
    [0.00075, 0.0009, 0.001, 0.0016, 0.0018, 0.003, 0.0045, 0.008, 0.01, 0.015, 0.04, 0.06, 0.1, 0.12,  0.14, 0.17, 0.2, 0.25, 0.3, 0.4, 0.44],
    [63, 71, 80, 90, 100, 112, 125, 140, 160, 180, 200, 224, 250, 280, 315, 355, 400, 450, 500, 560, 630]
])
_TOOTHBELT_PARAMS_ = np.array([
    # teilung, zahnhöhe, min Länge, max Länge, min Scheiben-Zähnezahl, max Scheiben-Zähnezahl, min Zähnezahl bei gegenbiegung
    [2.5, 0.7, 120, 1475, 10, 114, 11, 0.5, 20000, 80],
    [5, 1.2, 100, 1500, 10, 114, 12, 5, 10000, 80],
    [10, 2.5, 260, 3620, 15, 114, 20, 30, 10000, 60],
    [20, 5, 1260, 3620, 15, 114, 20, 100, 6500, 40]
])
_TOOTHBELT_FORCE_ = np.array([
    [39, 65, 117, 195, 312, 403, 0, 0, 0, 0],
    [0, 150, 300, 510, 870, 1100, 1800, 2730, 3660, 0],
    [0, 0, 0, 1200, 2000, 2700, 4300, 6600,  8800, 13400],
    [0, 0, 0, 0, 0, 4750, 7750, 12000, 16000, 24500]
])
_TOOTHBELT_WIDTH_ = np.array([4, 6, 10, 16, 25, 32, 50, 75, 100, 150])

class Belt(object):
    def __init__(
        self,
        torque,
        num_of_rev,
        ratio,
        app_factor
    ):
        self.torque = torque
        self.num_of_rev = num_of_rev
        self.app_factor = app_factor
        self.ratio = ratio


class ToothBelt(Belt):
    def __init__(
        self,
        torque: float|int,
        num_of_rev: float|int,
        ratio: float|int,
        trolley_diameter_drive: float|int,
        app_factor: float|int,
        pitch: float|int
    ):
        # rotationsgeschwindigkeit der Antriebsseite
        self.angular_velocity = num_of_rev / 60 * 2 * np.pi
        # zu übertragende Leistung
        self.power = torque * self.angular_velocity * app_factor
        self.trolley_diameter_drive = trolley_diameter_drive
        self.trolley_diameter_load = trolley_diameter_drive * ratio
        self.pitch = pitch
        super().__init__(torque, num_of_rev, ratio, app_factor)

    def step_1_calc_toothbelt_length(self):
        """
        Berechne die Riemenlänge und wähle aus RM TB 16 eine passende Länge.

        :returns:
            e' (float|int): Theoretischer Wellenabstand [mm]
            Ld' (float|int): Theoretische Riemenlänge [mm]
        """
        self.v = self.trolley_diameter_drive/1000 * np.pi * self.num_of_rev/60
        self.tangential_force = self.power / self.v
        dm = self.trolley_diameter_drive + self.trolley_diameter_load
        e_ = (15 + dm / 2 + dm / 2) / 2
        Ld_ = 2 * e_ + np.pi/2 * dm + dm**2 / (4 * e_)
        return e_, Ld_


    def step_2_calc_shaft_distance(self, belt_length: float|int, custom_wrap_angle: float|int = 0):
        """
        Berechne den Achsabstand, den Umschlingwinkel am treibenden Rad und den Spannweg des Riemens.

        :param belt_length: (float|int) gewählte Riemenlänge [mm]

        :returns:
            Wellenabstand e [mm],
            Umschlingwinkel beta_k [rad],
            Spannweg x [mm]
        """
        self.belt_length = belt_length
        self.index = np.where(_TOOTHBELT_PARAMS_[:, 0] >= self.pitch)[0]
        self.belt_params = _TOOTHBELT_PARAMS_[self.index][0]
        if self.belt_length < self.belt_params[2]:
            print(f"Warnung, Riemenlänge kürzer als erlaubt: Riemenlänge festgelegt auf {self.belt_params[2]} mm")
            self.belt_length = self.belt_params[2]
        elif self.belt_length > self.belt_params[3]:
            print(f"Warnung, Riemenlänge länger als erlaubt: Riemenlänge festgelegt auf {self.belt_params[3]} mm")
            self.belt_length = self.belt_params[3]
        dsum = self.trolley_diameter_drive + self.trolley_diameter_load
        a = self.belt_length / 4 - np.pi / 8 * dsum
        b = np.sqrt((belt_length / 4 - np.pi / 8 * dsum)**2 - dsum**2 / 8)
        self.e = a + b
        if not custom_wrap_angle:
            self.wrap_angle = 2 * np.arccos((self.trolley_diameter_load - self.trolley_diameter_drive) / (2 * self.e))
        else:
            self.wrap_angle = custom_wrap_angle
        self.x = 0.005 * belt_length
        return self.e, self.wrap_angle, self.x

    def step_3_calc_module(self):
        """
        Berechne das Modul des Riemens.

        :returns:
            Anzahl Zähne im Eingriff ze,
            Zähne des kleineren Rads zk,
            erforderlicher Umschlingwinkel damit ze = 12 [grad],
            erforderliche Übersetzung damit ze = 12
        """
        self.zk = self.trolley_diameter_drive / self.pitch * np.pi
        self.ze = self.zk * np.rad2deg(self.wrap_angle) / 360
        # theoretisch benötigter Umschlingwinkel für ze = 12
        wrap_a_ = 12 / self.zk * 360
        # theoretisch benötigte Übersetzung für ze = 12
        ratio_ = np.cos(np.deg2rad(wrap_a_) / 2) * 2 * self.belt_length * np.pi / (self.pitch * self.zk) + 1
        return self.ze, self.zk, wrap_a_, ratio_

    def step_4_calc_beltwidth(self, P_spez: float|int):
        """
        Berechne die Riemenbreite.

        :param P_spez: (float|int) Spezifische übertragbare Leistung des Riemens [kW/mm]

        :returns:
            theoretische Riemenbreite b' [mm]
        """
        self.P_spez = P_spez
        belt_width_ = self.power/1000 / (self.zk * self.ze * P_spez)
        return belt_width_

    def step_5_control_calculation(
            self,
            belt_width: float|int,
            num_of_trolleys: int
        ):
        """
        Berechne die Ausnutzung der zulässigen Tangentialkraft des Riemens.

        :param belt_width: (float|int) gewählte Riemenbreite [mm]
        :param num_of_trolleys: (int) Anzahl Rollen inklusive Spannrollen

        :returns:
            Sicherheitsfaktor Ft_zul / Ft
            Sicherheitsfaktor v_zul / v
            Seitenkraft auf die Welle F_w0 [N]
            Biegefrequenz des Riemens f_B [Hz]
        """
        self.belt_width = belt_width
        Ftzul = _TOOTHBELT_FORCE_[self.index][0]
        index_b = np.where(_TOOTHBELT_WIDTH_ <= belt_width)[0]
        self.Ft_zul = Ftzul[index_b[-1]]
        sf1 = self.Ft_zul / self.tangential_force
        sf2 = self.belt_params[-1] / self.v
        Fw0 = 1.1 * self.tangential_force
        fB = self.v * num_of_trolleys / self.belt_length
        return sf1, sf2, Fw0, fB


class FlatBelt(Belt):
    def __init__(
        self,
        torque: float|int,
        num_of_rev: float|int,
        ratio: float|int,
        app_factor: float|int
    ):
        # rotationsgeschwindigkeit der Antriebsseite
        self.angular_velocity = num_of_rev / 60 * 2 * np.pi
        # zu übertragende Leistung
        self.power = torque * self.angular_velocity * app_factor
        # P / n verhältnis für die Wahl der Raddurchmesser
        ptn = self.power / 1000 / num_of_rev
        index = np.where(_TROLLEY_DIAMETER_[0] >= ptn)
        # Durchmesser des treibenden Rads
        d1 = _TROLLEY_DIAMETER_[1, index[0]]
        d1 = d1[0]
        # Durchmesser des getriebenen Rads
        d2 = d1 * ratio
        pass

