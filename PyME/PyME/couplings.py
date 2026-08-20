class Coupling(object):
    def __init__(
            self,
            inertia_load,
            inertia_drive,
            inertia_coupling,
            drive_power,
            load_torque,
            number_of_rev,
            Ki_Kn,
            app_factor
    ):
        self.inertia_load = inertia_load
        self.inertia_drive = inertia_drive
        self.inertia_coupling = inertia_coupling
        self.power = drive_power
        self.load_torque = load_torque
        self.number_of_rev = number_of_rev
        self.Ki_Kn = Ki_Kn
        self.app_factor = app_factor

    def calc_coupling(
            self,
            torque_factor_load: int|float = 1,
            torque_factor_drive: int|float = 1,
            start_up_factor: int|float = 1,
            temperatur_factor: int|float = 1
    ):
        self.equivalent_drive_torque = self.power * self.app_factor / self.number_of_rev * 9550 * self.Ki_Kn
        self.equivalent_load_torque = self.app_factor * self.load_torque

        self.J_A = self.inertia_drive + self.inertia_coupling / 2
        self.J_L = self.inertia_load + self.inertia_coupling / 2
        self.J =self.J_L + (self.J_A + self.J_L)
        dynamical_torque_load = self.equivalent_load_torque * torque_factor_load
        dynamical_torque_drive = self.equivalent_drive_torque * torque_factor_drive
        self.torque_coupling = self.J * (dynamical_torque_drive + dynamical_torque_load) * start_up_factor * temperatur_factor
        return self.torque_coupling

    def markdown(self):
        return (
            r" ### Kupplungsberechnung " " \n " "***" " \n "
            r" Antriebsseitiges equivalentes Drehmoment " " \n "
            r" $$ T_{A,eq} = [Ti/Tn] \frac{9550 P K_A}{n} = " f" {self.equivalent_drive_torque} Nm $$ "
            r" Lastseitiges equivalentes Spitzendrehmoment " " \n "
            r" $$ T_{L,eq} = K_A T_L = " f" {round(self.equivalent_load_torque, 1)} Nm $$ "
            r" Lastseitiges Trägheitsmoment " " \n "
            r" $$ J_L = J_{Verbraucher} + \frac{J_K}{2} = " f" {round(self.J_L, 6)} kgm² $$ "
            r" Antriebsseitiges Trägheitsmoment " " \n "
            r" $$ J_A = J_{Motor} + \frac{J_K}{2} = " f" {round(self.J_A, 6)} kgm² $$ "
            r" rechnerisches Kupplungsmoment " " \n "
            r" $$ T_K = \frac{J_L}{J_A + J_L} S_z S_T (S_A T_{A,eq} + S_L T_{L,eq) = " f" {round(self.torque_coupling, 1)} Nm $$ "
        )
