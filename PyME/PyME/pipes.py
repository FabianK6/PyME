import numpy as np

def propose_diameter(Vdot: float|int, w: float|int):
    return np.sqrt(4 * Vdot / (np.pi * w))

def propose_wallthickness(
        p: float|int,
        DN: float|int,
        Rp02: float|int,
        c1: int|float = 0.01,
        c2: float|int = 1,
        S: int|float = 1.5,
        weld_factor: float|int = 0.7
):
    """
    Berechne die Wandstärke t bestehend aus der
    Mindest-Wandstärke tv und den Zusätzen c1 und c2

    :param p: Innendruck [bar]
    :param DN: Aussendurchmesser [mm]
    :param Rp02: Elastizitätsgrenze des Rohrs [MPa]
    :param c1: Zuschlag zum Ausgleich von Fertigungstoleranzen. Meist im Bereich von 8% bis 20% [%]
    :param c2: Korrosions- Erosionszuschlag. Meist 1 mm, wenn Korrosion zu erwarten ist. [mm]
    :param S: Sicherheitsfaktor gegen Fliessen (1.5) oder Bruch (3) [1]
    :param weld_factor: Schweissnahtfaktor, je nach Nachweisverfahren im Bereich 0.7...1 [1]
    :return: Wandstärke t [mm]
    """
    p /= 10
    return (p * DN / (2 * Rp02/S * weld_factor + p) + c2) * (c1 + 1)

class Pipe(object):
    def __init__(
            self,
            DN: float|int,
            PN: float|int,
            t: float|int,
            L: float|int,
            b: float|int,
            k: float|int,
            surface_quality: str
    ):
        self.DN = DN
        self.PN = PN
        self.t = t
        self.L = L
        self.b = b
        self.k = k
        self.surface_quality = surface_quality
        self.di = self.DN - 2 * self.t

    def calc_fluid_velocity(self, Q: float|int) -> float|int:
        A = self.di**2 * np.pi / 4
        return Q / A

    def calc_pressure_loss(
            self,
            Q: float|int,
            rho: float|int,
            eta: float|int
    ) -> float|int:
        w = self.calc_fluid_velocity(Q)
        Re = w * self.di / eta

        roughness = 64/ Re
        match self.surface_quality:
            case "rough":
                roughness = 1 / (2 * np.log(self.di / self.k) + 1.14) ** 2
            case "mid":
                r_array = np.linspace(0, 0.1, 1000)
                r = 1 / np.sqrt(r_array) + 2 * np.log(2.51 / (Re * np.sqrt(r_array) + self.k / (3.71 * self.di)))
                roughness = r_array[np.where(r >= 0)]
            case "smooth":
                roughness = 0.309 / np.log(Re / 7) ** 2
        return rho * w**2 / 2 * (self.b + roughness * self.L / self.di)

