import bending as beam
import numpy as np


class BoschProfile(object):
    def __init__(
            self, length: float,
            area: float, resistance_x: float,
            resistance_y: float, areaInertia_x: float,
            areaInertia_y: float, youngModulus: float, Rp02: float,
            density: float
    ):
        """
        Bosch-Profile. Verwendung der Parameter aus dem Bosch Rexroth-Katalog.

        Argumente:
            length (float): Länge des Profils [mm]
            area (float): Querschnitt des Profils [mm²]
            resistance_x (float): Widerstandsmoment bezüglich der x-Achse [mm³]
            resistance_y (float): Widerstandsmoment bezüglich der y-Achse [mm³]
            areaInertia_x (float): Flächenmoment 2. Grades bezüglich der x-Achse [mm⁴]
            areaInertia_y (float): Flächenmoment 2. Grades bezüglich der y-Achse [mm⁴]
            youngModulus (float): Elastizitätsmodul des Materials [MPa]
            Rp02 (float): Streckgrenze des Materials [MPa]
            density (float): Dichte des Materials [kg/m³]
        """
        self.length = length
        self.area = area
        self.resistance_x = resistance_x
        self.resistance_y = resistance_y
        self.areaInertia_x = areaInertia_x
        self.areaInertia_y = areaInertia_y
        self.youngModulus = youngModulus
        self.density = density
        self.Rp02 = Rp02
        pass

    def bendingForces(
            self, setOfNodes,
            direction,
            show_eqs: bool = False,
            show_plot: bool = False,
            K_A: float = 1.5,
            tol=1e-26
    ):
        """
        Biegespannungsanalyse des Profils.

        Argumente:
            setOfNodes (list): NDArray mit Knotenparametern pro Zeile im Format
                [position, type, [qy, Fy, Mt]]\n
                gültige Typen: boundary, mid, loose, glider, rigid, load
            direction (str): "x" oder "y"
            show_eqs (bool): Gleichungen anzeigen
            show_plots (bool): Diagramme anzeigen
            K_A (float): Anwendungsfaktor
            tol (float): Toleranz zwischen xi und x (Position, die Mbmax am nächsten liegt)

        Rückgabe:
            array: Daten
        """
        self.setOfNodes = setOfNodes
        if direction.lower() == "x":
            I = self.areaInertia_x
            W = self.resistance_x
        else:
            I = self.areaInertia_y
            W = self.resistance_y

        bearings = []

        for i, node in enumerate(setOfNodes):
            n = beam.SymNode(
                subscript=i + 1,
                position=node[0],
                inertia=I,
                young=self.youngModulus
            )
            dF = -9.81 * self.density * 1e-9 * self.area
            match node[1]:
                case "boundary":
                    bearings.append(
                        n.set_as_boundarybearing().set_force(node[2][1] * self.K_A, node[2][2] * self.K_A).set_load(
                            node[2][0] * self.K_A + dF)
                    )
                case "mid":
                    bearings.append(
                        n.set_as_midbearing().set_force(node[2][1] * self.K_A, node[2][2] * self.K_A).set_load(
                            node[2][0] * self.K_A + dF)
                    )
                case "loose":
                    bearings.append(
                        n.set_as_loose().set_force(node[2][1] * self.K_A, node[2][2] * self.K_A).set_load(
                            node[2][0] * self.K_A + dF)
                    )
                case "rigid":
                    bearings.append(
                        n.set_as_rigid().set_force(node[2][1] * self.K_A, node[2][2] * self.K_A).set_load(
                            node[2][0] * self.K_A + dF)
                    )
                case "glider":
                    bearings.append(
                        n.set_as_glider().set_force(node[2][1] * self.K_A, node[2][2] * self.K_A).set_load(
                            node[2][0] * self.K_A + dF)
                    )
                case "load":
                    bearings.append(
                        n.set_force(node[2][1] * self.K_A, node[2][2] * self.K_A).set_load(node[2][0] * self.K_A + dF)
                    )

        elements = []
        for i, b in enumerate(bearings[:-1]):
            elements.append(
                beam.SubSegment(b, bearings[i + 1], i + 1)
            )
        system = beam.BeamSystem(elements, bearings)

        system.solve_system(show=show_eqs)
        self.values = system.plot_system(show=show_plot)
        self.positions = np.array(self.values[2])
        self.forces = np.array(self.values[0][0])
        self.moments = np.array(self.values[0][1])
        i_xi = np.argmax(np.abs(self.moments))
        self.Mbmax = abs(self.moments[i_xi])
        self.xi = self.positions[i_xi]

        self.bearingforces = []
        for b in self.setOfNodes:
            pos = b[0]
            closest = np.argmin(np.abs(self.positions - pos))
            if np.abs(self.positions[closest] - pos) < tol:
                f = self.forces[closest]
                self.bearingforces.append(f)
        self.sigmaB = self.Mbmax / W
        self.safety = self.Rp02 / self.sigmaB
        return self.values, self.xi, self.bearingforces

    def markdown(self, title):
        return (
            " ### Berechnungsresultat Profildurchbiegung " " \n "
            " *** " "\n"
            f" {title} " " \n "
            r" $ M_{b,max} = E I \frac{\delta W(x_i)}{\delta x} = " f" {round(self.Mbmax)} Nmm $ " " \n "
            r" $ \sigma_b = \frac{M_{b,max}}{W} = " f" {round(self.sigmaB, 1)} MPa $ " " \n "
            r" $ S_F = \frac{R_{p0.2}}{\sigma_b} = " f" {round(self.safety, 1)} $ " " \n "
            " #### Querkräfte "  " \n "
            " *** " " \n "
            f" $ F_A, F_B, ..., F_Z = {list(np.round(np.array(self.bearingforces), 1))} N $ " " \n "
        )


class Alu40x40L(BoschProfile):
    def __init__(self, length: float):
        super().__init__(
            length=length, area=5.6e2,
            resistance_x=4.5e3, resistance_y=4.5e3,
            areaInertia_x=9.1e4, areaInertia_y=9.1e4,
            youngModulus=70000, Rp02=195, density=2700
        )


class Alu40x80L(BoschProfile):
    def __init__(self, length: float):
        super().__init__(
            length, area=9.9e2,
            resistance_x=15.9e3, resistance_y=8.7e3,
            areaInertia_x=63.4e4, areaInertia_y=17.3e4,
            youngModulus=70000, Rp02=195, density=2700)


class Alu40x80x80L(BoschProfile):
    def __init__(self, length: float):
        super().__init__(
            length, area=15.4e2,
            resistance_x=24.2e3, resistance_y=24.2e3,
            areaInertia_x=96.6e4, areaInertia_y=96.6e4,
            youngModulus=70000, Rp02=195, density=2700)


class Alu80x80L(BoschProfile):
    def __init__(self, length):
        super().__init__(
            length, area=18.2e2,
            resistance_x=33e3, resistance_y=33e3,
            areaInertia_x=132.1e4, areaInertia_y=132.1e4,
            youngModulus=70000, Rp02=195, density=2700)