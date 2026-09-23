import dataclasses
import configparser
import numpy as np
from typing import List, ClassVar, Dict

_rhoc = 2.77536627e11
_km_in_Mpc = 3.08568e19
_sec_in_year = 60 * 60 * 24 * 365.25

# fmt: off
# OuterRim sims
_outerrim_analysis_steps = [
        42, 43, 44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 59, 60,
        62, 63, 65, 67, 68, 70, 72, 74, 76, 77, 79, 81, 84, 86, 88,
        90, 92, 95, 97, 100, 102, 105, 107, 110, 113, 116, 119, 121,
        124, 127, 131, 134, 137, 141, 144, 148, 151, 155, 159, 163, 167,
        171, 176, 180, 184, 189, 194, 198, 203, 208, 213, 219, 224, 230,
        235, 241, 247, 253, 266, 272, 279, 286, 293, 300, 307, 315,
        323, 331, 338, 355, 365, 373, 382, 392, 401, 411, 421, 432,
        442, 453, 464, 475, 487, 499
    ]

_alphaq_analysis_steps = [
        44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 59, 60,
        62, 63, 65, 67, 68, 70, 72, 74, 76, 77, 79, 81, 84, 86, 88,
        90, 92, 95, 97, 100, 102, 105, 107, 110, 113, 116, 119, 121,
        124, 127, 131, 134, 137, 141, 144, 148, 151, 155, 159, 163, 167,
        171, 176, 180, 184, 189, 194, 198, 203, 208, 213, 219, 224, 230,
        235, 241, 247, 253, 259, 266, 272, 279, 286, 293, 300, 307, 315,
        323, 331, 338, 347, 355, 365, 373, 382, 392, 401, 411, 421, 432,
        442, 453, 464, 475, 487, 499
    ]

# Last Journey and Farpoint sims
_ljfp_analysis_steps = [
        42, 43, 44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 59, 60,
        62, 63, 65, 67, 68, 70, 72, 74, 76, 77, 79, 81, 84, 86, 88,
        90, 92, 95, 97, 100, 102, 105, 107, 110, 113, 116, 119, 121,
        124, 127, 131, 134, 137, 141, 144, 148, 151, 155, 159, 163, 167,
        171, 176, 180, 184, 189, 194, 198, 203, 208, 213, 219, 224, 230,
        235, 241, 247, 253, 259, 266, 272, 279, 286, 293, 300, 307, 315,
        323, 331, 338, 347, 355, 365, 373, 382, 392, 401, 411, 421, 432,
        442, 453, 464, 475, 487, 499]

# Last Journey and Farpoint sims
_borgcube_analysis_steps = [
        46, 48, 49, 50, 52, 53, 54, 56, 57, 59, 60,
        62, 63, 65, 67, 68, 70, 72, 74, 76, 77, 79, 81, 84, 86, 88,
        90, 92, 95, 97, 100, 102, 105, 107, 110, 113, 116, 119, 121,
        124, 127, 131, 134, 137, 141, 144, 148, 151, 155, 159, 163, 167,
        171, 176, 180, 184, 189, 194, 198, 203, 208, 213, 219, 224, 230,
        235, 241, 247, 253, 259, 266, 272, 279, 286, 293, 300, 307, 315,
        323, 331, 338, 347, 355, 365, 373, 382, 392, 401, 411, 421, 432,
        442, 453, 464, 475, 487, 499]

#Discovery sims
_discovery_analysis_steps = [
        42, 43, 44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 59, 60,
        62, 63, 65, 67, 68, 70, 72, 74, 76, 77, 79, 81, 84, 86, 88,
        90, 92, 95, 97, 100, 102, 105, 107, 110, 113, 116, 119, 121,
        124, 127, 131, 134, 137, 141, 144, 148, 151, 155, 159, 163, 167,
        171, 176, 180, 184, 189, 194, 198, 203, 208, 213, 219, 224, 230,
        235, 241, 247, 253, 259, 266, 272, 279, 286, 293, 300, 307, 315,
        323, 331, 338, 347, 355, 365, 373, 382, 392, 401, 411, 421, 432,
        442, 453, 464, 475, 487, 499]


# fmt: on


@dataclasses.dataclass(frozen=True)
class Cosmology:
    name: str
    Omega_m: float
    Omega_b: float
    Omega_L: float
    h: float
    ns: float
    s8: float
    w0: float = -1.0
    wa: float = 0.0

    cosmologies: ClassVar[Dict] = {}

    def __post_init__(self):
        Cosmology.cosmologies[self.name] = self

    def __repr__(self):
        return f"Cosmology({self.name})"

    @property
    def hubble_time(self):
        """Hubble time in Gyr (1/H0)"""
        return 1 / (100 * self.h) * _km_in_Mpc / _sec_in_year * 1e-9

    # from Linder 2003 et al. (removed curvature)
    def hubble_parameter(self, a):
        """redshift dependend Hubble parameter, H(a)"""
        if self.w0 == -1.0 and self.wa == 0.0:
            w_term = 1.0
        else:
            w_term = a ** (-3 * (1 + self.w0 + self.wa)) * np.exp(
                -3.0 * self.wa * (1 - a)
            )

        return 100 * self.h * np.sqrt(self.Omega_m * a**-3 + self.Omega_L * w_term)

    def lookback_time(self, a):
        """Lookback time in Gyr from a=1"""
        # Integrate 1/(a'*H(a')) da' from a to 1
        # TODO: add radiation / neutrinos
        # the dark-energy term is a function of the integration variable (it used to be
        # evaluated once at the lower bound, wrong for w0 != -1 or wa != 0)
        if self.w0 == -1.0 and self.wa == 0.0:
            w_term = lambda a: 1.0
        else:
            w_term = lambda a: a ** (-3 * (1 + self.w0 + self.wa)) * np.exp(
                -3.0 * self.wa * (1 - a)
            )
        integrand = lambda a: (self.Omega_m / a + self.Omega_L * a**2 * w_term(a)) ** (
            -0.5
        )
        da = 1e-3
        _a = np.linspace(a, 1, int(np.max((1 - a) / da)))
        return self.hubble_time * np.trapz(integrand(_a), _a, axis=0)

    def virial_overdensity(self, a):
        """Bryan & Norman Delta_vir"""
        x = self.func_Omega_m(a) - 1
        return 18 * np.pi**2 + 82 * x - 39 * x**2

    def func_E2(self, a):
        if self.w0 == -1.0 and self.wa == 0.0:
            w_term = 1.0
        else:
            w_term = a ** (-3 * (1 + self.w0 + self.wa)) * np.exp(
                -3.0 * self.wa * (1 - a)
            )
        return self.Omega_m * a**-3 + self.Omega_L * w_term

    def func_Omega_m(self, a):
        return self.Omega_m * a**-3 / self.func_E2(a)


@dataclasses.dataclass(frozen=True)
class Simulation:
    name: str
    nsteps: int
    zstart: float
    zfin: float
    rl: float
    ng: int
    np: int
    cosmo: Cosmology
    cosmotools_steps: List[int]
    fullalive_steps: List[int]

    simulations: ClassVar[Dict] = {}

    def __post_init__(self):
        Simulation.simulations[self.name] = self

    def __repr__(self):
        return f"Simulation({self.name}: RL={self.rl} NP={self.np}, NG={self.ng})"

    def step2a(self, step):
        aini = 1 / (self.zstart + 1)
        afin = 1 / (self.zfin + 1)
        return aini + (afin - aini) / self.nsteps * (step + 1)

    def step2z(self, step):
        return 1 / self.step2a(step) - 1

    def step2lookback(self, step):
        a = self.step2a(step)
        return self.cosmo.lookback_time(a)

    @property
    def particle_mass(self):
        return _rhoc * self.cosmo.Omega_m * (self.rl / self.np) ** 3

    @classmethod
    def parse_config(cls, config_path: str) -> "Simulation":
        config = configparser.ConfigParser()
        config.read(config_path)
        cosmology_name = config.get("cosmology", "name")
        cosmology = Cosmology(
            name=cosmology_name,
            Omega_m=float(config.get("cosmology", "Omega_m")),
            Omega_b=float(config.get("cosmology", "Omega_b")),
            Omega_L=float(config.get("cosmology", "Omega_L")),
            h=float(config.get("cosmology", "h")),
            ns=float(config.get("cosmology", "ns")),
            s8=float(config.get("cosmology", "s8")),
        )

        simulation_name = config.get("simulation", "name")
        simulation = Simulation(
            name=simulation_name,
            cosmo=cosmology,
            rl=float(config.get("simulation", "rl")),
            ng=int(config.get("simulation", "ng")),
            np=int(config.get("simulation", "np")),
            nsteps=int(config.get("simulation", "nsteps")),
            zstart=float(config.get("simulation", "zstart")),
            zfin=float(config.get("simulation", "zfin")),
            cosmotools_steps=[
                int(s) for s in config.get("simulation", "cosmotools_steps").split()
            ],
            fullalive_steps=[],
        )
        return simulation


# Cosmological parameters used for OuterRim
OuterRimCosmo = Cosmology(
    "OuterRimCosmo",
    Omega_m=0.22 + 0.02258 / 0.71**2,
    Omega_b=0.02258 / 0.71**2,
    Omega_L=1 - 0.22 - 0.02258 / 0.71**2,
    h=0.71,
    ns=0.963,
    s8=0.8,
)


# Cosmological parameters used for LJ
LastJourneyCosmo = Cosmology(
    "LastJourneyCosmo",
    Omega_m=0.26067 + 0.02242 / 0.6766**2,
    Omega_b=0.02242 / 0.6766**2,
    Omega_L=1 - 0.26067 - 0.02242 / 0.6766**2,
    h=0.6766,
    ns=0.9665,
    s8=0.8102,
    w0=-1.0,
    wa=0,
)

DiscoveryW0WACosmo = Cosmology(
    "DiscoveryW0WACosmo",
    Omega_m=0.2903 + 0.0224 / 0.6466**2,
    Omega_b=0.0224 / 0.6466**2,
    Omega_L=1 - 0.2903 - 0.0224 / 0.6466**2,
    h=0.6466,
    ns=0.965,
    s8=0.7917,
    w0=-0.45,
    wa=-1.79,
)

DiscoveryLCDMCosmo = Cosmology(
    "DiscoveryLCDMCosmo",
    Omega_m=0.2582 + 0.0225 / 0.6797**2,
    Omega_b=0.0225 / 0.6797**2,
    Omega_L=1 - 0.2582 - 0.0225 / 0.6797**2,
    h=0.6797,
    ns=0.968,
    s8=0.8135,
)

AlphaQ = Simulation(
    name="AlphaQ",
    cosmo=OuterRimCosmo,
    rl=256,
    ng=1024,
    np=1024,
    nsteps=500,
    zstart=200,
    zfin=0.0,
    cosmotools_steps=_alphaq_analysis_steps,
    fullalive_steps=_ljfp_analysis_steps,
)


OuterRim = Simulation(
    name="OuterRim",
    cosmo=OuterRimCosmo,
    rl=3000.0,
    ng=10240,
    np=10240,
    nsteps=500,
    zstart=200.0,
    zfin=0.0,
    cosmotools_steps=_outerrim_analysis_steps,
    fullalive_steps=_outerrim_analysis_steps,
)


LastJourney = Simulation(
    name="LastJourney",
    cosmo=LastJourneyCosmo,
    rl=3400,
    ng=10752,
    np=10752,
    nsteps=500,
    zstart=200.0,
    zfin=0.0,
    cosmotools_steps=_ljfp_analysis_steps,
    fullalive_steps=_ljfp_analysis_steps,
)

LastJourneySV = dataclasses.replace(
    LastJourney, name="LastJourneySV", rl=250, ng=1024, np=1024
)

Farpoint = Simulation(
    name="Farpoint",
    cosmo=LastJourneyCosmo,
    rl=1000,
    ng=12288,
    np=12288,
    nsteps=500,
    zstart=200.0,
    zfin=0.0,
    cosmotools_steps=_ljfp_analysis_steps,
    fullalive_steps=_ljfp_analysis_steps,
)


FarpointSV = dataclasses.replace(
    Farpoint,
    name="FarpointSV",
    rl=250,
    ng=3072,
    np=3072,
    cosmotools_steps=_ljfp_analysis_steps[1:],
)

BorgCube = Simulation(
    name="BorgCube",
    cosmo=OuterRimCosmo,
    rl=800,
    ng=2304,
    np=2304,
    nsteps=500,
    zstart=200.0,
    zfin=0.0,
    cosmotools_steps=_borgcube_analysis_steps,
    fullalive_steps=_ljfp_analysis_steps,
)

DiscoveryLCDM = Simulation(
    name="DiscoveryLCDM",
    cosmo=DiscoveryLCDMCosmo,
    rl=1019.55,
    ng=6720,
    np=6720,
    nsteps=500,
    zstart=200.0,
    zfin=0.0,
    cosmotools_steps=_discovery_analysis_steps,
    fullalive_steps=_discovery_analysis_steps,
)

DiscoveryW0WA = Simulation(
    name="DiscoveryW0WA",
    cosmo=DiscoveryW0WACosmo,
    rl=969.9,
    ng=6720,
    np=6720,
    nsteps=500,
    zstart=200.0,
    zfin=0.0,
    cosmotools_steps=_discovery_analysis_steps,
    fullalive_steps=_discovery_analysis_steps,
)
