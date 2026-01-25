"""
Capability Detection and Routing

Determines what physics can be computed from available data.
Routes data to appropriate engines.
Reports what's possible and what's missing.

The PRISM Curriculum:
    Level 0: Raw time series        → Statistics, entropy, memory
    Level 1: Labeled signals        → Derivatives, specific energy
    Level 2: Physical constants     → Real energy, momentum
    Level 3: Related signals        → Gibbs, transfer functions
    Level 4: Spatial fields         → Navier-Stokes, Maxwell

You bring the data. We compute everything possible.
You bring more context. We unlock more physics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from enum import Enum, auto


# =============================================================================
# DATA LEVELS
# =============================================================================

class DataLevel(Enum):
    """
    What level of data context do we have?

    Each level unlocks more physics capabilities.
    """
    RAW_TIMESERIES = 0      # Just numbers in columns
    LABELED_SIGNALS = 1     # Know what signals represent physically
    WITH_CONSTANTS = 2      # Have physical constants (mass, k, etc.)
    RELATED_SIGNALS = 3     # Multiple signals with known relationships
    SPATIAL_FIELD = 4       # Full 3D/4D field data


# =============================================================================
# CAPABILITIES
# =============================================================================

class Capability(Enum):
    """
    What can PRISM compute?

    Organized by the data level required to unlock each capability.
    """
    # ─────────────────────────────────────────────────────────────────────────
    # LEVEL 0: Always available (raw time series)
    # ─────────────────────────────────────────────────────────────────────────
    STATISTICS = auto()           # mean, std, skew, kurtosis
    DISTRIBUTION = auto()         # distribution fitting
    STATIONARITY = auto()         # ADF, KPSS tests
    ENTROPY = auto()              # sample, permutation, spectral entropy
    MEMORY = auto()               # Hurst exponent (R/S, DFA)
    SPECTRAL = auto()             # FFT, wavelets, power spectrum
    RECURRENCE = auto()           # RQA metrics
    CHAOS = auto()                # Lyapunov exponent
    VOLATILITY = auto()           # realized vol, GARCH
    EVENT_DETECTION = auto()      # step and spike detection
    HEAVISIDE_DIRAC = auto()      # H(t-t₀), δ(t-t₀) at events
    GEOMETRY = auto()             # PCA, manifold, clustering
    DYNAMICS = auto()             # trajectory, regimes, hd_slope

    # ─────────────────────────────────────────────────────────────────────────
    # LEVEL 1: With signal labels (know what x represents)
    # ─────────────────────────────────────────────────────────────────────────
    DERIVATIVES = auto()          # dx/dt, d²x/dt² with proper units
    SPECIFIC_KINETIC = auto()     # T/m = ½v² [J/kg]
    SPECIFIC_POTENTIAL = auto()   # V/k = ½x² [m²]
    SPECIFIC_HAMILTONIAN = auto() # H = T/m + V/k (unitless proxy)

    # ─────────────────────────────────────────────────────────────────────────
    # LEVEL 2: With physical constants
    # ─────────────────────────────────────────────────────────────────────────
    KINETIC_ENERGY = auto()       # T = ½mv² [J]
    POTENTIAL_ENERGY = auto()     # V = ½kx² [J]
    MOMENTUM = auto()             # p = mv [kg·m/s]
    HAMILTONIAN = auto()          # H = T + V [J]
    LAGRANGIAN = auto()           # L = T - V [J]
    ROTATIONAL_KE = auto()        # T = ½Iω² [J]

    # ─────────────────────────────────────────────────────────────────────────
    # LEVEL 3: With related signals
    # ─────────────────────────────────────────────────────────────────────────
    WORK = auto()                 # W = ∫F·dx [J]
    POWER = auto()                # P = F·v [W]
    ANGULAR_MOMENTUM = auto()     # L = r × p [kg·m²/s]
    PHASE_SPACE = auto()          # (q, p) trajectories
    ENERGY_CONSERVATION = auto()  # dH/dt ≈ 0 check

    GIBBS_FREE_ENERGY = auto()    # G = H - TS [J]
    ENTHALPY = auto()             # H = U + PV [J]
    ENTROPY_THERMO = auto()       # S from T, P, V [J/K]
    CHEMICAL_POTENTIAL = auto()   # μ = ∂G/∂n [J/mol]

    TRANSFER_FUNCTION = auto()    # G(s) from step/impulse response
    FREQUENCY_RESPONSE = auto()   # Bode plot, bandwidth
    POLES_ZEROS = auto()          # Stability analysis
    IMPULSE_RESPONSE = auto()     # h(t) = L⁻¹{G(s)}

    GRANGER_CAUSALITY = auto()    # Causal relationships
    TRANSFER_ENTROPY = auto()     # Directed information flow

    # ─────────────────────────────────────────────────────────────────────────
    # LEVEL 4: Spatial field data
    # ─────────────────────────────────────────────────────────────────────────
    VORTICITY = auto()            # ω = ∇ × v
    STRAIN_TENSOR = auto()        # S_ij = ½(∂vᵢ/∂xⱼ + ∂vⱼ/∂xᵢ)
    Q_CRITERION = auto()          # Q = ½(||Ω||² - ||S||²)
    TURBULENT_KE = auto()         # k = ½⟨u'ᵢu'ᵢ⟩
    DISSIPATION = auto()          # ε = 2ν⟨SᵢⱼSᵢⱼ⟩
    ENERGY_SPECTRUM = auto()      # E(k) with k⁻⁵/³ check
    REYNOLDS_NUMBER = auto()      # Re = UL/ν
    KOLMOGOROV_SCALES = auto()    # η, τ_η, v_η

    HEAT_FLUX = auto()            # q = -k∇T
    LAPLACIAN_T = auto()          # ∇²T

    MAXWELL_DIV_E = auto()        # ∇·E = ρ/ε₀
    MAXWELL_CURL_B = auto()       # ∇×B = μ₀J + μ₀ε₀∂E/∂t
    POYNTING = auto()             # S = E × B / μ₀

    # ─────────────────────────────────────────────────────────────────────────
    # PIPE FLOW: With pipe geometry
    # ─────────────────────────────────────────────────────────────────────────
    PIPE_REYNOLDS = auto()        # Re = vD/ν (per-pipe)
    FLOW_REGIME = auto()          # Laminar/Transitional/Turbulent
    FRICTION_FACTOR = auto()      # f (Darcy-Weisbach)
    PRESSURE_DROP = auto()        # ΔP = f(L/D)(ρv²/2)
    HEAD_LOSS = auto()            # h_f = ΔP/(ρg)
    PIPE_POWER_LOSS = auto()      # P = ΔP × Q


# =============================================================================
# PHYSICAL QUANTITIES
# =============================================================================

class PhysicalQuantity(Enum):
    """Known physical quantities that signals can represent."""
    # Mechanical
    POSITION = "position"
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"
    FORCE = "force"
    ANGULAR_POSITION = "angular_position"
    ANGULAR_VELOCITY = "angular_velocity"

    # Thermodynamic
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    VOLUME = "volume"
    HEAT = "heat"

    # Electrical
    VOLTAGE = "voltage"
    CURRENT = "current"

    # Field components
    VELOCITY_X = "velocity_x"
    VELOCITY_Y = "velocity_y"
    VELOCITY_Z = "velocity_z"

    # Control
    INPUT = "input"
    OUTPUT = "output"

    # Generic
    UNKNOWN = "unknown"


# =============================================================================
# UNITS - Full Unit Algebra System
# =============================================================================

# Standard units for physical quantities (SI)
STANDARD_UNITS = {
    PhysicalQuantity.POSITION: "m",
    PhysicalQuantity.VELOCITY: "m/s",
    PhysicalQuantity.ACCELERATION: "m/s²",
    PhysicalQuantity.FORCE: "N",
    PhysicalQuantity.ANGULAR_POSITION: "rad",
    PhysicalQuantity.ANGULAR_VELOCITY: "rad/s",
    PhysicalQuantity.TEMPERATURE: "K",
    PhysicalQuantity.PRESSURE: "Pa",
    PhysicalQuantity.VOLUME: "m³",
    PhysicalQuantity.VOLTAGE: "V",
    PhysicalQuantity.CURRENT: "A",
}


@dataclass
class UnitSpec:
    """
    Unit specification with dimensional analysis.

    Tracks base SI dimensions: [M, L, T, Θ, I, N, J]
    - M: mass (kg)
    - L: length (m)
    - T: time (s)
    - Θ: temperature (K)
    - I: electric current (A)
    - N: amount of substance (mol)
    - J: luminous intensity (cd)

    Examples:
        velocity = UnitSpec.parse("m/s")       # [0, 1, -1, 0, 0, 0, 0]
        force = UnitSpec.parse("N")            # [1, 1, -2, 0, 0, 0, 0] = kg⋅m/s²
        energy = velocity * velocity * mass    # [1, 2, -2, 0, 0, 0, 0] = J
    """
    # Dimensional exponents [M, L, T, Θ, I, N, J]
    dimensions: Tuple[int, int, int, int, int, int, int]
    symbol: str = ""  # Display symbol (e.g., "m/s", "N", "J")

    # Common unit conversions to SI
    scale: float = 1.0  # Multiplier to convert to SI base units
    offset: float = 0.0  # Offset for temperature conversions (C, F)

    def __post_init__(self):
        if isinstance(self.dimensions, list):
            self.dimensions = tuple(self.dimensions)

    @classmethod
    def dimensionless(cls) -> 'UnitSpec':
        """Return dimensionless unit (pure number)."""
        return cls(dimensions=(0, 0, 0, 0, 0, 0, 0), symbol="")

    @classmethod
    def parse(cls, unit_str: str) -> 'UnitSpec':
        """
        Parse unit string into UnitSpec.

        Supports common formats:
            SI: m, kg, s, K, A, mol, cd
            Derived: m/s, m/s², N, J, W, Pa, Hz
            Imperial: ft, lb, °F, °C
            Compound: kg⋅m/s², N⋅m, J/K
        """
        if not unit_str or unit_str.lower() in ('', 'none', 'unitless', '1'):
            return cls.dimensionless()

        # Normalize input
        unit_str = unit_str.strip()

        # Check known units first
        known = KNOWN_UNITS.get(unit_str)
        if known:
            return known

        # Try to parse compound units
        return cls._parse_compound(unit_str)

    @classmethod
    def _parse_compound(cls, unit_str: str) -> 'UnitSpec':
        """Parse compound unit like kg⋅m/s² or N⋅m."""
        # Split by / for numerator/denominator
        if '/' in unit_str:
            parts = unit_str.split('/')
            if len(parts) == 2:
                num = cls.parse(parts[0]) if parts[0] else cls.dimensionless()
                den = cls.parse(parts[1])
                return num / den

        # Split by ⋅ or * for multiplication
        for sep in ['⋅', '*', '·']:
            if sep in unit_str:
                parts = unit_str.split(sep)
                result = cls.dimensionless()
                for part in parts:
                    result = result * cls.parse(part.strip())
                return result

        # Handle powers like s², m³
        for i, char in enumerate(unit_str):
            if char in '²³⁴⁵⁶⁷⁸⁹' or char.isdigit():
                base = unit_str[:i]
                exp_str = unit_str[i:]
                # Convert superscript to number
                exp = _parse_exponent(exp_str)
                base_unit = cls.parse(base)
                return base_unit ** exp

        # Unknown unit - return dimensionless with warning
        return cls(dimensions=(0, 0, 0, 0, 0, 0, 0), symbol=f"?{unit_str}")

    def __mul__(self, other: 'UnitSpec') -> 'UnitSpec':
        """Multiply units: m * s⁻¹ = m/s"""
        new_dims = tuple(a + b for a, b in zip(self.dimensions, other.dimensions))
        new_symbol = f"{self.symbol}⋅{other.symbol}" if self.symbol and other.symbol else (self.symbol or other.symbol)
        return UnitSpec(dimensions=new_dims, symbol=new_symbol, scale=self.scale * other.scale)

    def __truediv__(self, other: 'UnitSpec') -> 'UnitSpec':
        """Divide units: m / s = m/s"""
        new_dims = tuple(a - b for a, b in zip(self.dimensions, other.dimensions))
        new_symbol = f"{self.symbol}/{other.symbol}" if other.symbol else self.symbol
        return UnitSpec(dimensions=new_dims, symbol=new_symbol, scale=self.scale / other.scale)

    def __pow__(self, exp: int) -> 'UnitSpec':
        """Raise unit to power: (m/s)² = m²/s²"""
        new_dims = tuple(d * exp for d in self.dimensions)
        if exp == 2:
            new_symbol = f"{self.symbol}²"
        elif exp == 3:
            new_symbol = f"{self.symbol}³"
        elif exp == -1:
            new_symbol = f"1/{self.symbol}"
        else:
            new_symbol = f"({self.symbol})^{exp}"
        return UnitSpec(dimensions=new_dims, symbol=new_symbol, scale=self.scale ** exp)

    def __eq__(self, other: 'UnitSpec') -> bool:
        """Units are equal if dimensions match."""
        return self.dimensions == other.dimensions

    def is_compatible(self, other: 'UnitSpec') -> bool:
        """Check if units are dimensionally compatible (can be added/compared)."""
        return self.dimensions == other.dimensions

    def is_dimensionless(self) -> bool:
        """Check if unit is dimensionless."""
        return all(d == 0 for d in self.dimensions)

    def to_si_symbol(self) -> str:
        """Get standard SI symbol for the dimensions."""
        # Check if matches known derived unit
        for symbol, unit in KNOWN_UNITS.items():
            if unit.dimensions == self.dimensions:
                return symbol

        # Build from base units
        M, L, T, Θ, I, N, J = self.dimensions
        parts_num = []
        parts_den = []

        base_symbols = [('kg', M), ('m', L), ('s', T), ('K', Θ), ('A', I), ('mol', N), ('cd', J)]

        for sym, exp in base_symbols:
            if exp > 0:
                if exp == 1:
                    parts_num.append(sym)
                else:
                    parts_num.append(f"{sym}{'²' if exp == 2 else '³' if exp == 3 else f'^{exp}'}")
            elif exp < 0:
                exp = -exp
                if exp == 1:
                    parts_den.append(sym)
                else:
                    parts_den.append(f"{sym}{'²' if exp == 2 else '³' if exp == 3 else f'^{exp}'}")

        if not parts_num and not parts_den:
            return ""
        elif not parts_den:
            return "⋅".join(parts_num)
        elif not parts_num:
            return "1/" + "⋅".join(parts_den)
        else:
            return "⋅".join(parts_num) + "/" + "⋅".join(parts_den)

    def __str__(self) -> str:
        return self.symbol or self.to_si_symbol() or "dimensionless"

    def __repr__(self) -> str:
        return f"UnitSpec({self.dimensions}, '{self.symbol}')"


def _parse_exponent(s: str) -> int:
    """Parse exponent string like ², ³, ^2, 2."""
    superscript_map = {'²': 2, '³': 3, '⁴': 4, '⁵': 5, '⁶': 6, '⁷': 7, '⁸': 8, '⁹': 9, '⁰': 0, '¹': 1}

    if s in superscript_map:
        return superscript_map[s]

    if s.startswith('^'):
        return int(s[1:])

    # Handle negative exponents
    if s.startswith('⁻'):
        return -_parse_exponent(s[1:])

    try:
        return int(s)
    except ValueError:
        return 1


# =============================================================================
# KNOWN UNITS REGISTRY
# =============================================================================

# Dimensions: [M, L, T, Θ, I, N, J]
#              kg  m  s  K  A mol cd

KNOWN_UNITS: Dict[str, UnitSpec] = {
    # Base SI units
    "kg": UnitSpec((1, 0, 0, 0, 0, 0, 0), "kg"),
    "m": UnitSpec((0, 1, 0, 0, 0, 0, 0), "m"),
    "s": UnitSpec((0, 0, 1, 0, 0, 0, 0), "s"),
    "K": UnitSpec((0, 0, 0, 1, 0, 0, 0), "K"),
    "A": UnitSpec((0, 0, 0, 0, 1, 0, 0), "A"),
    "mol": UnitSpec((0, 0, 0, 0, 0, 1, 0), "mol"),
    "cd": UnitSpec((0, 0, 0, 0, 0, 0, 1), "cd"),

    # Derived SI units
    "N": UnitSpec((1, 1, -2, 0, 0, 0, 0), "N"),           # Newton = kg⋅m/s²
    "J": UnitSpec((1, 2, -2, 0, 0, 0, 0), "J"),           # Joule = kg⋅m²/s²
    "W": UnitSpec((1, 2, -3, 0, 0, 0, 0), "W"),           # Watt = kg⋅m²/s³
    "Pa": UnitSpec((1, -1, -2, 0, 0, 0, 0), "Pa"),        # Pascal = kg/(m⋅s²)
    "Hz": UnitSpec((0, 0, -1, 0, 0, 0, 0), "Hz"),         # Hertz = 1/s
    "V": UnitSpec((1, 2, -3, 0, -1, 0, 0), "V"),          # Volt = kg⋅m²/(A⋅s³)
    "Ω": UnitSpec((1, 2, -3, 0, -2, 0, 0), "Ω"),          # Ohm = kg⋅m²/(A²⋅s³)
    "C": UnitSpec((0, 0, 1, 0, 1, 0, 0), "C"),            # Coulomb = A⋅s
    "F": UnitSpec((-1, -2, 4, 0, 2, 0, 0), "F"),          # Farad = A²⋅s⁴/(kg⋅m²)
    "T": UnitSpec((1, 0, -2, 0, -1, 0, 0), "T"),          # Tesla = kg/(A⋅s²)

    # Common velocity units
    "m/s": UnitSpec((0, 1, -1, 0, 0, 0, 0), "m/s"),
    "km/h": UnitSpec((0, 1, -1, 0, 0, 0, 0), "km/h", scale=1/3.6),
    "mph": UnitSpec((0, 1, -1, 0, 0, 0, 0), "mph", scale=0.44704),
    "ft/s": UnitSpec((0, 1, -1, 0, 0, 0, 0), "ft/s", scale=0.3048),

    # Acceleration
    "m/s²": UnitSpec((0, 1, -2, 0, 0, 0, 0), "m/s²"),
    "m/s2": UnitSpec((0, 1, -2, 0, 0, 0, 0), "m/s²"),     # ASCII variant
    "ft/s²": UnitSpec((0, 1, -2, 0, 0, 0, 0), "ft/s²", scale=0.3048),
    "g": UnitSpec((0, 1, -2, 0, 0, 0, 0), "g", scale=9.80665),  # Standard gravity

    # Length variants
    "km": UnitSpec((0, 1, 0, 0, 0, 0, 0), "km", scale=1000),
    "cm": UnitSpec((0, 1, 0, 0, 0, 0, 0), "cm", scale=0.01),
    "mm": UnitSpec((0, 1, 0, 0, 0, 0, 0), "mm", scale=0.001),
    "ft": UnitSpec((0, 1, 0, 0, 0, 0, 0), "ft", scale=0.3048),
    "in": UnitSpec((0, 1, 0, 0, 0, 0, 0), "in", scale=0.0254),
    "mi": UnitSpec((0, 1, 0, 0, 0, 0, 0), "mi", scale=1609.344),

    # Mass variants
    "g": UnitSpec((1, 0, 0, 0, 0, 0, 0), "g", scale=0.001),
    "lb": UnitSpec((1, 0, 0, 0, 0, 0, 0), "lb", scale=0.453592),
    "lbm": UnitSpec((1, 0, 0, 0, 0, 0, 0), "lbm", scale=0.453592),

    # Force variants
    "lbf": UnitSpec((1, 1, -2, 0, 0, 0, 0), "lbf", scale=4.44822),
    "kN": UnitSpec((1, 1, -2, 0, 0, 0, 0), "kN", scale=1000),
    "dyn": UnitSpec((1, 1, -2, 0, 0, 0, 0), "dyn", scale=1e-5),

    # Pressure variants
    "kPa": UnitSpec((1, -1, -2, 0, 0, 0, 0), "kPa", scale=1000),
    "MPa": UnitSpec((1, -1, -2, 0, 0, 0, 0), "MPa", scale=1e6),
    "bar": UnitSpec((1, -1, -2, 0, 0, 0, 0), "bar", scale=1e5),
    "atm": UnitSpec((1, -1, -2, 0, 0, 0, 0), "atm", scale=101325),
    "psi": UnitSpec((1, -1, -2, 0, 0, 0, 0), "psi", scale=6894.76),
    "mmHg": UnitSpec((1, -1, -2, 0, 0, 0, 0), "mmHg", scale=133.322),

    # Energy variants
    "kJ": UnitSpec((1, 2, -2, 0, 0, 0, 0), "kJ", scale=1000),
    "MJ": UnitSpec((1, 2, -2, 0, 0, 0, 0), "MJ", scale=1e6),
    "cal": UnitSpec((1, 2, -2, 0, 0, 0, 0), "cal", scale=4.184),
    "kcal": UnitSpec((1, 2, -2, 0, 0, 0, 0), "kcal", scale=4184),
    "BTU": UnitSpec((1, 2, -2, 0, 0, 0, 0), "BTU", scale=1055.06),
    "eV": UnitSpec((1, 2, -2, 0, 0, 0, 0), "eV", scale=1.602e-19),
    "kWh": UnitSpec((1, 2, -2, 0, 0, 0, 0), "kWh", scale=3.6e6),

    # Power variants
    "kW": UnitSpec((1, 2, -3, 0, 0, 0, 0), "kW", scale=1000),
    "MW": UnitSpec((1, 2, -3, 0, 0, 0, 0), "MW", scale=1e6),
    "hp": UnitSpec((1, 2, -3, 0, 0, 0, 0), "hp", scale=745.7),

    # Temperature (with offset for conversion)
    "°C": UnitSpec((0, 0, 0, 1, 0, 0, 0), "°C", offset=273.15),
    "C": UnitSpec((0, 0, 0, 1, 0, 0, 0), "°C", offset=273.15),
    "°F": UnitSpec((0, 0, 0, 1, 0, 0, 0), "°F", scale=5/9, offset=255.372),
    "F": UnitSpec((0, 0, 0, 1, 0, 0, 0), "°F", scale=5/9, offset=255.372),

    # Angular
    "rad": UnitSpec((0, 0, 0, 0, 0, 0, 0), "rad"),  # Dimensionless
    "deg": UnitSpec((0, 0, 0, 0, 0, 0, 0), "deg", scale=3.14159/180),
    "°": UnitSpec((0, 0, 0, 0, 0, 0, 0), "°", scale=3.14159/180),
    "rad/s": UnitSpec((0, 0, -1, 0, 0, 0, 0), "rad/s"),
    "rpm": UnitSpec((0, 0, -1, 0, 0, 0, 0), "rpm", scale=2*3.14159/60),

    # Volume
    "m³": UnitSpec((0, 3, 0, 0, 0, 0, 0), "m³"),
    "m3": UnitSpec((0, 3, 0, 0, 0, 0, 0), "m³"),
    "L": UnitSpec((0, 3, 0, 0, 0, 0, 0), "L", scale=0.001),
    "mL": UnitSpec((0, 3, 0, 0, 0, 0, 0), "mL", scale=1e-6),
    "gal": UnitSpec((0, 3, 0, 0, 0, 0, 0), "gal", scale=0.00378541),
    "ft³": UnitSpec((0, 3, 0, 0, 0, 0, 0), "ft³", scale=0.0283168),

    # Flow rate
    "m³/s": UnitSpec((0, 3, -1, 0, 0, 0, 0), "m³/s"),
    "L/s": UnitSpec((0, 3, -1, 0, 0, 0, 0), "L/s", scale=0.001),
    "L/min": UnitSpec((0, 3, -1, 0, 0, 0, 0), "L/min", scale=0.001/60),
    "gpm": UnitSpec((0, 3, -1, 0, 0, 0, 0), "gpm", scale=6.309e-5),

    # Density
    "kg/m³": UnitSpec((1, -3, 0, 0, 0, 0, 0), "kg/m³"),
    "kg/m3": UnitSpec((1, -3, 0, 0, 0, 0, 0), "kg/m³"),
    "g/cm³": UnitSpec((1, -3, 0, 0, 0, 0, 0), "g/cm³", scale=1000),
    "lb/ft³": UnitSpec((1, -3, 0, 0, 0, 0, 0), "lb/ft³", scale=16.0185),

    # Viscosity
    "Pa⋅s": UnitSpec((1, -1, -1, 0, 0, 0, 0), "Pa⋅s"),    # Dynamic viscosity
    "m²/s": UnitSpec((0, 2, -1, 0, 0, 0, 0), "m²/s"),      # Kinematic viscosity
    "m2/s": UnitSpec((0, 2, -1, 0, 0, 0, 0), "m²/s"),
    "cSt": UnitSpec((0, 2, -1, 0, 0, 0, 0), "cSt", scale=1e-6),  # Centistokes
    "cP": UnitSpec((1, -1, -1, 0, 0, 0, 0), "cP", scale=0.001),   # Centipoise

    # Specific heat / entropy
    "J/(kg⋅K)": UnitSpec((0, 2, -2, -1, 0, 0, 0), "J/(kg⋅K)"),
    "J/(mol⋅K)": UnitSpec((1, 2, -2, -1, 0, -1, 0), "J/(mol⋅K)"),
    "J/K": UnitSpec((1, 2, -2, -1, 0, 0, 0), "J/K"),

    # Momentum
    "kg⋅m/s": UnitSpec((1, 1, -1, 0, 0, 0, 0), "kg⋅m/s"),
    "N⋅s": UnitSpec((1, 1, -1, 0, 0, 0, 0), "N⋅s"),  # Impulse = momentum

    # Torque / moment
    "N⋅m": UnitSpec((1, 2, -2, 0, 0, 0, 0), "N⋅m"),  # Same dimensions as energy
    "ft⋅lb": UnitSpec((1, 2, -2, 0, 0, 0, 0), "ft⋅lb", scale=1.35582),

    # Angular momentum
    "kg⋅m²/s": UnitSpec((1, 2, -1, 0, 0, 0, 0), "kg⋅m²/s"),
}


# =============================================================================
# UNIT COMPUTATION RESULTS
# =============================================================================

@dataclass
class UnitResult:
    """
    Result of a computation with unit tracking.

    Wraps a numeric value with its unit, enabling:
    - Display with proper units
    - Conversion to other compatible units
    - Dimensional analysis validation
    """
    value: float
    unit: UnitSpec

    def to(self, target_unit: str) -> 'UnitResult':
        """Convert to another compatible unit."""
        target = UnitSpec.parse(target_unit)

        if not self.unit.is_compatible(target):
            raise ValueError(
                f"Cannot convert {self.unit} to {target}: incompatible dimensions"
            )

        # Convert through SI
        si_value = (self.value * self.unit.scale) + self.unit.offset
        new_value = (si_value - target.offset) / target.scale

        return UnitResult(value=new_value, unit=target)

    def to_si(self) -> 'UnitResult':
        """Convert to SI base units."""
        si_value = (self.value * self.unit.scale) + self.unit.offset
        si_unit = UnitSpec(
            dimensions=self.unit.dimensions,
            symbol=self.unit.to_si_symbol()
        )
        return UnitResult(value=si_value, unit=si_unit)

    def __str__(self) -> str:
        if self.unit.is_dimensionless():
            return f"{self.value:.6g}"
        return f"{self.value:.6g} {self.unit}"

    def __repr__(self) -> str:
        return f"UnitResult({self.value}, {self.unit})"


# =============================================================================
# DERIVED UNIT COMPUTATION
# =============================================================================

# Maps computation → output unit dimensions
DERIVED_UNITS = {
    # Dynamics (from geometry)
    'geodesic_deviation': lambda inputs: inputs.get('velocity', UnitSpec.dimensionless()) / UnitSpec.parse('s'),
    'acceleration': lambda inputs: inputs.get('velocity', UnitSpec.dimensionless()) / UnitSpec.parse('s'),

    # Physics
    'kinetic_energy': lambda inputs: inputs.get('mass', UnitSpec.parse('kg')) * (inputs.get('velocity', UnitSpec.parse('m/s')) ** 2),
    'potential_energy': lambda inputs: inputs.get('spring_constant', UnitSpec.parse('N/m')) * (inputs.get('position', UnitSpec.parse('m')) ** 2),
    'momentum': lambda inputs: inputs.get('mass', UnitSpec.parse('kg')) * inputs.get('velocity', UnitSpec.parse('m/s')),
    'force': lambda inputs: inputs.get('mass', UnitSpec.parse('kg')) * inputs.get('acceleration', UnitSpec.parse('m/s²')),
    'work': lambda inputs: inputs.get('force', UnitSpec.parse('N')) * inputs.get('position', UnitSpec.parse('m')),
    'power': lambda inputs: inputs.get('force', UnitSpec.parse('N')) * inputs.get('velocity', UnitSpec.parse('m/s')),

    # Thermodynamic
    'entropy': lambda inputs: UnitSpec.parse('J/K'),
    'gibbs_free_energy': lambda inputs: UnitSpec.parse('J'),
    'enthalpy': lambda inputs: UnitSpec.parse('J'),

    # Fluid
    'reynolds_number': lambda inputs: UnitSpec.dimensionless(),
    'pressure_drop': lambda inputs: UnitSpec.parse('Pa'),
    'head_loss': lambda inputs: UnitSpec.parse('m'),
    'flow_rate': lambda inputs: UnitSpec.parse('m³/s'),
}


def compute_output_unit(computation: str, input_units: Dict[str, UnitSpec]) -> UnitSpec:
    """
    Compute the output unit for a given computation based on input units.

    Args:
        computation: Name of the computation (e.g., 'kinetic_energy', 'momentum')
        input_units: Dict mapping input names to their units

    Returns:
        UnitSpec for the output
    """
    if computation in DERIVED_UNITS:
        return DERIVED_UNITS[computation](input_units)
    return UnitSpec.dimensionless()


def propagate_units(
    computation: str,
    value: float,
    input_units: Dict[str, UnitSpec],
) -> UnitResult:
    """
    Compute a value and propagate units.

    Args:
        computation: Name of the computation
        value: Numeric result (already computed)
        input_units: Dict of input signal/constant units

    Returns:
        UnitResult with value and derived unit
    """
    output_unit = compute_output_unit(computation, input_units)
    return UnitResult(value=value, unit=output_unit)


# =============================================================================
# SIGNAL SPECIFICATION
# =============================================================================

@dataclass
class SignalSpec:
    """Specification for a single signal/column."""
    name: str
    physical_quantity: PhysicalQuantity = PhysicalQuantity.UNKNOWN
    units: Optional[str] = None

    # For 3D vectors
    is_component: bool = False
    vector_name: Optional[str] = None  # e.g., "velocity" for velocity_x
    component: Optional[str] = None    # e.g., "x"

    def get_unit_spec(self) -> UnitSpec:
        """Get parsed UnitSpec for this signal."""
        if self.units:
            return UnitSpec.parse(self.units)
        elif self.physical_quantity != PhysicalQuantity.UNKNOWN:
            # Use standard unit for physical quantity
            std_unit = STANDARD_UNITS.get(self.physical_quantity)
            return UnitSpec.parse(std_unit) if std_unit else UnitSpec.dimensionless()
        else:
            return UnitSpec.dimensionless()

    def has_units(self) -> bool:
        """Check if signal has explicit units."""
        return self.units is not None or self.physical_quantity != PhysicalQuantity.UNKNOWN

    @classmethod
    def from_dict(cls, d: Dict) -> 'SignalSpec':
        qty = d.get('physical_quantity', 'unknown')
        if isinstance(qty, str):
            try:
                qty = PhysicalQuantity(qty)
            except ValueError:
                qty = PhysicalQuantity.UNKNOWN

        return cls(
            name=d['name'],
            physical_quantity=qty,
            units=d.get('units'),
            is_component=d.get('is_component', False),
            vector_name=d.get('vector_name'),
            component=d.get('component'),
        )


# =============================================================================
# CONSTANTS SPECIFICATION
# =============================================================================

@dataclass
class ConstantsSpec:
    """Physical constants provided by user."""
    # Mechanical
    mass: Optional[float] = None                    # kg
    spring_constant: Optional[float] = None         # N/m
    moment_of_inertia: Optional[float] = None       # kg·m²
    damping_coefficient: Optional[float] = None     # N·s/m

    # Thermodynamic
    Cp: Optional[float] = None                      # J/(mol·K)
    Cv: Optional[float] = None                      # J/(mol·K)
    n_moles: Optional[float] = None                 # mol

    # Fluid
    kinematic_viscosity: Optional[float] = None     # m²/s
    density: Optional[float] = None                 # kg/m³

    # Spatial
    dx: Optional[float] = None                      # m
    dy: Optional[float] = None                      # m
    dz: Optional[float] = None                      # m
    dt: Optional[float] = None                      # s

    @classmethod
    def from_dict(cls, d: Dict) -> 'ConstantsSpec':
        return cls(
            mass=d.get('mass'),
            spring_constant=d.get('spring_constant'),
            moment_of_inertia=d.get('moment_of_inertia'),
            damping_coefficient=d.get('damping_coefficient'),
            Cp=d.get('Cp'),
            Cv=d.get('Cv'),
            n_moles=d.get('n_moles'),
            kinematic_viscosity=d.get('kinematic_viscosity') or d.get('nu'),
            density=d.get('density') or d.get('rho'),
            dx=d.get('dx'),
            dy=d.get('dy'),
            dz=d.get('dz'),
            dt=d.get('dt'),
        )


# =============================================================================
# PIPE GEOMETRY SPECIFICATION
# =============================================================================

@dataclass
class PipeSpec:
    """Per-pipe geometry specification."""
    signal: str                              # Velocity signal name
    diameter: float                          # m
    length: Optional[float] = None           # m
    roughness: float = 0.0                   # m (absolute roughness)
    description: Optional[str] = None

    @property
    def area(self) -> float:
        """Cross-sectional area [m²]."""
        import math
        return math.pi * (self.diameter / 2) ** 2

    @classmethod
    def from_dict(cls, d: Dict) -> 'PipeSpec':
        return cls(
            signal=d['signal'],
            diameter=d['diameter'],
            length=d.get('length'),
            roughness=d.get('roughness', 0.0),
            description=d.get('description'),
        )


@dataclass
class PipeNetworkSpec:
    """Pipe network specification."""
    pipes: List[PipeSpec]

    # Network topology (optional)
    connections: List[Dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict) -> 'PipeNetworkSpec':
        if not d:
            return cls(pipes=[])

        pipes_raw = d.get('pipes', [])

        # Handle both list format and dict format
        pipes = []
        if isinstance(pipes_raw, list):
            for p in pipes_raw:
                pipes.append(PipeSpec.from_dict(p))
        elif isinstance(pipes_raw, dict):
            # Dict format: {signal_name: {diameter: ..., length: ...}}
            for signal, spec in pipes_raw.items():
                spec['signal'] = signal
                pipes.append(PipeSpec.from_dict(spec))

        return cls(
            pipes=pipes,
            connections=d.get('connections', []),
        )

    def get_pipe(self, signal: str) -> Optional[PipeSpec]:
        """Get pipe spec for a signal."""
        for p in self.pipes:
            if p.signal == signal:
                return p
        return None

    def has_pipe(self, signal: str) -> bool:
        """Check if signal has pipe geometry."""
        return self.get_pipe(signal) is not None


# =============================================================================
# RELATIONSHIPS SPECIFICATION
# =============================================================================

@dataclass
class MechanicalRelationship:
    """Mechanical signal relationships."""
    position: Optional[str] = None      # Column name
    velocity: Optional[str] = None
    acceleration: Optional[str] = None
    force: Optional[str] = None

    # Derive flags
    derive_velocity: bool = False       # v = dx/dt
    derive_acceleration: bool = False   # a = dv/dt


@dataclass
class ThermodynamicRelationship:
    """Thermodynamic signal relationships."""
    temperature: Optional[str] = None   # Column name
    pressure: Optional[str] = None
    volume: Optional[str] = None
    heat: Optional[str] = None


@dataclass
class ControlRelationship:
    """Control system I/O relationship."""
    input: Optional[str] = None         # Input signal column
    output: Optional[str] = None        # Output signal column


@dataclass
class Vector3D:
    """3D vector from component signals."""
    name: str
    x: str      # Column name for x component
    y: str      # Column name for y component
    z: str      # Column name for z component


@dataclass
class RelationshipsSpec:
    """All signal relationships."""
    mechanical: Optional[MechanicalRelationship] = None
    thermodynamic: Optional[ThermodynamicRelationship] = None
    control: Optional[ControlRelationship] = None
    vectors_3d: List[Vector3D] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict) -> 'RelationshipsSpec':
        mech = d.get('mechanical', {})
        thermo = d.get('thermodynamic', {})
        control = d.get('control', {})
        vectors = d.get('vectors_3d', [])

        return cls(
            mechanical=MechanicalRelationship(
                position=mech.get('position'),
                velocity=mech.get('velocity'),
                acceleration=mech.get('acceleration'),
                force=mech.get('force'),
                derive_velocity=mech.get('derive_velocity', False),
                derive_acceleration=mech.get('derive_acceleration', False),
            ) if mech else None,
            thermodynamic=ThermodynamicRelationship(
                temperature=thermo.get('temperature'),
                pressure=thermo.get('pressure'),
                volume=thermo.get('volume'),
                heat=thermo.get('heat'),
            ) if thermo else None,
            control=ControlRelationship(
                input=control.get('input'),
                output=control.get('output'),
            ) if control else None,
            vectors_3d=[
                Vector3D(name=v['name'], x=v['x'], y=v['y'], z=v['z'])
                for v in vectors
            ],
        )


# =============================================================================
# SPATIAL SPECIFICATION
# =============================================================================

class SpatialType(Enum):
    """Type of spatial field data."""
    NONE = "none"
    VELOCITY_FIELD = "velocity_field"
    TEMPERATURE_FIELD = "temperature_field"
    EM_FIELD = "em_field"


@dataclass
class SpatialSpec:
    """Spatial field data specification."""
    type: SpatialType = SpatialType.NONE
    dimensions: Optional[Tuple[int, int, int]] = None  # nx, ny, nz

    # Component column names
    u: Optional[str] = None   # velocity_x or E_x
    v: Optional[str] = None   # velocity_y or E_y
    w: Optional[str] = None   # velocity_z or E_z

    # For EM fields
    Bx: Optional[str] = None
    By: Optional[str] = None
    Bz: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict) -> 'SpatialSpec':
        if not d:
            return cls()

        type_str = d.get('type', 'none')
        try:
            spatial_type = SpatialType(type_str)
        except ValueError:
            spatial_type = SpatialType.NONE

        dims = d.get('dimensions')
        if dims:
            dims = tuple(dims)

        return cls(
            type=spatial_type,
            dimensions=dims,
            u=d.get('u') or d.get('velocity_x'),
            v=d.get('v') or d.get('velocity_y'),
            w=d.get('w') or d.get('velocity_z'),
            Bx=d.get('Bx'),
            By=d.get('By'),
            Bz=d.get('Bz'),
        )


# =============================================================================
# FULL DATA SPECIFICATION
# =============================================================================

@dataclass
class DataSpec:
    """Complete specification of available data."""
    signals: List[SignalSpec]
    constants: ConstantsSpec
    relationships: RelationshipsSpec
    spatial: SpatialSpec
    pipe_network: PipeNetworkSpec

    # Windowing (required)
    window_size: int
    window_stride: int

    @classmethod
    def from_config(cls, config: Dict) -> 'DataSpec':
        """Parse configuration dict into DataSpec."""
        signals = [
            SignalSpec.from_dict(s) if isinstance(s, dict) else SignalSpec(name=s)
            for s in config.get('signals', [])
        ]

        return cls(
            signals=signals,
            constants=ConstantsSpec.from_dict(config.get('constants', {})),
            relationships=RelationshipsSpec.from_dict(config.get('relationships', {})),
            spatial=SpatialSpec.from_dict(config.get('spatial', {})),
            pipe_network=PipeNetworkSpec.from_dict(config.get('pipe_network', {})),
            window_size=config.get('window_size', 50),
            window_stride=config.get('window_stride', 25),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Helper methods
    # ─────────────────────────────────────────────────────────────────────────

    def has_signal_type(self, qty: PhysicalQuantity) -> bool:
        """Check if we have a signal of given type."""
        return any(s.physical_quantity == qty for s in self.signals)

    def get_signal(self, qty: PhysicalQuantity) -> Optional[SignalSpec]:
        """Get signal spec for a physical quantity."""
        for s in self.signals:
            if s.physical_quantity == qty:
                return s
        return None

    def has_labeled_signals(self) -> bool:
        """Check if any signals have physical quantity labels."""
        return any(s.physical_quantity != PhysicalQuantity.UNKNOWN for s in self.signals)

    def has_velocity(self) -> bool:
        return (
            self.has_signal_type(PhysicalQuantity.VELOCITY) or
            (self.relationships.mechanical and self.relationships.mechanical.velocity)
        )

    def has_position(self) -> bool:
        return (
            self.has_signal_type(PhysicalQuantity.POSITION) or
            (self.relationships.mechanical and self.relationships.mechanical.position)
        )

    def has_force(self) -> bool:
        return (
            self.has_signal_type(PhysicalQuantity.FORCE) or
            (self.relationships.mechanical and self.relationships.mechanical.force)
        )

    def has_temperature(self) -> bool:
        return (
            self.has_signal_type(PhysicalQuantity.TEMPERATURE) or
            (self.relationships.thermodynamic and self.relationships.thermodynamic.temperature)
        )

    def has_pressure(self) -> bool:
        return (
            self.has_signal_type(PhysicalQuantity.PRESSURE) or
            (self.relationships.thermodynamic and self.relationships.thermodynamic.pressure)
        )

    def has_volume(self) -> bool:
        return (
            self.has_signal_type(PhysicalQuantity.VOLUME) or
            (self.relationships.thermodynamic and self.relationships.thermodynamic.volume)
        )

    def has_io_pair(self) -> bool:
        return (
            self.relationships.control and
            self.relationships.control.input and
            self.relationships.control.output
        )

    def has_3d_position(self) -> bool:
        return any(v.name == 'position' for v in self.relationships.vectors_3d)

    def has_3d_velocity(self) -> bool:
        return any(v.name == 'velocity' for v in self.relationships.vectors_3d)

    def has_velocity_field(self) -> bool:
        return self.spatial.type == SpatialType.VELOCITY_FIELD

    # ─────────────────────────────────────────────────────────────────────────
    # Pipe network helpers
    # ─────────────────────────────────────────────────────────────────────────

    def has_pipe_network(self) -> bool:
        """Check if pipe geometry is defined."""
        return len(self.pipe_network.pipes) > 0

    def has_pipe_for_signal(self, signal: str) -> bool:
        """Check if a specific signal has pipe geometry."""
        return self.pipe_network.has_pipe(signal)

    def get_pipe(self, signal: str) -> Optional[PipeSpec]:
        """Get pipe geometry for a signal."""
        return self.pipe_network.get_pipe(signal)

    def can_compute_reynolds(self) -> bool:
        """Check if Reynolds number can be computed for any pipe."""
        return (
            self.has_pipe_network() and
            self.constants.kinematic_viscosity is not None
        )

    def can_compute_pressure_drop(self) -> bool:
        """Check if pressure drop can be computed."""
        return (
            self.can_compute_reynolds() and
            self.constants.density is not None and
            any(p.length is not None for p in self.pipe_network.pipes)
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Unit helpers
    # ─────────────────────────────────────────────────────────────────────────

    def get_signal_units(self) -> Dict[str, UnitSpec]:
        """Get units for all signals that have them."""
        units = {}
        for signal in self.signals:
            if signal.has_units():
                units[signal.name] = signal.get_unit_spec()
        return units

    def get_input_units(self, computation: str) -> Dict[str, UnitSpec]:
        """
        Get input units needed for a computation.

        Returns dict with keys like 'velocity', 'position', 'mass', etc.
        mapped to their UnitSpec.
        """
        units = {}

        # Add signal units by physical quantity
        for signal in self.signals:
            if signal.physical_quantity == PhysicalQuantity.VELOCITY:
                units['velocity'] = signal.get_unit_spec()
            elif signal.physical_quantity == PhysicalQuantity.POSITION:
                units['position'] = signal.get_unit_spec()
            elif signal.physical_quantity == PhysicalQuantity.ACCELERATION:
                units['acceleration'] = signal.get_unit_spec()
            elif signal.physical_quantity == PhysicalQuantity.FORCE:
                units['force'] = signal.get_unit_spec()
            elif signal.physical_quantity == PhysicalQuantity.TEMPERATURE:
                units['temperature'] = signal.get_unit_spec()
            elif signal.physical_quantity == PhysicalQuantity.PRESSURE:
                units['pressure'] = signal.get_unit_spec()
            elif signal.physical_quantity == PhysicalQuantity.VOLUME:
                units['volume'] = signal.get_unit_spec()

        # Add constant units (SI by default)
        if self.constants.mass is not None:
            units['mass'] = UnitSpec.parse('kg')
        if self.constants.spring_constant is not None:
            units['spring_constant'] = UnitSpec.parse('N/m')
        if self.constants.moment_of_inertia is not None:
            units['moment_of_inertia'] = UnitSpec.parse('kg⋅m²')
        if self.constants.Cp is not None:
            units['Cp'] = UnitSpec.parse('J/(mol⋅K)')
        if self.constants.kinematic_viscosity is not None:
            units['kinematic_viscosity'] = UnitSpec.parse('m²/s')
        if self.constants.density is not None:
            units['density'] = UnitSpec.parse('kg/m³')
        if self.constants.dt is not None:
            units['dt'] = UnitSpec.parse('s')

        return units

    def has_any_units(self) -> bool:
        """Check if any signals have units specified."""
        return any(s.has_units() for s in self.signals)

    def compute_with_units(
        self,
        computation: str,
        value: float,
    ) -> UnitResult:
        """
        Wrap a computed value with its derived unit.

        Args:
            computation: Name of computation (e.g., 'kinetic_energy')
            value: The computed numeric value

        Returns:
            UnitResult with value and unit
        """
        input_units = self.get_input_units(computation)
        return propagate_units(computation, value, input_units)


# =============================================================================
# CAPABILITY REQUIREMENTS
# =============================================================================

# What each capability requires
CAPABILITY_REQUIREMENTS: Dict[Capability, Dict[str, Any]] = {
    # Level 0: No requirements beyond data existing
    Capability.STATISTICS: {},
    Capability.DISTRIBUTION: {},
    Capability.STATIONARITY: {},
    Capability.ENTROPY: {},
    Capability.MEMORY: {},
    Capability.SPECTRAL: {},
    Capability.RECURRENCE: {},
    Capability.CHAOS: {},
    Capability.VOLATILITY: {},
    Capability.EVENT_DETECTION: {},
    Capability.HEAVISIDE_DIRAC: {},
    Capability.GEOMETRY: {},
    Capability.DYNAMICS: {},

    # Level 1: Need signal labels
    Capability.DERIVATIVES: {'labeled': True},
    Capability.SPECIFIC_KINETIC: {'signals': ['velocity']},
    Capability.SPECIFIC_POTENTIAL: {'signals': ['position']},
    Capability.SPECIFIC_HAMILTONIAN: {'signals': ['position', 'velocity']},

    # Level 2: Need constants
    Capability.KINETIC_ENERGY: {'signals': ['velocity'], 'constants': ['mass']},
    Capability.POTENTIAL_ENERGY: {'signals': ['position'], 'constants': ['spring_constant']},
    Capability.MOMENTUM: {'signals': ['velocity'], 'constants': ['mass']},
    Capability.HAMILTONIAN: {'signals': ['position', 'velocity'], 'constants': ['mass', 'spring_constant']},
    Capability.LAGRANGIAN: {'signals': ['position', 'velocity'], 'constants': ['mass', 'spring_constant']},
    Capability.ROTATIONAL_KE: {'signals': ['angular_velocity'], 'constants': ['moment_of_inertia']},

    # Level 3: Need related signals
    Capability.WORK: {'signals': ['position', 'force']},
    Capability.POWER: {'signals': ['velocity', 'force']},
    Capability.ANGULAR_MOMENTUM: {'vectors_3d': ['position', 'velocity'], 'constants': ['mass']},
    Capability.PHASE_SPACE: {'signals': ['position', 'velocity'], 'constants': ['mass']},
    Capability.ENERGY_CONSERVATION: {'signals': ['position', 'velocity'], 'constants': ['mass', 'spring_constant']},

    Capability.GIBBS_FREE_ENERGY: {'thermo': ['temperature', 'pressure', 'volume'], 'constants': ['Cp']},
    Capability.ENTHALPY: {'thermo': ['temperature', 'pressure', 'volume'], 'constants': ['Cp']},
    Capability.ENTROPY_THERMO: {'thermo': ['temperature', 'pressure', 'volume'], 'constants': ['Cp']},
    Capability.CHEMICAL_POTENTIAL: {'thermo': ['temperature', 'pressure'], 'constants': ['Cp']},

    Capability.TRANSFER_FUNCTION: {'io_pair': True},
    Capability.FREQUENCY_RESPONSE: {'io_pair': True},
    Capability.POLES_ZEROS: {'io_pair': True},
    Capability.IMPULSE_RESPONSE: {'io_pair': True},

    Capability.GRANGER_CAUSALITY: {'min_signals': 2},
    Capability.TRANSFER_ENTROPY: {'min_signals': 2},

    # Level 4: Need spatial data
    Capability.VORTICITY: {'spatial': 'velocity_field'},
    Capability.STRAIN_TENSOR: {'spatial': 'velocity_field'},
    Capability.Q_CRITERION: {'spatial': 'velocity_field'},
    Capability.TURBULENT_KE: {'spatial': 'velocity_field'},
    Capability.DISSIPATION: {'spatial': 'velocity_field', 'constants': ['kinematic_viscosity']},
    Capability.ENERGY_SPECTRUM: {'spatial': 'velocity_field'},
    Capability.REYNOLDS_NUMBER: {'spatial': 'velocity_field', 'constants': ['kinematic_viscosity']},
    Capability.KOLMOGOROV_SCALES: {'spatial': 'velocity_field', 'constants': ['kinematic_viscosity']},

    Capability.HEAT_FLUX: {'spatial': 'temperature_field'},
    Capability.LAPLACIAN_T: {'spatial': 'temperature_field'},

    Capability.MAXWELL_DIV_E: {'spatial': 'em_field'},
    Capability.MAXWELL_CURL_B: {'spatial': 'em_field'},
    Capability.POYNTING: {'spatial': 'em_field'},

    # Pipe flow: Need pipe geometry
    Capability.PIPE_REYNOLDS: {'pipe_network': True, 'constants': ['kinematic_viscosity']},
    Capability.FLOW_REGIME: {'pipe_network': True, 'constants': ['kinematic_viscosity']},
    Capability.FRICTION_FACTOR: {'pipe_network': True, 'constants': ['kinematic_viscosity']},
    Capability.PRESSURE_DROP: {'pipe_network': True, 'constants': ['kinematic_viscosity', 'density']},
    Capability.HEAD_LOSS: {'pipe_network': True, 'constants': ['kinematic_viscosity', 'density']},
    Capability.PIPE_POWER_LOSS: {'pipe_network': True, 'constants': ['kinematic_viscosity', 'density']},
}


# =============================================================================
# CAPABILITY REPORT
# =============================================================================

@dataclass
class CapabilityReport:
    """
    Report of what PRISM can compute from the available data.
    """
    available: Set[Capability]
    unavailable: Dict[Capability, str]  # Capability → what's missing
    data_level: DataLevel
    data_spec: DataSpec

    def __str__(self) -> str:
        """Pretty print the capability report."""
        lines = [
            "=" * 70,
            "PRISM CAPABILITY REPORT",
            "=" * 70,
            f"",
            f"Data Level: {self.data_level.name}",
            f"",
        ]

        # Group available by level
        level_0 = []
        level_1 = []
        level_2 = []
        level_3 = []
        level_4 = []

        for cap in sorted(self.available, key=lambda x: x.name):
            reqs = CAPABILITY_REQUIREMENTS.get(cap, {})
            if not reqs:
                level_0.append(cap)
            elif 'spatial' in reqs:
                level_4.append(cap)
            elif 'io_pair' in reqs or 'thermo' in reqs or 'vectors_3d' in reqs:
                level_3.append(cap)
            elif 'constants' in reqs:
                level_2.append(cap)
            else:
                level_1.append(cap)

        lines.append(f"AVAILABLE CAPABILITIES ({len(self.available)})")
        lines.append("-" * 70)

        if level_0:
            lines.append(f"\n  Level 0 - Raw Time Series ({len(level_0)}):")
            for cap in level_0:
                lines.append(f"    [x] {cap.name}")

        if level_1:
            lines.append(f"\n  Level 1 - Labeled Signals ({len(level_1)}):")
            for cap in level_1:
                lines.append(f"    [x] {cap.name}")

        if level_2:
            lines.append(f"\n  Level 2 - With Constants ({len(level_2)}):")
            for cap in level_2:
                lines.append(f"    [x] {cap.name}")

        if level_3:
            lines.append(f"\n  Level 3 - Related Signals ({len(level_3)}):")
            for cap in level_3:
                lines.append(f"    [x] {cap.name}")

        if level_4:
            lines.append(f"\n  Level 4 - Spatial Fields ({len(level_4)}):")
            for cap in level_4:
                lines.append(f"    [x] {cap.name}")

        if self.unavailable:
            lines.append(f"\n")
            lines.append(f"UNAVAILABLE - What's Missing ({len(self.unavailable)})")
            lines.append("-" * 70)
            for cap, missing in sorted(self.unavailable.items(), key=lambda x: x[0].name):
                lines.append(f"    [ ] {cap.name}")
                lines.append(f"        -> need: {missing}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'data_level': self.data_level.name,
            'available': [c.name for c in sorted(self.available, key=lambda x: x.name)],
            'unavailable': {
                c.name: msg for c, msg in self.unavailable.items()
            },
            'n_available': len(self.available),
            'n_unavailable': len(self.unavailable),
        }


# =============================================================================
# CAPABILITY DETECTOR
# =============================================================================

def check_requirements(spec: DataSpec, reqs: Dict) -> Optional[str]:
    """
    Check if requirements are met.

    Returns None if met, or a string describing what's missing.
    """
    missing = []

    # Check for labeled signals requirement
    if reqs.get('labeled') and not spec.has_labeled_signals():
        return "signal labels (physical_quantity)"

    # Check for specific signal types
    signals_needed = reqs.get('signals', [])
    for sig in signals_needed:
        if sig == 'velocity' and not spec.has_velocity():
            missing.append("velocity signal")
        elif sig == 'position' and not spec.has_position():
            missing.append("position signal")
        elif sig == 'force' and not spec.has_force():
            missing.append("force signal")
        elif sig == 'angular_velocity' and not spec.has_signal_type(PhysicalQuantity.ANGULAR_VELOCITY):
            missing.append("angular_velocity signal")

    # Check for constants
    constants_needed = reqs.get('constants', [])
    for const in constants_needed:
        if const == 'mass' and not spec.constants.mass:
            missing.append("mass [kg]")
        elif const == 'spring_constant' and not spec.constants.spring_constant:
            missing.append("spring_constant [N/m]")
        elif const == 'moment_of_inertia' and not spec.constants.moment_of_inertia:
            missing.append("moment_of_inertia [kg*m^2]")
        elif const == 'Cp' and not spec.constants.Cp:
            missing.append("Cp [J/(mol*K)]")
        elif const == 'kinematic_viscosity' and not spec.constants.kinematic_viscosity:
            missing.append("kinematic_viscosity (nu) [m^2/s]")

    # Check for thermodynamic signals
    thermo_needed = reqs.get('thermo', [])
    for sig in thermo_needed:
        if sig == 'temperature' and not spec.has_temperature():
            missing.append("temperature signal")
        elif sig == 'pressure' and not spec.has_pressure():
            missing.append("pressure signal")
        elif sig == 'volume' and not spec.has_volume():
            missing.append("volume signal")

    # Check for I/O pair
    if reqs.get('io_pair') and not spec.has_io_pair():
        missing.append("input + output signal pair")

    # Check for 3D vectors
    vectors_needed = reqs.get('vectors_3d', [])
    for vec in vectors_needed:
        if vec == 'position' and not spec.has_3d_position():
            missing.append("3D position vector (x, y, z)")
        elif vec == 'velocity' and not spec.has_3d_velocity():
            missing.append("3D velocity vector (vx, vy, vz)")

    # Check for spatial field
    spatial_needed = reqs.get('spatial')
    if spatial_needed:
        if spatial_needed == 'velocity_field' and not spec.has_velocity_field():
            missing.append("velocity field v(x,y,z,t)")
        elif spatial_needed == 'temperature_field' and spec.spatial.type != SpatialType.TEMPERATURE_FIELD:
            missing.append("temperature field T(x,y,z,t)")
        elif spatial_needed == 'em_field' and spec.spatial.type != SpatialType.EM_FIELD:
            missing.append("electromagnetic field E(x,y,z,t), B(x,y,z,t)")

    # Check minimum signals
    min_signals = reqs.get('min_signals')
    if min_signals and len(spec.signals) < min_signals:
        missing.append(f"at least {min_signals} signals")

    # Check for pipe network
    if reqs.get('pipe_network') and not spec.has_pipe_network():
        missing.append("pipe_network with pipe geometry (diameter)")

    # Check for density (used in pressure drop, etc.)
    if 'density' in reqs.get('constants', []) and not spec.constants.density:
        missing.append("density (rho) [kg/m^3]")

    if missing:
        return ", ".join(missing)

    return None


def detect_capabilities(config: Dict) -> CapabilityReport:
    """
    Analyze configuration and determine what's computable.

    This is the main entry point for capability detection.

    Args:
        config: Configuration dictionary with signals, constants, relationships

    Returns:
        CapabilityReport with available/unavailable capabilities
    """
    spec = DataSpec.from_config(config)

    available: Set[Capability] = set()
    unavailable: Dict[Capability, str] = {}

    # Determine data level
    data_level = DataLevel.RAW_TIMESERIES

    if spec.has_labeled_signals():
        data_level = DataLevel.LABELED_SIGNALS

    has_any_constant = any([
        spec.constants.mass,
        spec.constants.spring_constant,
        spec.constants.moment_of_inertia,
        spec.constants.Cp,
    ])
    if has_any_constant:
        data_level = DataLevel.WITH_CONSTANTS

    has_relationships = any([
        spec.has_io_pair(),
        (spec.relationships.thermodynamic and spec.has_temperature() and spec.has_pressure()),
        spec.has_3d_position(),
    ])
    if has_relationships:
        data_level = DataLevel.RELATED_SIGNALS

    if spec.has_velocity_field():
        data_level = DataLevel.SPATIAL_FIELD

    # Check each capability
    for cap, reqs in CAPABILITY_REQUIREMENTS.items():
        missing = check_requirements(spec, reqs)

        if not missing:
            available.add(cap)
        else:
            unavailable[cap] = missing

    return CapabilityReport(
        available=available,
        unavailable=unavailable,
        data_level=data_level,
        data_spec=spec,
    )


# =============================================================================
# CAPABILITY → ENGINE MAPPING
# =============================================================================

CAPABILITY_TO_ENGINE: Dict[Capability, str] = {
    # Level 0
    Capability.STATISTICS: 'typology.statistics',
    Capability.DISTRIBUTION: 'typology.distribution',
    Capability.STATIONARITY: 'typology.stationarity',
    Capability.ENTROPY: 'information.*',
    Capability.MEMORY: 'memory.*',
    Capability.SPECTRAL: 'frequency.*',
    Capability.RECURRENCE: 'recurrence.rqa',
    Capability.CHAOS: 'dynamics.lyapunov',
    Capability.VOLATILITY: 'volatility.*',
    Capability.EVENT_DETECTION: 'events.heaviside_dirac',
    Capability.HEAVISIDE_DIRAC: 'events.heaviside_dirac',
    Capability.GEOMETRY: 'geometry.*',
    Capability.DYNAMICS: 'dynamics.*',

    # Level 1
    Capability.DERIVATIVES: 'pointwise.derivatives',
    Capability.SPECIFIC_KINETIC: 'physics.kinetic_energy',
    Capability.SPECIFIC_POTENTIAL: 'physics.potential_energy',
    Capability.SPECIFIC_HAMILTONIAN: 'physics.hamiltonian',

    # Level 2
    Capability.KINETIC_ENERGY: 'physics.kinetic_energy',
    Capability.POTENTIAL_ENERGY: 'physics.potential_energy',
    Capability.MOMENTUM: 'physics.momentum',
    Capability.HAMILTONIAN: 'physics.hamiltonian',
    Capability.LAGRANGIAN: 'physics.lagrangian',
    Capability.ROTATIONAL_KE: 'physics.kinetic_energy',

    # Level 3
    Capability.WORK: 'physics.work_energy',
    Capability.POWER: 'physics.work_energy',
    Capability.ANGULAR_MOMENTUM: 'physics.momentum',
    Capability.PHASE_SPACE: 'physics.hamiltonian',
    Capability.ENERGY_CONSERVATION: 'physics.hamiltonian',

    Capability.GIBBS_FREE_ENERGY: 'physics.gibbs_free_energy',
    Capability.ENTHALPY: 'physics.gibbs_free_energy',
    Capability.ENTROPY_THERMO: 'physics.gibbs_free_energy',
    Capability.CHEMICAL_POTENTIAL: 'physics.gibbs_free_energy',

    Capability.TRANSFER_FUNCTION: 'systems.transfer_function',
    Capability.FREQUENCY_RESPONSE: 'systems.frequency_response',
    Capability.POLES_ZEROS: 'systems.poles_zeros',
    Capability.IMPULSE_RESPONSE: 'systems.transfer_function',

    Capability.GRANGER_CAUSALITY: 'state.granger',
    Capability.TRANSFER_ENTROPY: 'state.transfer_entropy',

    # Level 4
    Capability.VORTICITY: 'fields.navier_stokes',
    Capability.STRAIN_TENSOR: 'fields.navier_stokes',
    Capability.Q_CRITERION: 'fields.navier_stokes',
    Capability.TURBULENT_KE: 'fields.navier_stokes',
    Capability.DISSIPATION: 'fields.navier_stokes',
    Capability.ENERGY_SPECTRUM: 'fields.navier_stokes',
    Capability.REYNOLDS_NUMBER: 'fields.navier_stokes',
    Capability.KOLMOGOROV_SCALES: 'fields.navier_stokes',

    Capability.HEAT_FLUX: 'fields.heat_transfer',
    Capability.LAPLACIAN_T: 'fields.heat_transfer',

    Capability.MAXWELL_DIV_E: 'fields.electromagnetism',
    Capability.MAXWELL_CURL_B: 'fields.electromagnetism',
    Capability.POYNTING: 'fields.electromagnetism',
}


def get_engines_for_capabilities(capabilities: Set[Capability]) -> Set[str]:
    """
    Get the set of engines needed for the given capabilities.
    """
    engines = set()

    for cap in capabilities:
        engine = CAPABILITY_TO_ENGINE.get(cap)
        if engine:
            if '*' in engine:
                # Wildcard - add the module
                engines.add(engine.replace('.*', ''))
            else:
                engines.add(engine)

    return engines


# =============================================================================
# PRINT CURRICULUM
# =============================================================================

def print_curriculum():
    """Print the full PRISM capability curriculum."""
    print("""
+==============================================================================+
|                    PRISM COMPUTATIONAL PHYSICS CURRICULUM                    |
+==============================================================================+
|                                                                              |
|  You bring the data. We compute everything possible.                         |
|  You bring more context. We unlock more physics.                             |
|                                                                              |
+==============================================================================+
|                                                                              |
|  LEVEL 0: Raw Time Series                                                    |
|  ------------------------                                                    |
|  Input:  Just columns of numbers                                             |
|  Unlock: Statistics, Entropy, Memory, Spectral, Stationarity,                |
|          Recurrence, Chaos, Volatility, Event Detection, H(t), d(t)          |
|                                                                              |
+==============================================================================+
|                                                                              |
|  LEVEL 1: Labeled Signals                                                    |
|  ------------------------                                                    |
|  Input:  + Tell us what signals represent                                    |
|          physical_quantity: position, velocity, temperature, etc.            |
|  Unlock: + Derivatives with units, Specific KE (1/2 v^2), Specific PE        |
|                                                                              |
+==============================================================================+
|                                                                              |
|  LEVEL 2: Physical Constants                                                 |
|  ---------------------------                                                 |
|  Input:  + mass [kg], spring_constant [N/m], Cp [J/(mol*K)], etc.            |
|  Unlock: + REAL kinetic energy (1/2 mv^2), REAL potential (1/2 kx^2)         |
|          + Momentum (mv), Hamiltonian (T+V), Lagrangian (T-V)                |
|                                                                              |
+==============================================================================+
|                                                                              |
|  LEVEL 3: Related Signals                                                    |
|  ------------------------                                                    |
|  Input:  + Signal relationships                                              |
|          mechanical: {position: x, velocity: v, force: F}                    |
|          thermodynamic: {temperature: T, pressure: P, volume: V}             |
|          control: {input: u, output: y}                                      |
|  Unlock: + Work (integral F dx), Power (F v)                                 |
|          + Angular momentum (r x p) for 3D data                              |
|          + Gibbs free energy (H - TS)                                        |
|          + Transfer functions G(s), Bode plots, stability                    |
|                                                                              |
+==============================================================================+
|                                                                              |
|  LEVEL 4: Spatial Fields                                                     |
|  -----------------------                                                     |
|  Input:  + Velocity field v(x,y,z,t), grid spacing, viscosity                |
|  Unlock: + FULL NAVIER-STOKES                                                |
|            Vorticity (curl v), Strain tensor, Q-criterion                    |
|            Turbulent KE, Dissipation rate, Energy spectrum E(k)              |
|            Reynolds number, Kolmogorov scales                                |
|                                                                              |
+==============================================================================+
""")


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test-units':
        print("=" * 70)
        print("PRISM Unit System Test")
        print("=" * 70)

        # Test basic unit parsing
        print("\n1. Basic Unit Parsing:")
        for unit_str in ['m', 'm/s', 'm/s²', 'N', 'J', 'kg', 'Pa', 'K', 'ft/s', 'psi']:
            u = UnitSpec.parse(unit_str)
            print(f"   {unit_str:10} -> {u.dimensions} = {u.to_si_symbol()}")

        # Test unit algebra
        print("\n2. Unit Algebra:")
        m = UnitSpec.parse('m')
        s = UnitSpec.parse('s')
        kg = UnitSpec.parse('kg')

        velocity = m / s
        print(f"   m / s = {velocity} (velocity)")

        acceleration = velocity / s
        print(f"   (m/s) / s = {acceleration} (acceleration)")

        force = kg * acceleration
        print(f"   kg * (m/s²) = {force} (force = {force.to_si_symbol()})")

        energy = force * m
        print(f"   N * m = {energy} (energy = {energy.to_si_symbol()})")

        momentum = kg * velocity
        print(f"   kg * (m/s) = {momentum} (momentum)")

        # Test unit result
        print("\n3. Unit Results:")
        ke_value = 450.0  # Computed kinetic energy
        ke_units = kg * (velocity ** 2)
        ke_result = UnitResult(value=ke_value, unit=ke_units)
        print(f"   Kinetic energy: {ke_result}")

        # Test conversion
        print("\n4. Unit Conversion:")
        temp_c = UnitResult(value=25.0, unit=UnitSpec.parse('°C'))
        print(f"   Temperature: {temp_c}")

        vel_ms = UnitResult(value=10.0, unit=UnitSpec.parse('m/s'))
        vel_kmh = vel_ms.to('km/h')
        print(f"   Velocity: {vel_ms} = {vel_kmh}")

        # Test with config
        print("\n5. Config Integration:")
        config = {
            'signals': [
                {'name': 'position', 'physical_quantity': 'position', 'units': 'm'},
                {'name': 'velocity', 'physical_quantity': 'velocity', 'units': 'm/s'},
                {'name': 'sensor_raw'},  # No units
            ],
            'constants': {
                'mass': 2.0,
                'spring_constant': 50.0,
            },
            'window_size': 100,
            'window_stride': 50,
        }

        spec = DataSpec.from_config(config)

        print(f"   Has any units: {spec.has_any_units()}")
        print(f"   Signal units:")
        for name, unit in spec.get_signal_units().items():
            print(f"      {name}: {unit}")

        print(f"\n   Input units for kinetic_energy:")
        input_units = spec.get_input_units('kinetic_energy')
        for name, unit in input_units.items():
            print(f"      {name}: {unit}")

        # Compute with units
        ke_raw = 0.5 * 2.0 * (3.0 ** 2)  # ½mv² with m=2kg, v=3m/s
        ke_with_units = spec.compute_with_units('kinetic_energy', ke_raw)
        print(f"\n   KE = ½mv² = ½ * 2.0 * 3.0² = {ke_with_units}")

        # Imperial units example
        print("\n6. Imperial Units:")
        imperial_config = {
            'signals': [
                {'name': 'velocity', 'physical_quantity': 'velocity', 'units': 'ft/s'},
                {'name': 'pressure', 'physical_quantity': 'pressure', 'units': 'psi'},
            ],
            'constants': {
                'mass': 10.0,  # Still in kg (SI)
            },
            'window_size': 100,
            'window_stride': 50,
        }

        imperial_spec = DataSpec.from_config(imperial_config)
        for sig in imperial_spec.signals:
            u = sig.get_unit_spec()
            print(f"   {sig.name}: {sig.units} -> scale={u.scale} to SI")

        print("\n" + "=" * 70)
        print("Unit system test complete!")

    elif len(sys.argv) > 1 and sys.argv[1] == '--curriculum':
        print_curriculum()

    else:
        print("Usage:")
        print("  python -m prism.capability --test-units    Test unit system")
        print("  python -m prism.capability --curriculum    Print curriculum")
