"""
PRISM UnitSpec — Comprehensive Unit Handling for Engineering Data

Parses user input in ANY common engineering unit, converts to SI internally,
and provides dimensional analysis to enable/gate physics computations.

Usage:
    >>> from prism.unitspec import Q, Quantity

    # Parse user input
    >>> flow = Q(100, "gpm")
    >>> flow.to("m³/s")
    0.006309...

    # Automatic conversion
    >>> diameter = Q("4 in")
    >>> diameter.si
    0.1016  # meters

    # Dimensional analysis
    >>> velocity = flow / (3.14159 * (diameter/2)**2)
    >>> velocity.dimensions
    Dimensions(length=1, time=-1)  # m/s

    # Capability gating
    >>> can_compute_reynolds = (
    ...     velocity.is_compatible("m/s") and
    ...     density.is_compatible("kg/m³") and
    ...     viscosity.is_compatible("Pa·s")
    ... )

Author: PRISM Team
License: MIT
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import re
import math


# =============================================================================
# DIMENSIONS (Base SI quantities)
# =============================================================================

@dataclass(frozen=True)
class Dimensions:
    """
    Dimensional analysis using the 7 SI base quantities.

    Each dimension is an exponent:
        velocity = length^1 * time^-1 -> Dimensions(length=1, time=-1)
        force = mass^1 * length^1 * time^-2 -> Dimensions(mass=1, length=1, time=-2)
    """
    length: int = 0           # L (meter)
    mass: int = 0             # M (kilogram)
    time: int = 0             # T (second)
    current: int = 0          # I (ampere)
    temperature: int = 0      # Theta (kelvin)
    amount: int = 0           # N (mole)
    luminosity: int = 0       # J (candela)

    def __mul__(self, other: Dimensions) -> Dimensions:
        """Multiply quantities -> add exponents"""
        return Dimensions(
            self.length + other.length,
            self.mass + other.mass,
            self.time + other.time,
            self.current + other.current,
            self.temperature + other.temperature,
            self.amount + other.amount,
            self.luminosity + other.luminosity,
        )

    def __truediv__(self, other: Dimensions) -> Dimensions:
        """Divide quantities -> subtract exponents"""
        return Dimensions(
            self.length - other.length,
            self.mass - other.mass,
            self.time - other.time,
            self.current - other.current,
            self.temperature - other.temperature,
            self.amount - other.amount,
            self.luminosity - other.luminosity,
        )

    def __pow__(self, exp: int) -> Dimensions:
        """Raise to power -> multiply all exponents"""
        return Dimensions(
            self.length * exp,
            self.mass * exp,
            self.time * exp,
            self.current * exp,
            self.temperature * exp,
            self.amount * exp,
            self.luminosity * exp,
        )

    def __eq__(self, other: Dimensions) -> bool:
        return (
            self.length == other.length and
            self.mass == other.mass and
            self.time == other.time and
            self.current == other.current and
            self.temperature == other.temperature and
            self.amount == other.amount and
            self.luminosity == other.luminosity
        )

    def __hash__(self) -> int:
        return hash((self.length, self.mass, self.time, self.current,
                     self.temperature, self.amount, self.luminosity))

    def is_dimensionless(self) -> bool:
        return self == DIMENSIONLESS

    def __repr__(self) -> str:
        parts = []
        names = ['L', 'M', 'T', 'I', 'Theta', 'N', 'J']
        vals = [self.length, self.mass, self.time, self.current,
                self.temperature, self.amount, self.luminosity]
        for name, val in zip(names, vals):
            if val == 1:
                parts.append(name)
            elif val != 0:
                parts.append(f"{name}^{val}")
        return ' * '.join(parts) if parts else '1'


# Common dimension constants
DIMENSIONLESS = Dimensions()
LENGTH = Dimensions(length=1)
MASS = Dimensions(mass=1)
TIME = Dimensions(time=1)
CURRENT = Dimensions(current=1)
TEMPERATURE = Dimensions(temperature=1)
AMOUNT = Dimensions(amount=1)

# Derived dimensions
AREA = Dimensions(length=2)
VOLUME = Dimensions(length=3)
VELOCITY = Dimensions(length=1, time=-1)
ACCELERATION = Dimensions(length=1, time=-2)
FREQUENCY = Dimensions(time=-1)
FORCE = Dimensions(mass=1, length=1, time=-2)
PRESSURE = Dimensions(mass=1, length=-1, time=-2)
ENERGY = Dimensions(mass=1, length=2, time=-2)
POWER = Dimensions(mass=1, length=2, time=-3)
DENSITY = Dimensions(mass=1, length=-3)
DYNAMIC_VISCOSITY = Dimensions(mass=1, length=-1, time=-1)
KINEMATIC_VISCOSITY = Dimensions(length=2, time=-1)
VOLUMETRIC_FLOW = Dimensions(length=3, time=-1)
MASS_FLOW = Dimensions(mass=1, time=-1)
VOLTAGE = Dimensions(mass=1, length=2, time=-3, current=-1)
RESISTANCE = Dimensions(mass=1, length=2, time=-3, current=-2)
CAPACITANCE = Dimensions(mass=-1, length=-2, time=4, current=2)
INDUCTANCE = Dimensions(mass=1, length=2, time=-2, current=-2)
MAGNETIC_FLUX = Dimensions(mass=1, length=2, time=-2, current=-1)
MAGNETIC_FIELD = Dimensions(mass=1, time=-2, current=-1)
ENTROPY = Dimensions(mass=1, length=2, time=-2, temperature=-1)
SPECIFIC_HEAT = Dimensions(length=2, time=-2, temperature=-1)
SPECIFIC_ENTHALPY = Dimensions(length=2, time=-2)  # J/kg = m^2/s^2
THERMAL_CONDUCTIVITY = Dimensions(mass=1, length=1, time=-3, temperature=-1)
MOLAR_MASS = Dimensions(mass=1, amount=-1)
CONCENTRATION = Dimensions(amount=1, length=-3)
TORQUE = ENERGY  # Same dimensions as energy (N*m vs J)
ANGULAR_VELOCITY = Dimensions(time=-1)  # rad/s, radians are dimensionless
MOMENT_OF_INERTIA = Dimensions(mass=1, length=2)
MOMENTUM = Dimensions(mass=1, length=1, time=-1)
ANGULAR_MOMENTUM = Dimensions(mass=1, length=2, time=-1)

# Field-specific dimensions
ELECTRIC_FIELD = Dimensions(mass=1, length=1, time=-3, current=-1)  # V/m = N/C
CHARGE = Dimensions(current=1, time=1)  # Coulomb = A*s
PERMITTIVITY = Dimensions(mass=-1, length=-3, time=4, current=2)
PERMEABILITY = Dimensions(mass=1, length=1, time=-2, current=-2)


# =============================================================================
# UNIT DEFINITIONS
# =============================================================================

@dataclass
class UnitDef:
    """Definition of a single unit"""
    symbol: str                    # Primary symbol (e.g., "m")
    name: str                      # Full name (e.g., "meter")
    dimensions: Dimensions         # Physical dimensions
    to_si: float                   # Multiply by this to get SI
    offset: float = 0.0           # For temperature scales (C, F)
    aliases: List[str] = field(default_factory=list)  # Alternative symbols


# Master unit registry
UNITS: Dict[str, UnitDef] = {}


def register_unit(symbol: str, name: str, dimensions: Dimensions,
                  to_si: float, offset: float = 0.0, aliases: List[str] = None):
    """Register a unit in the global registry"""
    unit = UnitDef(symbol, name, dimensions, to_si, offset, aliases or [])
    UNITS[symbol] = unit
    for alias in unit.aliases:
        UNITS[alias] = unit


# -----------------------------------------------------------------------------
# LENGTH
# -----------------------------------------------------------------------------
register_unit("m", "meter", LENGTH, 1.0, aliases=["meter", "meters", "metre"])
register_unit("km", "kilometer", LENGTH, 1000.0, aliases=["kilometer"])
register_unit("cm", "centimeter", LENGTH, 0.01, aliases=["centimeter"])
register_unit("mm", "millimeter", LENGTH, 0.001, aliases=["millimeter"])
register_unit("um", "micrometer", LENGTH, 1e-6, aliases=["micron"])
register_unit("nm", "nanometer", LENGTH, 1e-9, aliases=["nanometer"])
register_unit("in", "inch", LENGTH, 0.0254, aliases=["inch", "inches", "in."])
register_unit("ft", "foot", LENGTH, 0.3048, aliases=["foot", "feet"])
register_unit("yd", "yard", LENGTH, 0.9144, aliases=["yard", "yards"])
register_unit("mi", "mile", LENGTH, 1609.344, aliases=["mile", "miles"])
register_unit("nmi", "nautical mile", LENGTH, 1852.0, aliases=["nautical_mile"])
register_unit("mil", "mil (thou)", LENGTH, 2.54e-5, aliases=["thou"])
register_unit("angstrom", "angstrom", LENGTH, 1e-10, aliases=["A"])

# -----------------------------------------------------------------------------
# MASS
# -----------------------------------------------------------------------------
register_unit("kg", "kilogram", MASS, 1.0, aliases=["kilogram", "kilograms"])
register_unit("g", "gram", MASS, 0.001, aliases=["gram", "grams"])
register_unit("mg", "milligram", MASS, 1e-6, aliases=["milligram"])
register_unit("ug", "microgram", MASS, 1e-9, aliases=["microgram"])
register_unit("t", "metric ton", MASS, 1000.0, aliases=["tonne", "metric_ton"])
register_unit("lb", "pound", MASS, 0.453592, aliases=["pound", "pounds", "lbm"])
register_unit("oz", "ounce", MASS, 0.0283495, aliases=["ounce", "ounces"])
register_unit("ton", "short ton", MASS, 907.185, aliases=["short_ton"])
register_unit("long_ton", "long ton", MASS, 1016.05, aliases=["imperial_ton"])
register_unit("slug", "slug", MASS, 14.5939, aliases=[])
register_unit("grain", "grain", MASS, 6.47989e-5, aliases=["gr"])

# -----------------------------------------------------------------------------
# TIME
# -----------------------------------------------------------------------------
register_unit("s", "second", TIME, 1.0, aliases=["sec", "second", "seconds"])
register_unit("ms", "millisecond", TIME, 0.001, aliases=["millisecond"])
register_unit("us", "microsecond", TIME, 1e-6, aliases=["microsecond"])
register_unit("ns", "nanosecond", TIME, 1e-9, aliases=["nanosecond"])
register_unit("min", "minute", TIME, 60.0, aliases=["minute", "minutes"])
register_unit("hr", "hour", TIME, 3600.0, aliases=["h", "hour", "hours"])
register_unit("d", "day", TIME, 86400.0, aliases=["day", "days"])
register_unit("wk", "week", TIME, 604800.0, aliases=["week", "weeks"])
register_unit("yr", "year", TIME, 31557600.0, aliases=["year", "years"])  # Julian year

# -----------------------------------------------------------------------------
# TEMPERATURE (with offsets)
# -----------------------------------------------------------------------------
register_unit("K", "kelvin", TEMPERATURE, 1.0, 0.0, aliases=["kelvin"])
register_unit("degC", "celsius", TEMPERATURE, 1.0, 273.15, aliases=["C", "celsius"])
register_unit("degF", "fahrenheit", TEMPERATURE, 5/9, 459.67 * 5/9, aliases=["F", "fahrenheit"])
register_unit("R", "rankine", TEMPERATURE, 5/9, 0.0, aliases=["rankine", "degR"])

# Temperature DIFFERENCE (no offset) - for deltaT calculations
register_unit("deltaC", "delta celsius", TEMPERATURE, 1.0, 0.0, aliases=["dC"])
register_unit("deltaF", "delta fahrenheit", TEMPERATURE, 5/9, 0.0, aliases=["dF"])

# -----------------------------------------------------------------------------
# ELECTRIC CURRENT
# -----------------------------------------------------------------------------
register_unit("A", "ampere", CURRENT, 1.0, aliases=["amp", "ampere", "amperes"])
register_unit("mA", "milliampere", CURRENT, 0.001, aliases=["milliamp"])
register_unit("uA", "microampere", CURRENT, 1e-6, aliases=["microamp"])
register_unit("kA", "kiloampere", CURRENT, 1000.0, aliases=["kiloamp"])

# -----------------------------------------------------------------------------
# AMOUNT OF SUBSTANCE
# -----------------------------------------------------------------------------
register_unit("mol", "mole", AMOUNT, 1.0, aliases=["mole", "moles"])
register_unit("kmol", "kilomole", AMOUNT, 1000.0, aliases=["kilomole"])
register_unit("mmol", "millimole", AMOUNT, 0.001, aliases=["millimole"])
register_unit("umol", "micromole", AMOUNT, 1e-6, aliases=["micromole"])

# -----------------------------------------------------------------------------
# AREA
# -----------------------------------------------------------------------------
register_unit("m2", "square meter", AREA, 1.0, aliases=["m^2", "sq_m"])
register_unit("cm2", "square centimeter", AREA, 1e-4, aliases=["cm^2", "sq_cm"])
register_unit("mm2", "square millimeter", AREA, 1e-6, aliases=["mm^2", "sq_mm"])
register_unit("km2", "square kilometer", AREA, 1e6, aliases=["km^2", "sq_km"])
register_unit("in2", "square inch", AREA, 6.4516e-4, aliases=["in^2", "sq_in"])
register_unit("ft2", "square foot", AREA, 0.092903, aliases=["ft^2", "sq_ft"])
register_unit("yd2", "square yard", AREA, 0.836127, aliases=["yd^2", "sq_yd"])
register_unit("acre", "acre", AREA, 4046.86, aliases=["acres"])
register_unit("ha", "hectare", AREA, 10000.0, aliases=["hectare", "hectares"])

# -----------------------------------------------------------------------------
# VOLUME
# -----------------------------------------------------------------------------
register_unit("m3", "cubic meter", VOLUME, 1.0, aliases=["m^3", "cu_m"])
register_unit("cm3", "cubic centimeter", VOLUME, 1e-6, aliases=["cm^3", "cc", "mL", "ml"])
register_unit("mm3", "cubic millimeter", VOLUME, 1e-9, aliases=["mm^3", "cu_mm"])
register_unit("L", "liter", VOLUME, 0.001, aliases=["l", "liter", "litre", "liters"])
register_unit("dL", "deciliter", VOLUME, 1e-4, aliases=["dl", "deciliter"])
register_unit("uL", "microliter", VOLUME, 1e-9, aliases=["microliter"])
register_unit("in3", "cubic inch", VOLUME, 1.6387e-5, aliases=["in^3", "cu_in"])
register_unit("ft3", "cubic foot", VOLUME, 0.0283168, aliases=["ft^3", "cu_ft", "cf"])
register_unit("yd3", "cubic yard", VOLUME, 0.764555, aliases=["yd^3", "cu_yd"])
register_unit("gal", "US gallon", VOLUME, 0.00378541, aliases=["gallon", "gallons", "US_gal"])
register_unit("qt", "US quart", VOLUME, 0.000946353, aliases=["quart", "quarts"])
register_unit("pt", "US pint", VOLUME, 0.000473176, aliases=["pint", "pints"])
register_unit("fl_oz", "US fluid ounce", VOLUME, 2.9574e-5, aliases=["fluid_ounce"])
register_unit("bbl", "barrel (oil)", VOLUME, 0.158987, aliases=["barrel", "barrels"])
register_unit("imp_gal", "imperial gallon", VOLUME, 0.00454609, aliases=["imperial_gallon"])

# -----------------------------------------------------------------------------
# VELOCITY
# -----------------------------------------------------------------------------
register_unit("m/s", "meter per second", VELOCITY, 1.0, aliases=["m/sec", "mps"])
register_unit("km/h", "kilometer per hour", VELOCITY, 1/3.6, aliases=["km/hr", "kph", "kmh"])
register_unit("ft/s", "foot per second", VELOCITY, 0.3048, aliases=["ft/sec", "fps"])
register_unit("ft/min", "foot per minute", VELOCITY, 0.00508, aliases=["fpm"])
register_unit("mph", "mile per hour", VELOCITY, 0.44704, aliases=["mi/h", "mi/hr"])
register_unit("kn", "knot", VELOCITY, 0.514444, aliases=["knot", "knots", "kt"])
register_unit("in/s", "inch per second", VELOCITY, 0.0254, aliases=["in/sec", "ips"])
register_unit("cm/s", "centimeter per second", VELOCITY, 0.01, aliases=["cm/sec"])
register_unit("mm/s", "millimeter per second", VELOCITY, 0.001, aliases=["mm/sec"])

# -----------------------------------------------------------------------------
# ACCELERATION
# -----------------------------------------------------------------------------
register_unit("m/s2", "meter per second squared", ACCELERATION, 1.0, aliases=["m/s^2"])
register_unit("ft/s2", "foot per second squared", ACCELERATION, 0.3048, aliases=["ft/s^2"])
register_unit("g0", "standard gravity", ACCELERATION, 9.80665, aliases=["gee", "G"])
register_unit("Gal", "gal (galileo)", ACCELERATION, 0.01, aliases=["gal"])

# -----------------------------------------------------------------------------
# FREQUENCY
# -----------------------------------------------------------------------------
register_unit("Hz", "hertz", FREQUENCY, 1.0, aliases=["hertz", "hz", "1/s"])
register_unit("kHz", "kilohertz", FREQUENCY, 1000.0, aliases=["kilohertz"])
register_unit("MHz", "megahertz", FREQUENCY, 1e6, aliases=["megahertz"])
register_unit("GHz", "gigahertz", FREQUENCY, 1e9, aliases=["gigahertz"])
register_unit("rpm", "revolutions per minute", FREQUENCY, 1/60, aliases=["RPM", "rev/min"])
register_unit("rps", "revolutions per second", FREQUENCY, 1.0, aliases=["rev/s"])
register_unit("rad/s", "radians per second", ANGULAR_VELOCITY, 1.0, aliases=["rad/sec"])
register_unit("deg/s", "degrees per second", ANGULAR_VELOCITY, math.pi/180, aliases=["deg/sec"])

# -----------------------------------------------------------------------------
# FORCE
# -----------------------------------------------------------------------------
register_unit("N", "newton", FORCE, 1.0, aliases=["newton", "newtons"])
register_unit("kN", "kilonewton", FORCE, 1000.0, aliases=["kilonewton"])
register_unit("MN", "meganewton", FORCE, 1e6, aliases=["meganewton"])
register_unit("mN", "millinewton", FORCE, 0.001, aliases=["millinewton"])
register_unit("uN", "micronewton", FORCE, 1e-6, aliases=["micronewton"])
register_unit("dyn", "dyne", FORCE, 1e-5, aliases=["dyne"])
register_unit("lbf", "pound-force", FORCE, 4.44822, aliases=["pound_force", "lb_f"])
register_unit("kip", "kilopound-force", FORCE, 4448.22, aliases=["kilopound"])
register_unit("kgf", "kilogram-force", FORCE, 9.80665, aliases=["kilogram_force", "kp"])
register_unit("ozf", "ounce-force", FORCE, 0.278014, aliases=["ounce_force"])

# -----------------------------------------------------------------------------
# PRESSURE / STRESS
# -----------------------------------------------------------------------------
register_unit("Pa", "pascal", PRESSURE, 1.0, aliases=["pascal", "N/m2", "N/m^2"])
register_unit("kPa", "kilopascal", PRESSURE, 1000.0, aliases=["kilopascal"])
register_unit("MPa", "megapascal", PRESSURE, 1e6, aliases=["megapascal"])
register_unit("GPa", "gigapascal", PRESSURE, 1e9, aliases=["gigapascal"])
register_unit("hPa", "hectopascal", PRESSURE, 100.0, aliases=["hectopascal"])
register_unit("bar", "bar", PRESSURE, 1e5, aliases=["bars"])
register_unit("mbar", "millibar", PRESSURE, 100.0, aliases=["millibar"])
register_unit("atm", "atmosphere", PRESSURE, 101325.0, aliases=["atmosphere", "atmospheres"])
register_unit("psi", "pound per square inch", PRESSURE, 6894.76, aliases=["lb/in2", "lbf/in2", "PSI", "psia", "psig"])
register_unit("ksi", "kilopound per square inch", PRESSURE, 6.89476e6, aliases=["KSI"])
register_unit("psf", "pound per square foot", PRESSURE, 47.8803, aliases=["lb/ft2", "lbf/ft2"])
register_unit("torr", "torr", PRESSURE, 133.322, aliases=["Torr"])
register_unit("mmHg", "millimeter of mercury", PRESSURE, 133.322, aliases=["mm_Hg"])
register_unit("inHg", "inch of mercury", PRESSURE, 3386.39, aliases=["in_Hg"])
register_unit("mmH2O", "millimeter of water", PRESSURE, 9.80665, aliases=["mm_H2O"])
register_unit("inH2O", "inch of water", PRESSURE, 249.089, aliases=["in_H2O", "iwc", "wc"])
register_unit("ftH2O", "foot of water", PRESSURE, 2989.07, aliases=["ft_H2O"])

# -----------------------------------------------------------------------------
# ENERGY / WORK / HEAT
# -----------------------------------------------------------------------------
register_unit("J", "joule", ENERGY, 1.0, aliases=["joule", "joules", "N*m"])
register_unit("kJ", "kilojoule", ENERGY, 1000.0, aliases=["kilojoule"])
register_unit("MJ", "megajoule", ENERGY, 1e6, aliases=["megajoule"])
register_unit("GJ", "gigajoule", ENERGY, 1e9, aliases=["gigajoule"])
register_unit("mJ", "millijoule", ENERGY, 0.001, aliases=["millijoule"])
register_unit("Wh", "watt-hour", ENERGY, 3600.0, aliases=["watt_hour"])
register_unit("kWh", "kilowatt-hour", ENERGY, 3.6e6, aliases=["kilowatt_hour"])
register_unit("MWh", "megawatt-hour", ENERGY, 3.6e9, aliases=["megawatt_hour"])
register_unit("cal", "calorie (thermochemical)", ENERGY, 4.184, aliases=["calorie"])
register_unit("kcal", "kilocalorie", ENERGY, 4184.0, aliases=["kilocalorie", "Cal", "Calorie"])
register_unit("BTU", "British thermal unit", ENERGY, 1055.06, aliases=["Btu", "btu"])
register_unit("therm", "therm", ENERGY, 1.055e8, aliases=["therms"])
register_unit("erg", "erg", ENERGY, 1e-7, aliases=["ergs"])
register_unit("eV", "electronvolt", ENERGY, 1.60218e-19, aliases=["electronvolt"])
register_unit("keV", "kiloelectronvolt", ENERGY, 1.60218e-16, aliases=[])
register_unit("MeV", "megaelectronvolt", ENERGY, 1.60218e-13, aliases=[])
register_unit("ft*lbf", "foot-pound", ENERGY, 1.35582, aliases=["ft-lb", "ft_lb"])
register_unit("in*lbf", "inch-pound", ENERGY, 0.112985, aliases=["in-lb", "in_lb"])

# -----------------------------------------------------------------------------
# POWER
# -----------------------------------------------------------------------------
register_unit("W", "watt", POWER, 1.0, aliases=["watt", "watts", "J/s"])
register_unit("kW", "kilowatt", POWER, 1000.0, aliases=["kilowatt"])
register_unit("MW", "megawatt", POWER, 1e6, aliases=["megawatt"])
register_unit("GW", "gigawatt", POWER, 1e9, aliases=["gigawatt"])
register_unit("mW", "milliwatt", POWER, 0.001, aliases=["milliwatt"])
register_unit("uW", "microwatt", POWER, 1e-6, aliases=["microwatt"])
register_unit("hp", "horsepower (mechanical)", POWER, 745.7, aliases=["horsepower", "HP"])
register_unit("hp_e", "horsepower (electric)", POWER, 746.0, aliases=["HP_e"])
register_unit("hp_m", "horsepower (metric)", POWER, 735.499, aliases=["PS", "cv", "pk"])
register_unit("BTU/h", "BTU per hour", POWER, 0.293071, aliases=["BTU/hr", "Btu/h"])
register_unit("BTU/s", "BTU per second", POWER, 1055.06, aliases=["Btu/s"])
register_unit("ton_ref", "ton of refrigeration", POWER, 3516.85, aliases=["TR", "ton_cooling"])
register_unit("ft*lbf/s", "foot-pound per second", POWER, 1.35582, aliases=[])

# -----------------------------------------------------------------------------
# DENSITY
# -----------------------------------------------------------------------------
register_unit("kg/m3", "kilogram per cubic meter", DENSITY, 1.0, aliases=["kg/m^3"])
register_unit("g/cm3", "gram per cubic centimeter", DENSITY, 1000.0, aliases=["g/cm^3", "g/cc", "g/mL"])
register_unit("g/L", "gram per liter", DENSITY, 1.0, aliases=["g/l"])
register_unit("kg/L", "kilogram per liter", DENSITY, 1000.0, aliases=["kg/l"])
register_unit("lb/ft3", "pound per cubic foot", DENSITY, 16.0185, aliases=["lb/ft^3", "pcf"])
register_unit("lb/in3", "pound per cubic inch", DENSITY, 27679.9, aliases=["lb/in^3"])
register_unit("lb/gal", "pound per gallon (US)", DENSITY, 119.826, aliases=["lb/US_gal"])
register_unit("oz/in3", "ounce per cubic inch", DENSITY, 1729.99, aliases=["oz/in^3"])
register_unit("slug/ft3", "slug per cubic foot", DENSITY, 515.379, aliases=["slug/ft^3"])

# Specific gravity (dimensionless but often used)
register_unit("SG", "specific gravity", DIMENSIONLESS, 1.0, aliases=["sg", "sp_gr"])
register_unit("API", "API gravity", DIMENSIONLESS, 1.0, aliases=["degAPI"])  # Needs special conversion

# -----------------------------------------------------------------------------
# DYNAMIC VISCOSITY
# -----------------------------------------------------------------------------
register_unit("Pa*s", "pascal-second", DYNAMIC_VISCOSITY, 1.0, aliases=["Pa-s", "N*s/m2"])
register_unit("mPa*s", "millipascal-second", DYNAMIC_VISCOSITY, 0.001, aliases=["mPa-s"])
register_unit("P", "poise", DYNAMIC_VISCOSITY, 0.1, aliases=["poise"])
register_unit("cP", "centipoise", DYNAMIC_VISCOSITY, 0.001, aliases=["centipoise", "cp"])
register_unit("lb/(ft*s)", "pound per foot-second", DYNAMIC_VISCOSITY, 1.48816, aliases=["lb/ft/s"])
register_unit("lb/(ft*h)", "pound per foot-hour", DYNAMIC_VISCOSITY, 0.000413379, aliases=["lb/ft/h"])
register_unit("lbf*s/ft2", "pound-force second per square foot", DYNAMIC_VISCOSITY, 47.8803, aliases=[])
register_unit("reyn", "reyn", DYNAMIC_VISCOSITY, 6894.76, aliases=[])

# -----------------------------------------------------------------------------
# KINEMATIC VISCOSITY
# -----------------------------------------------------------------------------
register_unit("m2/s", "square meter per second", KINEMATIC_VISCOSITY, 1.0, aliases=["m^2/s"])
register_unit("mm2/s", "square millimeter per second", KINEMATIC_VISCOSITY, 1e-6, aliases=["mm^2/s"])
register_unit("St", "stokes", KINEMATIC_VISCOSITY, 1e-4, aliases=["stokes", "stoke"])
register_unit("cSt", "centistokes", KINEMATIC_VISCOSITY, 1e-6, aliases=["centistokes", "centistoke"])
register_unit("ft2/s", "square foot per second", KINEMATIC_VISCOSITY, 0.092903, aliases=["ft^2/s"])
register_unit("ft2/h", "square foot per hour", KINEMATIC_VISCOSITY, 2.58064e-5, aliases=["ft^2/h"])
register_unit("in2/s", "square inch per second", KINEMATIC_VISCOSITY, 6.4516e-4, aliases=["in^2/s"])

# -----------------------------------------------------------------------------
# VOLUMETRIC FLOW RATE
# -----------------------------------------------------------------------------
register_unit("m3/s", "cubic meter per second", VOLUMETRIC_FLOW, 1.0, aliases=["m^3/s", "cms"])
register_unit("m3/h", "cubic meter per hour", VOLUMETRIC_FLOW, 1/3600, aliases=["m^3/h", "cmh"])
register_unit("m3/d", "cubic meter per day", VOLUMETRIC_FLOW, 1/86400, aliases=["m^3/d"])
register_unit("L/s", "liter per second", VOLUMETRIC_FLOW, 0.001, aliases=["l/s", "lps", "L/sec"])
register_unit("L/min", "liter per minute", VOLUMETRIC_FLOW, 1.66667e-5, aliases=["l/min", "lpm", "L/m"])
register_unit("L/h", "liter per hour", VOLUMETRIC_FLOW, 2.77778e-7, aliases=["l/h", "lph"])
register_unit("mL/min", "milliliter per minute", VOLUMETRIC_FLOW, 1.66667e-8, aliases=["ml/min"])
register_unit("gpm", "gallon per minute (US)", VOLUMETRIC_FLOW, 6.30902e-5, aliases=["GPM", "gal/min"])
register_unit("gph", "gallon per hour (US)", VOLUMETRIC_FLOW, 1.05150e-6, aliases=["GPH", "gal/h"])
register_unit("gpd", "gallon per day (US)", VOLUMETRIC_FLOW, 4.38126e-8, aliases=["GPD", "gal/d"])
register_unit("MGD", "million gallons per day", VOLUMETRIC_FLOW, 0.0438126, aliases=["mgd"])
register_unit("cfm", "cubic foot per minute", VOLUMETRIC_FLOW, 4.71947e-4, aliases=["CFM", "ft3/min", "ft^3/min"])
register_unit("cfs", "cubic foot per second", VOLUMETRIC_FLOW, 0.0283168, aliases=["CFS", "ft3/s", "ft^3/s"])
register_unit("cfh", "cubic foot per hour", VOLUMETRIC_FLOW, 7.86579e-6, aliases=["CFH", "ft3/h"])
register_unit("bbl/d", "barrel per day", VOLUMETRIC_FLOW, 1.84013e-6, aliases=["bpd", "BOPD", "bbl/day"])
register_unit("bbl/h", "barrel per hour", VOLUMETRIC_FLOW, 4.41631e-5, aliases=["bph"])
register_unit("scfm", "standard cubic foot per minute", VOLUMETRIC_FLOW, 4.71947e-4, aliases=["SCFM"])
register_unit("scfh", "standard cubic foot per hour", VOLUMETRIC_FLOW, 7.86579e-6, aliases=["SCFH"])
register_unit("Nm3/h", "normal cubic meter per hour", VOLUMETRIC_FLOW, 1/3600, aliases=["Nm3/h"])

# -----------------------------------------------------------------------------
# MASS FLOW RATE
# -----------------------------------------------------------------------------
register_unit("kg/s", "kilogram per second", MASS_FLOW, 1.0, aliases=["kg/sec"])
register_unit("kg/h", "kilogram per hour", MASS_FLOW, 1/3600, aliases=["kg/hr"])
register_unit("kg/min", "kilogram per minute", MASS_FLOW, 1/60, aliases=[])
register_unit("kg/d", "kilogram per day", MASS_FLOW, 1/86400, aliases=[])
register_unit("g/s", "gram per second", MASS_FLOW, 0.001, aliases=["g/sec"])
register_unit("g/min", "gram per minute", MASS_FLOW, 1.66667e-5, aliases=[])
register_unit("t/h", "metric ton per hour", MASS_FLOW, 1000/3600, aliases=["tonne/h"])
register_unit("t/d", "metric ton per day", MASS_FLOW, 1000/86400, aliases=["tonne/d", "tpd"])
register_unit("lb/s", "pound per second", MASS_FLOW, 0.453592, aliases=["lb/sec", "lbm/s", "lbm/sec"])
register_unit("lb/min", "pound per minute", MASS_FLOW, 0.00755987, aliases=["lb/m"])
register_unit("lb/h", "pound per hour", MASS_FLOW, 0.000125998, aliases=["lb/hr", "pph"])
register_unit("lb/d", "pound per day", MASS_FLOW, 5.24991e-6, aliases=["ppd"])
register_unit("ton/h", "short ton per hour", MASS_FLOW, 0.251996, aliases=["short_ton/h"])

# -----------------------------------------------------------------------------
# VOLTAGE (Electric Potential)
# -----------------------------------------------------------------------------
register_unit("V", "volt", VOLTAGE, 1.0, aliases=["volt", "volts"])
register_unit("kV", "kilovolt", VOLTAGE, 1000.0, aliases=["kilovolt"])
register_unit("MV", "megavolt", VOLTAGE, 1e6, aliases=["megavolt"])
register_unit("mV", "millivolt", VOLTAGE, 0.001, aliases=["millivolt"])
register_unit("uV", "microvolt", VOLTAGE, 1e-6, aliases=["microvolt"])

# -----------------------------------------------------------------------------
# RESISTANCE
# -----------------------------------------------------------------------------
register_unit("ohm", "ohm", RESISTANCE, 1.0, aliases=["ohms"])
register_unit("kohm", "kilohm", RESISTANCE, 1000.0, aliases=["kilohm"])
register_unit("Mohm", "megohm", RESISTANCE, 1e6, aliases=["megohm"])
register_unit("mohm", "milliohm", RESISTANCE, 0.001, aliases=["milliohm"])
register_unit("uohm", "microhm", RESISTANCE, 1e-6, aliases=["microhm"])

# -----------------------------------------------------------------------------
# CAPACITANCE
# -----------------------------------------------------------------------------
register_unit("F", "farad", CAPACITANCE, 1.0, aliases=["farad", "farads"])
register_unit("mF", "millifarad", CAPACITANCE, 0.001, aliases=["millifarad"])
register_unit("uF", "microfarad", CAPACITANCE, 1e-6, aliases=["microfarad"])
register_unit("nF", "nanofarad", CAPACITANCE, 1e-9, aliases=["nanofarad"])
register_unit("pF", "picofarad", CAPACITANCE, 1e-12, aliases=["picofarad"])

# -----------------------------------------------------------------------------
# INDUCTANCE
# -----------------------------------------------------------------------------
register_unit("H", "henry", INDUCTANCE, 1.0, aliases=["henry", "henries"])
register_unit("mH", "millihenry", INDUCTANCE, 0.001, aliases=["millihenry"])
register_unit("uH", "microhenry", INDUCTANCE, 1e-6, aliases=["microhenry"])
register_unit("nH", "nanohenry", INDUCTANCE, 1e-9, aliases=["nanohenry"])

# -----------------------------------------------------------------------------
# MAGNETIC
# -----------------------------------------------------------------------------
register_unit("T", "tesla", MAGNETIC_FIELD, 1.0, aliases=["tesla"])
register_unit("mT", "millitesla", MAGNETIC_FIELD, 0.001, aliases=["millitesla"])
register_unit("uT", "microtesla", MAGNETIC_FIELD, 1e-6, aliases=["microtesla"])
register_unit("gauss", "gauss", MAGNETIC_FIELD, 1e-4, aliases=["G"])
register_unit("Wb", "weber", MAGNETIC_FLUX, 1.0, aliases=["weber"])

# -----------------------------------------------------------------------------
# THERMAL
# -----------------------------------------------------------------------------
register_unit("W/(m*K)", "watt per meter-kelvin", THERMAL_CONDUCTIVITY, 1.0,
              aliases=["W/m/K", "W/m*K"])
register_unit("BTU/(h*ft*F)", "BTU per hour-foot-fahrenheit", THERMAL_CONDUCTIVITY, 1.73073,
              aliases=["BTU/h/ft/F"])
register_unit("J/(kg*K)", "joule per kilogram-kelvin", SPECIFIC_HEAT, 1.0,
              aliases=["J/kg/K"])
register_unit("kJ/(kg*K)", "kilojoule per kilogram-kelvin", SPECIFIC_HEAT, 1000.0,
              aliases=["kJ/kg/K"])
register_unit("BTU/(lb*F)", "BTU per pound-fahrenheit", SPECIFIC_HEAT, 4186.8,
              aliases=["BTU/lb/F"])
register_unit("cal/(g*C)", "calorie per gram-celsius", SPECIFIC_HEAT, 4184.0,
              aliases=["cal/g/C"])

# -----------------------------------------------------------------------------
# SPECIFIC ENTHALPY (Energy per mass)
# -----------------------------------------------------------------------------
register_unit("J/kg", "joule per kilogram", SPECIFIC_ENTHALPY, 1.0, aliases=["joule/kg"])
register_unit("kJ/kg", "kilojoule per kilogram", SPECIFIC_ENTHALPY, 1000.0, aliases=["kilojoule/kg"])
register_unit("MJ/kg", "megajoule per kilogram", SPECIFIC_ENTHALPY, 1e6, aliases=["megajoule/kg"])
register_unit("BTU/lb", "BTU per pound", SPECIFIC_ENTHALPY, 2326.0, aliases=["BTU/lbm", "Btu/lb", "btu/lb"])
register_unit("cal/g", "calorie per gram", SPECIFIC_ENTHALPY, 4184.0, aliases=["calorie/g"])
register_unit("kcal/kg", "kilocalorie per kilogram", SPECIFIC_ENTHALPY, 4184.0, aliases=["Cal/kg"])

register_unit("W/m2", "watt per square meter", Dimensions(mass=1, time=-3), 1.0,
              aliases=["W/m^2"])
register_unit("BTU/(h*ft2)", "BTU per hour-square foot", Dimensions(mass=1, time=-3), 3.15459,
              aliases=["BTU/h/ft^2"])

# -----------------------------------------------------------------------------
# CONCENTRATION / MOLAR
# -----------------------------------------------------------------------------
register_unit("mol/m3", "mole per cubic meter", CONCENTRATION, 1.0, aliases=["mol/m^3"])
register_unit("mol/L", "mole per liter", CONCENTRATION, 1000.0, aliases=["M", "molar", "mol/l"])
register_unit("mmol/L", "millimole per liter", CONCENTRATION, 1.0, aliases=["mM", "mmolar"])
register_unit("umol/L", "micromole per liter", CONCENTRATION, 0.001, aliases=["uM"])
register_unit("kmol/m3", "kilomole per cubic meter", CONCENTRATION, 1000.0, aliases=["kmol/m^3"])
register_unit("g/mol", "gram per mole", MOLAR_MASS, 0.001, aliases=["g/mole"])
register_unit("kg/mol", "kilogram per mole", MOLAR_MASS, 1.0, aliases=["kg/mole"])
register_unit("kg/kmol", "kilogram per kilomole", MOLAR_MASS, 0.001, aliases=[])

# -----------------------------------------------------------------------------
# TORQUE
# -----------------------------------------------------------------------------
register_unit("N*m", "newton-meter", TORQUE, 1.0, aliases=["Nm", "N-m"])
register_unit("kN*m", "kilonewton-meter", TORQUE, 1000.0, aliases=["kNm"])
register_unit("mN*m", "millinewton-meter", TORQUE, 0.001, aliases=["mNm"])
register_unit("kgf*m", "kilogram-force meter", TORQUE, 9.80665, aliases=["kgm"])
register_unit("lbf*ft", "pound-force foot", TORQUE, 1.35582, aliases=["lb-ft", "ft-lb"])
register_unit("lbf*in", "pound-force inch", TORQUE, 0.112985, aliases=["lb-in", "in-lb"])
register_unit("ozf*in", "ounce-force inch", TORQUE, 0.00706155, aliases=["oz-in"])

# -----------------------------------------------------------------------------
# MOMENT OF INERTIA
# -----------------------------------------------------------------------------
register_unit("kg*m2", "kilogram square meter", MOMENT_OF_INERTIA, 1.0, aliases=["kg*m^2"])
register_unit("g*cm2", "gram square centimeter", MOMENT_OF_INERTIA, 1e-7, aliases=["g*cm^2"])
register_unit("lb*ft2", "pound square foot", MOMENT_OF_INERTIA, 0.0421401, aliases=["lb*ft^2"])
register_unit("lb*in2", "pound square inch", MOMENT_OF_INERTIA, 2.9264e-4, aliases=["lb*in^2"])
register_unit("slug*ft2", "slug square foot", MOMENT_OF_INERTIA, 1.35582, aliases=["slug*ft^2"])

# -----------------------------------------------------------------------------
# MOMENTUM
# -----------------------------------------------------------------------------
register_unit("kg*m/s", "kilogram meter per second", MOMENTUM, 1.0, aliases=["N*s"])
register_unit("lb*ft/s", "pound foot per second", MOMENTUM, 0.138255, aliases=[])

# -----------------------------------------------------------------------------
# ANGULAR MOMENTUM
# -----------------------------------------------------------------------------
register_unit("kg*m2/s", "kilogram square meter per second", ANGULAR_MOMENTUM, 1.0, aliases=["J*s"])

# -----------------------------------------------------------------------------
# ANGLE (dimensionless but useful)
# -----------------------------------------------------------------------------
register_unit("rad", "radian", DIMENSIONLESS, 1.0, aliases=["radian", "radians"])
register_unit("deg", "degree", DIMENSIONLESS, math.pi/180, aliases=["degree", "degrees"])
register_unit("arcmin", "arcminute", DIMENSIONLESS, math.pi/10800, aliases=["arcminute"])
register_unit("arcsec", "arcsecond", DIMENSIONLESS, math.pi/648000, aliases=["arcsecond"])
register_unit("rev", "revolution", DIMENSIONLESS, 2*math.pi, aliases=["revolution", "turn"])
register_unit("grad", "gradian", DIMENSIONLESS, math.pi/200, aliases=["gon", "gradian"])

# -----------------------------------------------------------------------------
# DIMENSIONLESS RATIOS
# -----------------------------------------------------------------------------
register_unit("%", "percent", DIMENSIONLESS, 0.01, aliases=["percent", "pct"])
register_unit("permille", "permille", DIMENSIONLESS, 0.001, aliases=["ppt"])
register_unit("ppm", "parts per million", DIMENSIONLESS, 1e-6, aliases=["PPM"])
register_unit("ppb", "parts per billion", DIMENSIONLESS, 1e-9, aliases=["PPB"])
register_unit("dB", "decibel", DIMENSIONLESS, 1.0, aliases=["decibel"])  # Logarithmic, special handling needed

# -----------------------------------------------------------------------------
# ELECTROMAGNETIC FIELD CONSTANTS (for Level 4)
# -----------------------------------------------------------------------------
register_unit("V/m", "volt per meter", ELECTRIC_FIELD, 1.0, aliases=["N/C"])
register_unit("kV/m", "kilovolt per meter", ELECTRIC_FIELD, 1000.0, aliases=[])
register_unit("C", "coulomb", CHARGE, 1.0, aliases=["coulomb"])
register_unit("mC", "millicoulomb", CHARGE, 0.001, aliases=[])
register_unit("uC", "microcoulomb", CHARGE, 1e-6, aliases=[])
register_unit("nC", "nanocoulomb", CHARGE, 1e-9, aliases=[])
register_unit("F/m", "farad per meter", PERMITTIVITY, 1.0, aliases=[])
register_unit("H/m", "henry per meter", PERMEABILITY, 1.0, aliases=[])


# =============================================================================
# PHYSICAL CONSTANTS (for PRISM physics/fields stages)
# =============================================================================

PHYSICAL_CONSTANTS = {
    # Fundamental
    'c': (299792458.0, 'm/s', 'speed of light in vacuum'),
    'h': (6.62607015e-34, 'J*s', 'Planck constant'),
    'hbar': (1.054571817e-34, 'J*s', 'reduced Planck constant'),
    'G': (6.67430e-11, 'm3/(kg*s2)', 'gravitational constant'),
    'e': (1.602176634e-19, 'C', 'elementary charge'),
    'epsilon_0': (8.8541878128e-12, 'F/m', 'vacuum permittivity'),
    'mu_0': (1.25663706212e-6, 'H/m', 'vacuum permeability'),
    'k_B': (1.380649e-23, 'J/K', 'Boltzmann constant'),
    'N_A': (6.02214076e23, '1/mol', 'Avogadro constant'),
    'R': (8.314462618, 'J/(mol*K)', 'gas constant'),

    # Derived / Common
    'g': (9.80665, 'm/s2', 'standard gravity'),
    'atm': (101325.0, 'Pa', 'standard atmosphere'),
    'sigma': (5.670374419e-8, 'W/(m2*K4)', 'Stefan-Boltzmann constant'),

    # Water properties (at 20C, 1 atm)
    'rho_water': (998.2, 'kg/m3', 'water density at 20C'),
    'mu_water': (0.001002, 'Pa*s', 'water dynamic viscosity at 20C'),
    'nu_water': (1.004e-6, 'm2/s', 'water kinematic viscosity at 20C'),
    'Cp_water': (4182.0, 'J/(kg*K)', 'water specific heat at 20C'),

    # Air properties (at 20C, 1 atm)
    'rho_air': (1.204, 'kg/m3', 'air density at 20C'),
    'mu_air': (1.825e-5, 'Pa*s', 'air dynamic viscosity at 20C'),
    'nu_air': (1.516e-5, 'm2/s', 'air kinematic viscosity at 20C'),
    'Cp_air': (1005.0, 'J/(kg*K)', 'air specific heat at 20C'),
    'gamma_air': (1.4, '', 'air heat capacity ratio'),
}


def get_constant(name: str) -> 'Quantity':
    """Get a physical constant as a Quantity."""
    if name not in PHYSICAL_CONSTANTS:
        raise ValueError(f"Unknown constant: {name}. Available: {list(PHYSICAL_CONSTANTS.keys())}")
    value, unit, _ = PHYSICAL_CONSTANTS[name]
    if unit:
        return Quantity(value, unit)
    else:
        return value  # Dimensionless


# =============================================================================
# QUANTITY CLASS
# =============================================================================

class Quantity:
    """
    A physical quantity with value, unit, and dimensional tracking.

    Examples:
        >>> Q(100, "gpm").to("m3/s")
        6.30902e-05

        >>> Q("4 in").si
        0.1016

        >>> flow = Q(100, "gpm")
        >>> area = Q(0.01, "m2")
        >>> velocity = flow / area
        >>> velocity.to("ft/s")
        2.07...
    """

    def __init__(self, value: Union[float, str], unit: str = None):
        """
        Create a quantity.

        Args:
            value: Numeric value, or string like "4 in" or "100 gpm"
            unit: Unit string (optional if value is a string with unit)
        """
        if isinstance(value, str) and unit is None:
            # Parse "4 in" or "100.5 gpm"
            value, unit = self._parse_string(value)

        if unit is None:
            raise ValueError("Unit must be specified")

        self.value = float(value)
        self.unit_str = unit

        # Look up unit definition
        unit_def = self._get_unit(unit)
        self.unit_def = unit_def
        self.dimensions = unit_def.dimensions

        # Convert to SI
        self._si_value = self.value * unit_def.to_si + unit_def.offset

    @staticmethod
    def _parse_string(s: str) -> Tuple[float, str]:
        """Parse '4 in' or '100.5 gpm' into (value, unit)"""
        s = s.strip()

        # Match number (including scientific notation) + unit
        match = re.match(r'^([+-]?[\d.]+(?:[eE][+-]?\d+)?)\s*(.+)', s)
        if not match:
            raise ValueError(f"Cannot parse quantity string: '{s}'")

        return float(match.group(1)), match.group(2).strip()

    @staticmethod
    def _get_unit(unit_str: str) -> UnitDef:
        """Look up unit definition, with fuzzy matching"""
        # Direct lookup
        if unit_str in UNITS:
            return UNITS[unit_str]

        # Try lowercase
        if unit_str.lower() in UNITS:
            return UNITS[unit_str.lower()]

        # Try removing spaces
        no_space = unit_str.replace(" ", "")
        if no_space in UNITS:
            return UNITS[no_space]

        raise ValueError(f"Unknown unit: '{unit_str}'")

    @property
    def si(self) -> float:
        """Value in SI base units"""
        return self._si_value

    def to(self, target_unit: str) -> float:
        """
        Convert to target unit.

        Args:
            target_unit: Unit string to convert to

        Returns:
            Numeric value in target units

        Raises:
            ValueError: If units are incompatible
        """
        target_def = self._get_unit(target_unit)

        # Check dimensional compatibility
        if self.dimensions != target_def.dimensions:
            raise ValueError(
                f"Cannot convert {self.unit_str} ({self.dimensions}) "
                f"to {target_unit} ({target_def.dimensions})"
            )

        # Convert: SI value -> target unit
        return (self._si_value - target_def.offset) / target_def.to_si

    def is_compatible(self, target_unit: str) -> bool:
        """Check if conversion to target unit is possible"""
        try:
            target_def = self._get_unit(target_unit)
            return self.dimensions == target_def.dimensions
        except ValueError:
            return False

    def as_unit(self, target_unit: str) -> 'Quantity':
        """Return new Quantity in target units"""
        return Quantity(self.to(target_unit), target_unit)

    # Arithmetic operations with dimensional tracking

    def __add__(self, other: 'Quantity') -> 'Quantity':
        if not isinstance(other, Quantity):
            raise TypeError(f"Cannot add Quantity and {type(other)}")
        if self.dimensions != other.dimensions:
            raise ValueError(f"Cannot add {self.dimensions} and {other.dimensions}")

        # Add in SI, return in first unit
        new_si = self._si_value + other._si_value
        new_value = (new_si - self.unit_def.offset) / self.unit_def.to_si
        return Quantity(new_value, self.unit_str)

    def __sub__(self, other: 'Quantity') -> 'Quantity':
        if not isinstance(other, Quantity):
            raise TypeError(f"Cannot subtract Quantity and {type(other)}")
        if self.dimensions != other.dimensions:
            raise ValueError(f"Cannot subtract {self.dimensions} and {other.dimensions}")

        new_si = self._si_value - other._si_value
        new_value = (new_si - self.unit_def.offset) / self.unit_def.to_si
        return Quantity(new_value, self.unit_str)

    def __mul__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        if isinstance(other, (int, float)):
            return Quantity(self.value * other, self.unit_str)

        if isinstance(other, Quantity):
            # Multiply SI values, combine dimensions
            new_si = self._si_value * other._si_value
            new_dims = self.dimensions * other.dimensions

            # Find or create appropriate unit
            unit_str = self._find_unit_for_dims(new_dims) or f"({self.unit_str})*({other.unit_str})"

            # Create result (hacky: we inject the SI value directly)
            result = object.__new__(Quantity)
            result.value = new_si
            result.unit_str = unit_str
            result._si_value = new_si
            result.dimensions = new_dims
            result.unit_def = UnitDef(unit_str, "derived", new_dims, 1.0)
            return result

        raise TypeError(f"Cannot multiply Quantity and {type(other)}")

    def __rmul__(self, other: Union[float, int]) -> 'Quantity':
        return self.__mul__(other)

    def __truediv__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        if isinstance(other, (int, float)):
            return Quantity(self.value / other, self.unit_str)

        if isinstance(other, Quantity):
            new_si = self._si_value / other._si_value
            new_dims = self.dimensions / other.dimensions

            unit_str = self._find_unit_for_dims(new_dims) or f"({self.unit_str})/({other.unit_str})"

            result = object.__new__(Quantity)
            result.value = new_si
            result.unit_str = unit_str
            result._si_value = new_si
            result.dimensions = new_dims
            result.unit_def = UnitDef(unit_str, "derived", new_dims, 1.0)
            return result

        raise TypeError(f"Cannot divide Quantity by {type(other)}")

    def __pow__(self, exp: int) -> 'Quantity':
        new_si = self._si_value ** exp
        new_dims = self.dimensions ** exp

        unit_str = self._find_unit_for_dims(new_dims) or f"({self.unit_str})^{exp}"

        result = object.__new__(Quantity)
        result.value = new_si
        result.unit_str = unit_str
        result._si_value = new_si
        result.dimensions = new_dims
        result.unit_def = UnitDef(unit_str, "derived", new_dims, 1.0)
        return result

    def __neg__(self) -> 'Quantity':
        return Quantity(-self.value, self.unit_str)

    def __abs__(self) -> 'Quantity':
        return Quantity(abs(self.value), self.unit_str)

    @staticmethod
    def _find_unit_for_dims(dims: Dimensions) -> Optional[str]:
        """Find a standard unit symbol for given dimensions"""
        # Check common derived dimensions
        dim_to_unit = {
            VELOCITY: "m/s",
            ACCELERATION: "m/s2",
            FORCE: "N",
            PRESSURE: "Pa",
            ENERGY: "J",
            POWER: "W",
            DENSITY: "kg/m3",
            DYNAMIC_VISCOSITY: "Pa*s",
            KINEMATIC_VISCOSITY: "m2/s",
            VOLUMETRIC_FLOW: "m3/s",
            MASS_FLOW: "kg/s",
            AREA: "m2",
            VOLUME: "m3",
            DIMENSIONLESS: "1",
            MOMENTUM: "kg*m/s",
            ANGULAR_MOMENTUM: "kg*m2/s",
            TORQUE: "N*m",
            ENTROPY: "J/K",
        }
        return dim_to_unit.get(dims)

    def __repr__(self) -> str:
        return f"Q({self.value}, '{self.unit_str}')"

    def __str__(self) -> str:
        if abs(self.value) < 0.001 or abs(self.value) > 10000:
            return f"{self.value:.4e} {self.unit_str}"
        return f"{self.value:.4f} {self.unit_str}"


# Convenience alias
Q = Quantity


# =============================================================================
# UNIT CATEGORY HELPERS
# =============================================================================

class UnitCategory(Enum):
    """Categories for UI dropdowns and validation"""
    LENGTH = "length"
    MASS = "mass"
    TIME = "time"
    TEMPERATURE = "temperature"
    CURRENT = "current"
    AMOUNT = "amount"
    AREA = "area"
    VOLUME = "volume"
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"
    FREQUENCY = "frequency"
    FORCE = "force"
    PRESSURE = "pressure"
    ENERGY = "energy"
    POWER = "power"
    DENSITY = "density"
    DYNAMIC_VISCOSITY = "dynamic_viscosity"
    KINEMATIC_VISCOSITY = "kinematic_viscosity"
    VOLUMETRIC_FLOW = "volumetric_flow"
    MASS_FLOW = "mass_flow"
    VOLTAGE = "voltage"
    RESISTANCE = "resistance"
    CAPACITANCE = "capacitance"
    INDUCTANCE = "inductance"
    TORQUE = "torque"
    ANGLE = "angle"
    CONCENTRATION = "concentration"
    SPECIFIC_ENTHALPY = "specific_enthalpy"
    DIMENSIONLESS = "dimensionless"


CATEGORY_DIMENSIONS = {
    UnitCategory.LENGTH: LENGTH,
    UnitCategory.MASS: MASS,
    UnitCategory.TIME: TIME,
    UnitCategory.TEMPERATURE: TEMPERATURE,
    UnitCategory.CURRENT: CURRENT,
    UnitCategory.AMOUNT: AMOUNT,
    UnitCategory.AREA: AREA,
    UnitCategory.VOLUME: VOLUME,
    UnitCategory.VELOCITY: VELOCITY,
    UnitCategory.ACCELERATION: ACCELERATION,
    UnitCategory.FREQUENCY: FREQUENCY,
    UnitCategory.FORCE: FORCE,
    UnitCategory.PRESSURE: PRESSURE,
    UnitCategory.ENERGY: ENERGY,
    UnitCategory.POWER: POWER,
    UnitCategory.DENSITY: DENSITY,
    UnitCategory.DYNAMIC_VISCOSITY: DYNAMIC_VISCOSITY,
    UnitCategory.KINEMATIC_VISCOSITY: KINEMATIC_VISCOSITY,
    UnitCategory.VOLUMETRIC_FLOW: VOLUMETRIC_FLOW,
    UnitCategory.MASS_FLOW: MASS_FLOW,
    UnitCategory.VOLTAGE: VOLTAGE,
    UnitCategory.RESISTANCE: RESISTANCE,
    UnitCategory.CAPACITANCE: CAPACITANCE,
    UnitCategory.INDUCTANCE: INDUCTANCE,
    UnitCategory.TORQUE: TORQUE,
    UnitCategory.ANGLE: DIMENSIONLESS,
    UnitCategory.CONCENTRATION: CONCENTRATION,
    UnitCategory.SPECIFIC_ENTHALPY: SPECIFIC_ENTHALPY,
    UnitCategory.DIMENSIONLESS: DIMENSIONLESS,
}


def get_units_for_category(category: UnitCategory) -> List[str]:
    """Get all unit symbols in a category (for UI dropdowns)"""
    target_dims = CATEGORY_DIMENSIONS[category]
    return [symbol for symbol, unit in UNITS.items()
            if unit.dimensions == target_dims and symbol == unit.symbol]  # Primary symbols only


def get_category(unit_str: str) -> Optional[UnitCategory]:
    """Determine the category of a unit string"""
    try:
        unit_def = Quantity._get_unit(unit_str)
        dims = unit_def.dimensions

        for cat, cat_dims in CATEGORY_DIMENSIONS.items():
            if dims == cat_dims:
                return cat
        return None
    except ValueError:
        return None


def parse_value_with_unit(s: str) -> Tuple[float, str, UnitCategory]:
    """
    Parse user input like "4 in" or "100 gpm".

    Returns:
        Tuple of (value, unit_string, category)
    """
    q = Q(s)
    cat = get_category(q.unit_str)
    return q.value, q.unit_str, cat


# =============================================================================
# UNIT AUTODETECTION
# =============================================================================

def autodetect_unit(values: List[float], column_name: str = "") -> Optional[str]:
    """
    Attempt to autodetect units from values and column name.

    This is a heuristic - returns None if uncertain.

    Args:
        values: List of numeric values
        column_name: Optional column/signal name for hints

    Returns:
        Suggested unit string or None
    """
    if not values:
        return None

    # Clean column name
    name_lower = column_name.lower().strip()

    # Check for explicit unit in column name
    # e.g., "flow_gpm", "temp_C", "pressure_psi"
    for unit_str in UNITS.keys():
        if name_lower.endswith(f"_{unit_str.lower()}"):
            return unit_str
        if name_lower.endswith(f"({unit_str.lower()})"):
            return unit_str

    # Keyword-based hints
    if any(kw in name_lower for kw in ['temp', 'temperature']):
        # Check value range for temperature
        min_v, max_v = min(values), max(values)
        if 200 < min_v < 400:  # Kelvin range
            return 'K'
        elif -50 < min_v < 100:  # Celsius range
            return 'degC'
        elif 0 < min_v < 212:  # Fahrenheit range (overlaps)
            return 'degF'

    if any(kw in name_lower for kw in ['pressure', 'pres']):
        min_v = min(values)
        if min_v > 1000:  # Likely Pa
            return 'Pa'
        elif 0 < min_v < 500:  # Likely psi or bar
            return 'psi'

    if any(kw in name_lower for kw in ['velocity', 'speed', 'vel']):
        return 'm/s'

    if any(kw in name_lower for kw in ['flow', 'rate']):
        return 'm3/s'

    if any(kw in name_lower for kw in ['force']):
        return 'N'

    if any(kw in name_lower for kw in ['power', 'watt']):
        return 'W'

    if any(kw in name_lower for kw in ['energy', 'work']):
        return 'J'

    if any(kw in name_lower for kw in ['voltage', 'volt']):
        return 'V'

    if any(kw in name_lower for kw in ['current', 'amp']):
        return 'A'

    return None


# =============================================================================
# ENGINEERING CALCULATIONS
# =============================================================================

def reynolds_number(velocity: Quantity, diameter: Quantity,
                    density: Quantity, viscosity: Quantity) -> float:
    """
    Calculate Reynolds number: Re = rho*V*D/mu

    Args:
        velocity: Flow velocity
        diameter: Characteristic length (pipe diameter)
        density: Fluid density
        viscosity: Dynamic viscosity

    Returns:
        Reynolds number (dimensionless)
    """
    # All converted to SI automatically
    v = velocity.si  # m/s
    d = diameter.si  # m
    rho = density.si  # kg/m3
    mu = viscosity.si  # Pa*s

    re = (rho * v * d) / mu
    return re


def velocity_from_flow(flow_rate: Quantity, diameter: Quantity) -> Quantity:
    """
    Calculate velocity from volumetric flow rate and pipe diameter.

    v = Q / A = Q / (pi * (D/2)^2)
    """
    q = flow_rate.si  # m3/s
    d = diameter.si   # m

    area = math.pi * (d / 2) ** 2  # m2
    v = q / area  # m/s

    return Quantity(v, "m/s")


def pressure_drop_darcy(friction_factor: float, length: Quantity, diameter: Quantity,
                        density: Quantity, velocity: Quantity) -> Quantity:
    """
    Darcy-Weisbach pressure drop: dP = f * (L/D) * (rho*V^2/2)
    """
    L = length.si
    D = diameter.si
    rho = density.si
    V = velocity.si

    dp = friction_factor * (L / D) * (rho * V**2 / 2)
    return Quantity(dp, "Pa")


def kinetic_energy(mass: Quantity, velocity: Quantity) -> Quantity:
    """T = 1/2 * m * v^2"""
    m = mass.si
    v = velocity.si
    return Quantity(0.5 * m * v**2, "J")


def potential_energy_spring(spring_constant: Quantity, displacement: Quantity) -> Quantity:
    """V = 1/2 * k * x^2"""
    k = spring_constant.si
    x = displacement.si
    return Quantity(0.5 * k * x**2, "J")


def ohms_law_current(voltage: Quantity, resistance: Quantity) -> Quantity:
    """I = V/R"""
    v = voltage.si
    r = resistance.si
    return Quantity(v / r, "A")


def ohms_law_power(voltage: Quantity, current: Quantity) -> Quantity:
    """P = V*I"""
    v = voltage.si
    i = current.si
    return Quantity(v * i, "W")


# =============================================================================
# API GRAVITY SPECIAL HANDLING
# =============================================================================

def api_to_density(api_gravity: float) -> Quantity:
    """
    Convert API gravity to density.

    rho = 141.5 / (API + 131.5) * 999.016 kg/m3

    (Reference: water at 60F = 999.016 kg/m3)
    """
    sg = 141.5 / (api_gravity + 131.5)
    density = sg * 999.016  # kg/m3
    return Quantity(density, "kg/m3")


def density_to_api(density: Quantity) -> float:
    """
    Convert density to API gravity.

    API = 141.5 / SG - 131.5
    """
    rho = density.to("kg/m3")
    sg = rho / 999.016
    api = 141.5 / sg - 131.5
    return api


# =============================================================================
# CONFIG PARSING HELPERS
# =============================================================================

def parse_equipment_property(value_str: str) -> Quantity:
    """
    Parse equipment property from config.

    Examples:
        "4 in" -> Quantity(4, "in")
        "100" -> raises ValueError (unit required)
        "0.1016 m" -> Quantity(0.1016, "m")
    """
    return Q(value_str)


def parse_signal_unit(unit_str: str) -> UnitDef:
    """
    Validate and return unit definition for a signal.

    Raises ValueError if unit is unknown.
    """
    return Quantity._get_unit(unit_str)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core classes
    'Quantity', 'Q', 'Dimensions', 'UnitDef', 'UnitCategory',

    # Dimension constants
    'DIMENSIONLESS', 'LENGTH', 'MASS', 'TIME', 'CURRENT', 'TEMPERATURE', 'AMOUNT',
    'AREA', 'VOLUME', 'VELOCITY', 'ACCELERATION', 'FREQUENCY', 'FORCE', 'PRESSURE',
    'ENERGY', 'POWER', 'DENSITY', 'DYNAMIC_VISCOSITY', 'KINEMATIC_VISCOSITY',
    'VOLUMETRIC_FLOW', 'MASS_FLOW', 'VOLTAGE', 'RESISTANCE', 'MOMENTUM',
    'ANGULAR_MOMENTUM', 'MOMENT_OF_INERTIA', 'TORQUE', 'ENTROPY', 'SPECIFIC_ENTHALPY',
    'ELECTRIC_FIELD', 'MAGNETIC_FIELD', 'MAGNETIC_FLUX', 'CHARGE',

    # Physical constants
    'PHYSICAL_CONSTANTS', 'get_constant',

    # Helper functions
    'get_units_for_category', 'get_category', 'parse_value_with_unit',
    'autodetect_unit',

    # Engineering calculations
    'reynolds_number', 'velocity_from_flow', 'pressure_drop_darcy',
    'kinetic_energy', 'potential_energy_spring', 'ohms_law_current', 'ohms_law_power',
    'api_to_density', 'density_to_api',

    # Config helpers
    'parse_equipment_property', 'parse_signal_unit',

    # Registry
    'UNITS', 'register_unit',
]


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PRISM UnitSpec - Self Test")
    print("=" * 60)

    # Basic conversions
    print("\n--- Basic Conversions ---")
    print(f"4 in = {Q('4 in').to('m'):.6f} m")
    print(f"4 in = {Q('4 in').to('mm'):.2f} mm")
    print(f"100 gpm = {Q('100 gpm').to('m3/s'):.6e} m3/s")
    print(f"100 gpm = {Q('100 gpm').to('L/min'):.2f} L/min")
    print(f"120 psi = {Q('120 psi').to('kPa'):.2f} kPa")
    print(f"5 cP = {Q('5 cP').to('Pa*s'):.6f} Pa*s")
    print(f"850 kg/m3 = {Q('850 kg/m3').to('lb/ft3'):.2f} lb/ft3")

    # Temperature (with offset)
    print("\n--- Temperature ---")
    print(f"100 degC = {Q('100 degC').to('K'):.2f} K")
    print(f"212 degF = {Q('212 degF').to('degC'):.2f} degC")
    print(f"0 K = {Q('0 K').to('degC'):.2f} degC")

    # Dimensional analysis
    print("\n--- Dimensional Analysis ---")
    flow = Q("100 gpm")
    diameter = Q("4 in")
    velocity = velocity_from_flow(flow, diameter)
    print(f"Flow: {flow}")
    print(f"Pipe: {diameter}")
    print(f"Velocity: {velocity.to('ft/s'):.2f} ft/s")

    # Reynolds number
    print("\n--- Reynolds Number ---")
    density = Q("850 kg/m3")  # Light crude
    viscosity = Q("5 cP")
    Re = reynolds_number(velocity, diameter, density, viscosity)
    print(f"Density: {density}")
    print(f"Viscosity: {viscosity}")
    print(f"Reynolds: {Re:.0f}")
    if Re < 2300:
        print("-> LAMINAR flow")
    elif Re > 4000:
        print("-> TURBULENT flow")
    else:
        print("-> TRANSITIONAL flow")

    # Arithmetic
    print("\n--- Arithmetic ---")
    m = Q("2 kg")
    v = Q("5 m/s")
    ke = 0.5 * m * v * v
    print(f"Mass: {m}")
    print(f"Velocity: {v}")
    print(f"KE = 1/2*m*v^2 = {ke}")
    print(f"KE in J: {ke.to('J'):.2f} J")

    # Physical constants
    print("\n--- Physical Constants ---")
    for name in ['c', 'g', 'R', 'rho_water']:
        val, unit, desc = PHYSICAL_CONSTANTS[name]
        print(f"  {name}: {val} {unit} ({desc})")

    # API gravity
    print("\n--- API Gravity ---")
    api = 35  # Light crude
    rho = api_to_density(api)
    print(f"API {api} = {rho.to('kg/m3'):.1f} kg/m3 = {rho.to('lb/gal'):.2f} lb/gal")

    # Electrical
    print("\n--- Electrical ---")
    V = Q("120 V")
    R = Q("60 ohm")
    I = ohms_law_current(V, R)
    P = ohms_law_power(V, I)
    print(f"Voltage: {V}")
    print(f"Resistance: {R}")
    print(f"Current: {I.to('A'):.2f} A")
    print(f"Power: {P.to('W'):.2f} W")

    # Category lookup
    print("\n--- Unit Categories ---")
    for unit in ["gpm", "psi", "cP", "in", "degF", "V"]:
        cat = get_category(unit)
        print(f"  {unit:6s} -> {cat.value if cat else 'unknown'}")

    # Available units for dropdown
    print("\n--- Volumetric Flow Units (for UI) ---")
    flow_units = get_units_for_category(UnitCategory.VOLUMETRIC_FLOW)
    print(f"  {flow_units[:10]}...")

    # Unit autodetection
    print("\n--- Unit Autodetection ---")
    test_cases = [
        ([25, 30, 28], "temperature_C"),
        ([300, 320, 310], "temp_K"),
        ([100, 150, 120], "pressure_psi"),
        ([5.0, 6.2, 4.8], "velocity"),
    ]
    for values, name in test_cases:
        detected = autodetect_unit(values, name)
        print(f"  {name}: {detected or 'unknown'}")

    print("\n" + "=" * 60)
    print(f"Total units registered: {len(UNITS)}")
    print("=" * 60)
