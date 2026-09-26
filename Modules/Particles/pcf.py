"""
Reader for Source particle files (.pcf): a Valve DMX binary file holding a list
of particle system definitions, each with a table of parameters plus lists of
emitters, initializers, operators, force generators, constraints, renderers
and child systems (every entry is an element with a "functionName").

Only what particles need is read: the DMX binary encodings 2 and 5 (what Source
tools write PCF as), elements, their attributes and the references between
them. read_pcf() returns {system name: ParticleSystemDef}.
"""

import struct
from dataclasses import dataclass, field

# Attribute type ids (DMX binary v2; the array form of each is id + 14).
_ELEMENT, _INT, _FLOAT, _BOOL, _STRING, _BINARY, _TIME, _COLOR = range(1, 9)
_VEC2, _VEC3, _VEC4, _ANGLE, _QUAT, _MATRIX = range(9, 15)
_ARRAY = 14

# The lists a particle system definition holds, in its attribute names.
_FUNCTION_LISTS = ("emitters", "initializers", "operators", "forces", "constraints", "renderers")


@dataclass
class Function:
    """One emitter/initializer/operator/renderer/force/constraint: a Source
    function name ("emit_continuously", "Lifespan Random", ...) and its
    parameters."""
    name: str
    params: dict


@dataclass
class ParticleSystemDef:
    name: str
    params: dict
    emitters: list = field(default_factory=list)
    initializers: list = field(default_factory=list)
    operators: list = field(default_factory=list)
    forces: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    renderers: list = field(default_factory=list)
    children: list = field(default_factory=list)   # (ParticleSystemDef, delay seconds)

    @property
    def material(self):
        return self.params.get("material", "")


class _Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def take(self, fmt):
        size = struct.calcsize(fmt)
        value = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += size
        return value[0] if len(value) == 1 else value

    def cstring(self):
        end = self.data.index(b"\0", self.pos)
        text = self.data[self.pos:end].decode("utf-8", "replace")
        self.pos = end + 1
        return text


def _value(r, kind, strings, version):
    if kind == _ELEMENT:
        return ("element", r.take("<i"))
    if kind == _INT:
        return r.take("<i")
    if kind == _FLOAT:
        return r.take("<f")
    if kind == _BOOL:
        return bool(r.take("<B"))
    if kind == _STRING:
        return r.cstring()
    if kind == _BINARY:
        n = r.take("<i")
        raw = r.data[r.pos:r.pos + n]
        r.pos += n
        return bytes(raw)
    if kind == _TIME:
        return r.take("<i") / 10000.0
    if kind == _COLOR:
        return tuple(r.take("<4B"))
    if kind == _VEC2:
        return r.take("<2f")
    if kind == _VEC3 or kind == _ANGLE:
        return r.take("<3f")
    if kind == _VEC4 or kind == _QUAT:
        return r.take("<4f")
    if kind == _MATRIX:
        return r.take("<16f")
    raise ValueError(f"unsupported DMX attribute type {kind}")


def _parse(data):
    r = _Reader(data)
    header = r.cstring()   # "<!-- dmx encoding binary 2 format pcf 1 -->\n"
    if "encoding binary" not in header:
        raise ValueError("not a binary DMX file (keyvalues2 text PCFs aren't supported)")
    version = int(header.split("encoding binary")[1].split()[0])
    if version not in (2, 5):
        raise ValueError(f"unsupported DMX binary version {version}")
    # Both store a shared string table: version 2 counts it in 16 bits,
    # version 5 in 32, and version 5 refers to strings by 32-bit index.
    count = r.take("<i") if version >= 5 else r.take("<H")
    strings = [r.cstring() for _ in range(count)]
    index_fmt = "<i" if version >= 5 else "<H"

    def name_at():
        return strings[r.take(index_fmt)]

    elements = []
    for _ in range(r.take("<i")):
        etype = name_at()
        ename = name_at() if version >= 4 else r.cstring()
        r.pos += 16   # GUID
        elements.append({"type": etype, "name": ename, "attrs": {}})

    for element in elements:
        for _ in range(r.take("<i")):
            aname = name_at()
            kind = r.take("<B")
            if kind > _ARRAY:
                n = r.take("<i")
                element["attrs"][aname] = [_value(r, kind - _ARRAY, strings, version) for _ in range(n)]
            else:
                element["attrs"][aname] = _value(r, kind, strings, version)
    return elements


def read_pcf(path):
    """Loads every particle system in the file: {name: ParticleSystemDef}."""
    with open(path, "rb") as f:
        elements = _parse(f.read())

    def resolve(ref):
        return elements[ref[1]] if isinstance(ref, tuple) and ref[0] == "element" and ref[1] >= 0 else None

    def function(element):
        params = dict(element["attrs"])
        return Function(params.pop("functionName", element["name"]), params)

    systems = {}

    def build(element):
        if id(element) in systems:
            return systems[id(element)]
        attrs = element["attrs"]
        params = {k: v for k, v in attrs.items()
                  if k not in _FUNCTION_LISTS and k != "children"}
        system = ParticleSystemDef(element["name"], params)
        systems[id(element)] = system
        for key in _FUNCTION_LISTS:
            getattr(system, key).extend(
                function(e) for e in map(resolve, attrs.get(key, [])) if e is not None)
        for child in attrs.get("children", []):
            child_element = resolve(child)
            if child_element is None:
                continue
            # A child entry is a DmeParticleChild wrapping the real system.
            target = resolve(child_element["attrs"].get("child")) or child_element
            delay = float(child_element["attrs"].get("delay", 0.0) or 0.0)
            system.children.append((build(target), delay))
        return system

    result = {}
    for element in elements:
        if element["type"] == "DmeParticleSystemDefinition":
            result[element["name"]] = build(element)
    return result
