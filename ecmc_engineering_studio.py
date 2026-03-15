#!/usr/bin/env python3
"""
ecmc Engineering Studio.

The tool opens a startup file, shows it in an editor, and validates local
project references such as YAML, PLC, and nested script files. Validation is
intentionally conservative: it focuses on file existence and known hardware
descriptors without trying to fully execute IOC shell logic.
"""

from typing import Dict, List, Optional, Set, Tuple

import argparse
import json
import re
import sys
import time
from functools import lru_cache
from datetime import datetime
from compat_dataclasses import dataclass, field
from pathlib import Path


SCRIPT_EXTENSIONS = {".cmd", ".script", ".iocsh"}
PATH_SUFFIXES = {
    ".ax",
    ".cfg",
    ".cmd",
    ".iocsh",
    ".pax",
    ".plc",
    ".req",
    ".script",
    ".subs",
    ".substitutions",
    ".template",
    ".txt",
    ".yaml",
    ".yml",
}
COMMENT_PREFIXES = ("#", "#-")
SCRIPT_EXEC_MARKERS = ("${SCRIPTEXEC}", "$(SCRIPTEXEC)", "iocshLoad", "runScript")
KNOWN_IOCSH_COMMANDS = {
    "dbLoadDatabase",
    "dbLoadRecords",
    "dbLoadTemplate",
    "ecmcConfig",
    "ecmcConfigOrDie",
    "ecmcEpicsEnvSetCalc",
    "ecmcEpicsEnvSetCalcTernary",
    "ecmcExit",
    "ecmcFileExist",
    "epicsEnvSet",
    "epicsEnvShow",
    "epicsEnvUnset",
    "iocshLoad",
    "on",
    "require",
    "runScript",
    "system",
}

PAIR_RE = re.compile(
    r"(?P<key>[A-Z_]+)\s*=\s*(?P<value>\"[^\"]*\"|'[^']*'|[^,\s)]+)",
    re.IGNORECASE,
)
MACRO_REF_RE = re.compile(r"\$\{([A-Za-z0-9_]+)(=([^}]*))?\}|\$\(([A-Za-z0-9_]+)(=([^\)]*))?\)")
EC_LINK_RE = re.compile(r"\bec(?P<master>\d+)\.s(?P<slave>\d+)\.(?P<entry>[A-Za-z_][A-Za-z0-9_]*)")
PLC_SYMBOL_RE = re.compile(r"\b(?P<scope>static|global)\.(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b")
PLC_VAR_DECL_RE = re.compile(
    r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*:\s*(?P<scope>static|global)\.(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*;?\s*$"
)
GENERIC_EC_ENTRY_NAMES = {"ONE", "ZERO", "slavestatus"}


@dataclass(frozen=True)
class FileReference:
    source: Path
    target: Path
    kind: str
    line: int
    exists: bool


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    source: Path
    line: int
    message: str
    target: Optional[Path] = None


@dataclass
class ValidationResult:
    issues: List[ValidationIssue]
    references: List[FileReference]
    visited_files: List[Path]


@dataclass
class StartupObject:
    kind: str
    source: Path
    line: int
    title: str
    summary: str
    slave_id: Optional[int] = None
    parent_slave_id: Optional[int] = None
    parent_axis_line: Optional[int] = None
    parent_plc_id: Optional[int] = None
    linked_file: Optional[Path] = None
    details: List[Tuple[str, str]] = field(default_factory=list)
    command_details: List[Tuple[str, str]] = field(default_factory=list)
    linked_file_details: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class StartupFileNode:
    path: Path
    parent_path: Optional[Path]
    parent_line: int
    objects: List[StartupObject] = field(default_factory=list)


@dataclass
class StartupTreeModel:
    files: List[StartupFileNode]


@dataclass
class RepositoryInventory:
    ecmccfg_root: Optional[Path]
    ecmccomp_root: Optional[Path]
    ecmc_root: Optional[Path]
    module_scripts: Dict[str, List[Path]]
    module_macro_specs: Dict[str, "MacroSpec"]
    module_macro_usage: Dict[str, "FileMacroUsage"]
    hardware_descs: Set[str]
    hardware_component_types: Dict[str, str]
    hardware_configs: Dict[str, List[Path]]
    hardware_entries: Dict[str, Set[str]]
    component_definitions: Dict[str, "ComponentDefinition"]
    component_support: Dict[str, Dict[str, "ComponentSupport"]]
    ecmc_command_definitions: Dict[str, List["EcmcCommandDefinition"]]
    known_commands: Set[str]
    ecb_schema: Optional[Dict[str, object]]


@dataclass
class MacroSpec:
    allowed: Set[str]
    required: Set[str]


@dataclass
class FileMacroUsage:
    used: Set[str]
    required: Set[str]


@dataclass(frozen=True)
class ComponentDefinition:
    name: str
    path: Path
    comp_type: str


@dataclass(frozen=True)
class ComponentSupport:
    comp_hw_type: str
    comp_type: str
    path: Path
    supported_macros: Set[str]
    channel_count: Optional[int]


@dataclass(frozen=True)
class EcmcCommandDefinition:
    command_name: str
    syntax: str
    function_name: str
    brief: str
    header: Optional[Path]


@dataclass
class ParsedMappingLine:
    path: str
    value: Optional[str]
    line: int


@dataclass(frozen=True)
class ExpandedTextLine:
    source: Path
    line: int
    text: str


def _find_ecmccfg_root(anchor: Optional[Path] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if anchor is not None:
        resolved_anchor = anchor.resolve()
        candidates.extend([resolved_anchor, *resolved_anchor.parents])

    script_dir = Path(__file__).resolve().parent
    candidates.extend([script_dir, *script_dir.parents])

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    seen: Set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        direct_root = candidate
        if (direct_root / "hardware").is_dir() and (direct_root / "scripts").is_dir():
            return direct_root
        submodule_root = candidate / "ecmccfg"
        if (submodule_root / "hardware").is_dir() and (submodule_root / "scripts").is_dir():
            return submodule_root
    return None


def _find_ecmccomp_root(ecmccfg_root: Optional[Path], anchor: Optional[Path] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if anchor is not None:
        resolved_anchor = anchor.resolve()
        candidates.extend([resolved_anchor, *resolved_anchor.parents])
    if ecmccfg_root is not None:
        resolved_cfg_root = ecmccfg_root.resolve()
        candidates.extend([resolved_cfg_root, *resolved_cfg_root.parents])

    script_dir = Path(__file__).resolve().parent
    candidates.extend([script_dir, *script_dir.parents])

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    seen: Set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        direct_root = candidate
        if (
            (direct_root / "scripts" / "applyComponent.cmd").exists()
            and ((direct_root / "drive_slaves").is_dir() or (direct_root / "motors").is_dir())
        ):
            return direct_root
        submodule_root = candidate / "ecmccomp"
        if (
            (submodule_root / "scripts" / "applyComponent.cmd").exists()
            and ((submodule_root / "drive_slaves").is_dir() or (submodule_root / "motors").is_dir())
        ):
            return submodule_root
    return None


def _find_ecmc_root(ecmccfg_root: Optional[Path], anchor: Optional[Path] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if anchor is not None:
        resolved_anchor = anchor.resolve()
        candidates.extend([resolved_anchor, *resolved_anchor.parents])
    if ecmccfg_root is not None:
        resolved_cfg_root = ecmccfg_root.resolve()
        candidates.extend([resolved_cfg_root, *resolved_cfg_root.parents])

    script_dir = Path(__file__).resolve().parent
    candidates.extend([script_dir, *script_dir.parents])

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    seen: Set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        direct_root = candidate
        if (direct_root / "devEcmcSup" / "com" / "ecmcCmdParser.c").exists():
            return direct_root
        submodule_root = candidate / "ecmc"
        if (submodule_root / "devEcmcSup" / "com" / "ecmcCmdParser.c").exists():
            return submodule_root
    return None


def _index_by_name(paths: List[Path]) -> Dict[str, List[Path]]:
    indexed: Dict[str, List[Path]] = {}
    for path in sorted(paths):
        indexed.setdefault(path.name, []).append(path)
    return indexed


def _split_top_level(text: str, sep: str = ",") -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    single = False
    double = False
    brace_depth = 0

    for char in text:
        if char == "'" and not double:
            single = not single
        elif char == '"' and not single:
            double = not double
        elif char in "({[" and not single and not double:
            brace_depth += 1
        elif char in ")}]" and not single and not double and brace_depth > 0:
            brace_depth -= 1

        if char == sep and not single and not double and brace_depth == 0:
            piece = "".join(current).strip()
            if piece:
                parts.append(piece)
            current = []
            continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_argument_header(text: str) -> MacroSpec:
    match = re.search(r"^#-\s*Arguments:\s*(.+)$", text, re.MULTILINE)
    if not match:
        return MacroSpec(set(), set())

    allowed: Set[str] = set()
    required: Set[str] = set()
    remainder = match.group(1).strip()
    optional_ranges = [(m.start(), m.end()) for m in re.finditer(r"\[[^\]]*\]", remainder)]

    def is_optional(index: int) -> bool:
        return any(start <= index < end for start, end in optional_ranges)

    for part in _split_top_level(remainder):
        token = part.strip()
        if not token:
            continue
        optional = is_optional(remainder.find(token))
        token = token.strip("[] ").strip()
        if "=" in token:
            token = token.split("=", 1)[0].strip()
        if token.startswith("["):
            token = token[1:].strip()
        if token.endswith("]"):
            token = token[:-1].strip()
        if token.lower() in {"n/a", "none"}:
            continue
        if not token:
            continue
        allowed.add(token)
        if not optional:
            required.add(token)
    return MacroSpec(allowed, required)


def _parse_param_names(text: str) -> Set[str]:
    params: Set[str] = set()
    for line in text.splitlines():
        match = re.search(r"\\param\s+([A-Z0-9_]+)", line)
        if match:
            params.add(match.group(1).strip())
    return params


def _parse_optional_macro_doc_names(text: str) -> Set[str]:
    params: Set[str] = set()
    in_block = False
    for line in text.splitlines():
        if re.search(r"Macros\s*\(optional\)", line, re.IGNORECASE):
            in_block = True
            continue
        if not in_block:
            continue
        stripped = line.strip()
        if not stripped.startswith("#-d"):
            break
        match = re.search(r"#-d\s+([A-Z0-9_]+)\s*:", stripped)
        if match:
            params.add(match.group(1).strip())
            continue
        if stripped == "#-d":
            continue
        if params:
            break
    return params


def _parse_argument_section_spec(text: str) -> MacroSpec:
    allowed: Set[str] = set()
    required: Set[str] = set()
    in_block = False
    current_section = ""

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not in_block:
            if re.match(r"#-\s*Arguments\b", stripped, re.IGNORECASE):
                in_block = True
            continue

        if not stripped.startswith("#-"):
            break
        content = stripped[2:].strip()
        if not content:
            continue

        section_match = re.match(r"\[(mandatory|optional|set by module)\]", content, re.IGNORECASE)
        if section_match:
            current_section = section_match.group(1).lower()
            continue

        if current_section not in {"mandatory", "optional"}:
            continue

        name_match = re.match(r"([A-Z][A-Z0-9_]+)\b", content)
        if not name_match:
            continue
        name = name_match.group(1).strip()
        remainder = content[name_match.end() :].strip()
        if remainder and not remainder.startswith("="):
            continue
        allowed.add(name)
        if current_section == "mandatory":
            required.add(name)

    return MacroSpec(allowed, required)


def _build_macro_spec(path: Path) -> MacroSpec:
    try:
        text = _read_text(path)
    except Exception:
        return MacroSpec(set(), set())

    header_spec = _parse_argument_header(text)
    section_spec = _parse_argument_section_spec(text)
    param_names = _parse_param_names(text)
    optional_macro_names = _parse_optional_macro_doc_names(text)
    allowed = set(header_spec.allowed) | set(section_spec.allowed) | param_names | optional_macro_names
    required = set(header_spec.required) | set(section_spec.required)
    return MacroSpec(allowed, required)


def _build_macro_usage(path: Path) -> FileMacroUsage:
    try:
        return _scan_file_macro_usage(_read_text(path))
    except Exception:
        return FileMacroUsage(set(), set())


def _build_repository_inventory(ecmccfg_root: Optional[Path]) -> RepositoryInventory:
    if ecmccfg_root is None:
        return RepositoryInventory(
            ecmccfg_root=None,
            ecmccomp_root=None,
            ecmc_root=None,
            module_scripts={},
            module_macro_specs={},
            module_macro_usage={},
            hardware_descs=set(),
            hardware_component_types={},
            hardware_configs={},
            hardware_entries={},
            component_definitions={},
            component_support={},
            ecmc_command_definitions={},
            known_commands=set(KNOWN_IOCSH_COMMANDS),
            ecb_schema=None,
        )

    module_script_paths = sorted((ecmccfg_root / "scripts").rglob("*.cmd"))
    startup_cmd = ecmccfg_root / "startup.cmd"
    if startup_cmd.exists():
        module_script_paths.append(startup_cmd)
    hardware_paths = sorted((ecmccfg_root / "hardware").rglob("ecmc*.cmd"))

    hardware_descs: Set[str] = set()
    for path in hardware_paths:
        stem = path.stem
        if stem.lower().startswith("ecmc") and len(stem) > 4:
            hardware_descs.add(stem[4:])

    schema_path = ecmccfg_root / "scripts" / "jinja2" / "ecbSchema.json"
    ecb_schema = None
    if schema_path.exists():
        try:
            ecb_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except Exception:
            ecb_schema = None

    hardware_index = _index_by_name(hardware_paths)
    hardware_component_types = _build_hardware_component_type_inventory(hardware_index)
    ecmccomp_root = _find_ecmccomp_root(ecmccfg_root)
    ecmc_root = _find_ecmc_root(ecmccfg_root)
    component_definitions, component_support = _build_component_library_inventory(ecmccomp_root)
    ecmc_command_definitions = _build_ecmc_command_inventory(ecmc_root)

    return RepositoryInventory(
        ecmccfg_root=ecmccfg_root.resolve(),
        ecmccomp_root=ecmccomp_root.resolve() if ecmccomp_root is not None else None,
        ecmc_root=ecmc_root.resolve() if ecmc_root is not None else None,
        module_scripts=_index_by_name(module_script_paths),
        module_macro_specs={path.name: _build_macro_spec(path) for path in module_script_paths},
        module_macro_usage={path.name: _build_macro_usage(path) for path in module_script_paths},
        hardware_descs=hardware_descs,
        hardware_component_types=hardware_component_types,
        hardware_configs=hardware_index,
        hardware_entries=_build_hardware_entry_inventory(hardware_index),
        component_definitions=component_definitions,
        component_support=component_support,
        ecmc_command_definitions=ecmc_command_definitions,
        known_commands=_scan_known_commands(ecmccfg_root),
        ecb_schema=ecb_schema,
    )


def _extract_cfg_call_args(text: str, call_name: str) -> List[str]:
    marker = f"Cfg.{call_name}("
    start = text.find(marker)
    if start < 0:
        return []
    remainder = text[start + len(marker) :]
    end = remainder.rfind(")")
    if end < 0:
        return []
    return _split_top_level(remainder[:end])


def _looks_like_entry_name(token: str) -> bool:
    cleaned = _strip_wrapper_pairs(_normalize_value(token))
    if not cleaned:
        return False
    if re.fullmatch(r"[-+]?(?:0x[0-9a-fA-F]+|\d+(?:\.\d+)?)", cleaned):
        return False
    return bool(re.search(r"[A-Za-z_]", cleaned) or "${" in cleaned or "$(" in cleaned)


def _extract_named_cfg_argument(args: List[str]) -> str:
    for token in reversed(args):
        cleaned = _strip_wrapper_pairs(_normalize_value(token))
        if _looks_like_entry_name(cleaned):
            return cleaned
    return ""


def _extract_hardware_entry_names(
    hardware_path: Path,
    hardware_index: Dict[str, List[Path]],
    active_stack: Tuple[Path, ...] = (),
) -> Set[str]:
    resolved = hardware_path.resolve()
    if resolved in active_stack:
        return set()

    try:
        text = _read_text(resolved)
    except Exception:
        return set()

    entries: Set[str] = set()
    child_stack = (*active_stack, resolved)

    for raw_line in text.splitlines():
        line = _strip_inline_comment(raw_line)
        if not line.strip():
            continue

        for call_name in ("EcAddEntryComplete", "EcAddEntryDT", "EcAddDataDT", "WriteEcEntryIDString"):
            args = _extract_cfg_call_args(line, call_name)
            if not args:
                continue
            name = _extract_named_cfg_argument(args)
            if name:
                entries.add(name)

        if "addEcDataItem.cmd" in line:
            payload = _extract_script_call_macro_text(line)
            payload_pairs, _malformed = _parse_macro_payload(payload)
            name = next((value for key, value in payload_pairs if key == "NAME" and value), "")
            if name:
                entries.add(name)

        nested_target = _extract_script_target(line)
        nested_name = _extract_module_script_name(nested_target)
        if nested_name in hardware_index:
            for nested_path in hardware_index[nested_name]:
                entries.update(_extract_hardware_entry_names(nested_path, hardware_index, child_stack))

    return entries


def _build_hardware_entry_inventory(hardware_index: Dict[str, List[Path]]) -> Dict[str, Set[str]]:
    inventory: Dict[str, Set[str]] = {}

    for file_name, paths in hardware_index.items():
        stem = Path(file_name).stem
        if not stem.lower().startswith("ecmc") or len(stem) <= 4:
            continue
        hw_desc = stem[4:]
        hw_entries = inventory.setdefault(hw_desc, set())
        for path in paths:
            hw_entries.update(_extract_hardware_entry_names(path, hardware_index))

    return inventory


def _parse_epics_env_assignment(line: str) -> Optional[Tuple[str, str]]:
    stripped = _strip_inline_comment(line).strip()
    if not stripped.startswith("epicsEnvSet("):
        return None
    inner = stripped[len("epicsEnvSet(") :]
    if ")" not in inner:
        return None
    inner = inner.rsplit(")", 1)[0].strip()
    parts = _split_top_level(inner)
    if len(parts) >= 2:
        name = _normalize_value(parts[0])
        value = _normalize_value(",".join(parts[1:]))
    else:
        match = re.match(r'(".*?"|\'.*?\'|\S+)\s+(.+)$', inner)
        if not match:
            return None
        name = _normalize_value(match.group(1))
        value = _normalize_value(match.group(2))
    if not re.fullmatch(r"[A-Z0-9_]+", name):
        return None
    return name, value


def _extract_included_cmd_names(text: str) -> List[str]:
    names: List[str] = []
    for raw_line in text.splitlines():
        stripped = _strip_inline_comment(raw_line).strip()
        if not stripped.startswith("<"):
            continue
        match = re.search(r"([A-Za-z0-9_.+-]+\.cmd)\b", stripped)
        if match:
            names.append(match.group(1))
    return names


def _read_cmd_env_assignments(
    path: Path,
    cmd_index: Dict[str, List[Path]],
    stack: Optional[Set[Path]] = None,
) -> Dict[str, str]:
    resolved = path.resolve()
    if stack is None:
        stack = set()
    if resolved in stack:
        return {}
    stack = set(stack)
    stack.add(resolved)

    try:
        text = _read_text(path)
    except Exception:
        return {}

    assignments: Dict[str, str] = {}
    for raw_line in text.splitlines():
        stripped = _strip_inline_comment(raw_line).strip()
        if not stripped:
            continue
        if stripped.startswith("<"):
            for include_name in _extract_included_cmd_names(raw_line):
                for nested_path in cmd_index.get(include_name, []):
                    assignments.update(_read_cmd_env_assignments(nested_path, cmd_index, stack))
            continue
        parsed = _parse_epics_env_assignment(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        assignments[key] = value
    return assignments


def _build_hardware_component_type_inventory(hardware_index: Dict[str, List[Path]]) -> Dict[str, str]:
    inventory: Dict[str, str] = {}
    for file_name, paths in hardware_index.items():
        stem = Path(file_name).stem
        if not stem.lower().startswith("ecmc") or len(stem) <= 4:
            continue
        hw_desc = stem[4:]
        for path in paths:
            assignments = _read_cmd_env_assignments(path, hardware_index)
            comp_type = assignments.get("ECMC_EC_COMP_TYPE", "").strip()
            if comp_type:
                inventory[hw_desc] = comp_type
                break
    return inventory


def _build_component_library_inventory(
    ecmccomp_root: Optional[Path],
) -> Tuple[Dict[str, ComponentDefinition], Dict[str, Dict[str, ComponentSupport]]]:
    if ecmccomp_root is None or not ecmccomp_root.exists():
        return {}, {}

    cmd_paths = sorted(ecmccomp_root.rglob("*.cmd"))
    cmd_index = _index_by_name(cmd_paths)
    ignored_parts = {"scripts", "support", "hw_sdo_scripts", "not_ready"}

    component_definitions: Dict[str, ComponentDefinition] = {}
    component_types: Set[str] = set()
    for path in cmd_paths:
        relative_parts = set(path.relative_to(ecmccomp_root).parts)
        if relative_parts & ignored_parts:
            continue
        assignments = _read_cmd_env_assignments(path, cmd_index)
        comp_type = assignments.get("COMP_TYPE", "").strip()
        if not comp_type:
            continue
        component_definitions[path.stem] = ComponentDefinition(name=path.stem, path=path, comp_type=comp_type)
        component_types.add(comp_type)

    component_support: Dict[str, Dict[str, ComponentSupport]] = {}
    type_candidates = sorted(component_types, key=len, reverse=True)
    for path in cmd_paths:
        relative_parts = set(path.relative_to(ecmccomp_root).parts)
        if relative_parts & ignored_parts:
            continue
        stem = path.stem
        matched_type = ""
        for comp_type in type_candidates:
            if stem.endswith("_" + comp_type):
                matched_type = comp_type
                break
        if not matched_type:
            continue
        comp_hw_type = stem[: -(len(matched_type) + 1)].strip("_")
        if not comp_hw_type:
            continue
        assignments = _read_cmd_env_assignments(path, cmd_index)
        if not any(key in assignments for key in {"SLAVE_SCRIPT", "SLAVE_CHANNELS", "SUPP_MACROS", "SLAVE_TYPE"}):
            continue
        supported_macros = {
            item.strip()
            for item in _normalize_value(assignments.get("SUPP_MACROS", "")).split(",")
            if item.strip()
        }
        component_support.setdefault(comp_hw_type, {})[matched_type] = ComponentSupport(
            comp_hw_type=comp_hw_type,
            comp_type=matched_type,
            path=path,
            supported_macros=supported_macros,
            channel_count=_parse_int_value(assignments.get("SLAVE_CHANNELS", "")),
        )

    return component_definitions, component_support


def _clean_doc_block_lines(block: str) -> List[str]:
    lines: List[str] = []
    for raw_line in block.splitlines():
        cleaned = re.sub(r"^\s*/?\*+\s?", "", raw_line.rstrip())
        cleaned = cleaned.strip()
        if cleaned.endswith("*/"):
            cleaned = cleaned[:-2].rstrip()
        if cleaned:
            lines.append(cleaned)
    return lines


def _extract_doc_brief(block: str) -> str:
    lines = _clean_doc_block_lines(block)
    for line in lines:
        if "\\brief" in line:
            return line.split("\\brief", 1)[1].strip().rstrip("\\n")
        if "@brief" in line:
            return line.split("@brief", 1)[1].strip().rstrip("\\n")
    for line in lines:
        if line.startswith(("\\", "@", "*")):
            continue
        return line.rstrip("\\n")
    return ""


def _extract_parser_comment_syntax(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith(("///", "/*", "*", "//")):
        return ""
    single_line = " ".join(part.strip() for part in text.splitlines())
    match = re.search(r"(Cfg\.[A-Za-z0-9_?]+(?:\([^)]*\)|=[^\"*/]*))", single_line)
    if match:
        return match.group(1).strip()
    match = re.search(r"\"([A-Za-z][A-Za-z0-9_.?]+(?:\([^\"\n]*\)|=[^\"\n]*))\"", single_line)
    if match:
        return match.group(1).strip()
    return ""


def _extract_ecmc_inner_command_name(command_text: str) -> str:
    cleaned = _strip_wrapper_pairs(_normalize_value(command_text)).strip()
    if cleaned.startswith("Cfg."):
        cleaned = cleaned[4:]
    match = re.match(r"([A-Za-z_][A-Za-z0-9_?]*)", cleaned)
    return match.group(1) if match else ""


def _extract_parser_handler_name(lines: List[str], start_index: int) -> str:
    for offset in range(1, 12):
        if start_index + offset >= len(lines):
            break
        line = lines[start_index + offset]
        for pattern in (
            r"\breturn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            r"SEND_[A-Z_]+\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        ):
            match = re.search(pattern, line)
            if match:
                return match.group(1)
    return ""


def _build_ecmc_function_doc_index(ecmc_root: Optional[Path]) -> Dict[str, Tuple[str, Path]]:
    if ecmc_root is None or not ecmc_root.exists():
        return {}

    doc_index: Dict[str, Tuple[str, Path]] = {}
    pattern = re.compile(
        r"/\*\*(?P<doc>.*?)\*/\s*(?:[A-Za-z_][A-Za-z0-9_\s\*]*?\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(",
        re.S,
    )
    for header_path in sorted((ecmc_root / "devEcmcSup").rglob("*.h")):
        try:
            text = _read_text(header_path)
        except Exception:
            continue
        for match in pattern.finditer(text):
            name = match.group("name")
            if name in doc_index:
                continue
            brief = _extract_doc_brief(match.group("doc"))
            doc_index[name] = (brief, header_path)
    return doc_index


def _build_ecmc_command_inventory(ecmc_root: Optional[Path]) -> Dict[str, List[EcmcCommandDefinition]]:
    if ecmc_root is None:
        return {}

    parser_path = ecmc_root / "devEcmcSup" / "com" / "ecmcCmdParser.c"
    if not parser_path.exists():
        return {}

    doc_index = _build_ecmc_function_doc_index(ecmc_root)
    try:
        lines = _read_text(parser_path).splitlines()
    except Exception:
        return {}

    command_inventory: Dict[str, List[EcmcCommandDefinition]] = {}
    seen: Set[Tuple[str, str, str]] = set()
    comment_block = ""
    comment_start_index = -1
    for index, line in enumerate(lines):
        stripped = line.strip()
        syntax = ""
        syntax_index = index
        if comment_block:
            comment_block += "\n" + line
            if "*/" in line:
                syntax = _extract_parser_comment_syntax(comment_block)
                syntax_index = comment_start_index
                comment_block = ""
                comment_start_index = -1
        elif stripped.startswith("/*") and "Cfg." in stripped:
            if "*/" in stripped:
                syntax = _extract_parser_comment_syntax(line)
            else:
                comment_block = line
                comment_start_index = index
                continue
        else:
            syntax = _extract_parser_comment_syntax(line)
        if not syntax:
            continue
        command_name = _extract_ecmc_inner_command_name(syntax)
        if not command_name:
            continue
        function_name = _extract_parser_handler_name(lines, syntax_index)
        brief = ""
        header_path: Optional[Path] = None
        if function_name in doc_index:
            brief, header_path = doc_index[function_name]
        key = (command_name, syntax, function_name)
        if key in seen:
            continue
        seen.add(key)
        command_inventory.setdefault(command_name, []).append(
            EcmcCommandDefinition(
                command_name=command_name,
                syntax=syntax,
                function_name=function_name,
                brief=brief,
                header=header_path,
            )
        )

    for definitions in command_inventory.values():
        definitions.sort(key=lambda item: (item.syntax, item.function_name))
    return command_inventory


def _normalize_value(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        return cleaned[1:-1].strip()
    return cleaned.strip("'\"").strip()


def _expand_text_macros(text: str, macros: Dict[str, str], max_passes: int = 10) -> str:
    result = text
    for _ in range(max_passes):
        changed = False

        def replace(match) -> str:
            nonlocal changed
            name = match.group(1) or match.group(4) or ""
            default = match.group(3) if match.group(1) else match.group(6)
            replacement = macros.get(name)
            if replacement is None or replacement == "":
                replacement = default
            if replacement is None:
                return match.group(0)
            replacement = str(replacement)
            if replacement != match.group(0):
                changed = True
            return replacement

        expanded = MACRO_REF_RE.sub(replace, result)
        result = expanded
        if not changed:
            break
    return result


def _parse_int_value(value: str) -> Optional[int]:
    cleaned = _strip_wrapper_pairs(_normalize_value(value))
    if not cleaned:
        return None
    try:
        return int(cleaned, 0)
    except ValueError:
        return None


def _parse_float_value(value: str) -> Optional[float]:
    cleaned = _strip_wrapper_pairs(_normalize_value(value))
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _strip_wrapper_pairs(value: str) -> str:
    cleaned = value.strip()
    while len(cleaned) >= 2:
        if (cleaned[0], cleaned[-1]) in {("(", ")"), ('"', '"'), ("'", "'")}:
            cleaned = cleaned[1:-1].strip()
            continue
        break
    return cleaned


def _strip_inline_comment(line: str) -> str:
    single = False
    double = False
    for index, char in enumerate(line):
        if char == "'" and not double:
            single = not single
        elif char == '"' and not single:
            double = not double
        elif char == "#" and not single and not double:
            if index == 0 or line[index - 1].isspace():
                return line[:index].rstrip()
    return line.rstrip()


def _parse_require_macro_pairs(line: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for match in re.finditer(r'"([^"]*)"|\'([^\']*)\'', line):
        payload = match.group(1) if match.group(1) is not None else match.group(2) or ""
        payload_pairs, _malformed = _parse_macro_payload(payload)
        pairs.extend(payload_pairs)
    return pairs


def _parse_require_invocation(line: str) -> Tuple[str, str, List[Tuple[str, str]]]:
    payload_pairs = _parse_require_macro_pairs(line)
    stripped = _strip_inline_comment(line).strip()
    if not stripped.startswith("require"):
        return "", "", payload_pairs

    remainder = stripped[len("require") :].strip()
    quoted_segments = re.findall(r'"[^"]*"|\'[^\']*\'', remainder)
    for segment in quoted_segments:
        remainder = remainder.replace(segment, " ", 1)

    tokens = [token.strip() for token in re.split(r"[\s,]+", remainder) if token.strip()]
    module_name = tokens[0] if tokens else ""
    version = tokens[1] if len(tokens) > 1 else ""
    return module_name, version, payload_pairs


def _iter_key_values(line: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for match in PAIR_RE.finditer(line):
        key = match.group("key").strip().upper()
        value = _normalize_value(match.group("value"))
        pairs.append((key, value))
    return pairs


def _read_token(text: str, start: int) -> Tuple[str, int]:
    length = len(text)
    while start < length and text[start].isspace():
        start += 1
    if start >= length:
        return "", start

    quote = text[start] if text[start] in {"'", '"'} else ""
    if quote:
        end = start + 1
        while end < length and text[end] != quote:
            end += 1
        token = text[start : min(end + 1, length)]
        return token, min(end + 1, length)

    depth = 0
    end = start
    while end < length:
        char = text[end]
        if char in "({[":
            depth += 1
        elif char in ")}]" and depth > 0:
            depth -= 1
        elif depth == 0 and (char.isspace() or char == ","):
            break
        end += 1
    return text[start:end], end


def _extract_script_target(line: str) -> str:
    for marker in SCRIPT_EXEC_MARKERS:
        offset = line.find(marker)
        if offset < 0:
            continue
        token, _ = _read_token(line, offset + len(marker))
        return _strip_wrapper_pairs(_normalize_value(token))
    return ""


def _extract_script_call_macro_text(line: str) -> str:
    for marker in SCRIPT_EXEC_MARKERS:
        offset = line.find(marker)
        if offset < 0:
            continue
        _, token_end = _read_token(line, offset + len(marker))
        remaining = line[token_end:].strip()
        if remaining.startswith(","):
            remaining = remaining[1:].strip()
        if not remaining:
            return ""
        payload, _ = _read_token(remaining, 0)
        return _strip_wrapper_pairs(_normalize_value(payload))
    return ""


def _strip_leading_macro_prefixes(line: str) -> str:
    remaining = line.lstrip()
    while remaining:
        if any(remaining.startswith(marker) for marker in SCRIPT_EXEC_MARKERS):
            return remaining
        if remaining.startswith("${"):
            end = remaining.find("}")
            if end < 0:
                return remaining
            remaining = remaining[end + 1 :].lstrip()
            continue
        if remaining.startswith("$("):
            end = remaining.find(")")
            if end < 0:
                return remaining
            remaining = remaining[end + 1 :].lstrip()
            continue
        break
    return remaining


def _extract_command_name(line: str) -> str:
    remaining = _strip_leading_macro_prefixes(line)
    for marker in SCRIPT_EXEC_MARKERS:
        if remaining.startswith(marker):
            return marker
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", remaining)
    return match.group(1) if match else ""


def _extract_module_script_name(target: str) -> str:
    cleaned = _strip_wrapper_pairs(_normalize_value(target))
    patterns = [
        r"(?:\$\{ecmccfg_DIR\}|\$\(ecmccfg_DIR\))(?P<name>[A-Za-z0-9_./-]+\.cmd)$",
        r"(?:\$\{ECMC_CONFIG_ROOT\}|\$\(ECMC_CONFIG_ROOT\))(?P<name>[A-Za-z0-9_./-]+\.cmd)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            return Path(match.group("name")).name
    return ""


def _extract_epics_env_assignment(line: str) -> Optional[Tuple[str, str]]:
    match = re.match(r'epicsEnvSet\(\s*"?([A-Za-z0-9_]+)"?\s*,\s*(.+)\)\s*$', line)
    if not match:
        return None
    name = match.group(1).strip()
    value = _strip_wrapper_pairs(_normalize_value(match.group(2).strip()))
    return name, value


def _extract_epics_env_unset(line: str) -> str:
    match = re.match(r'epicsEnvUnset\(\s*"?([A-Za-z0-9_]+)"?\s*\)\s*$', line)
    return match.group(1).strip() if match else ""


def _extract_dbloadrecords_call(line: str) -> Optional[Tuple[str, Dict[str, str]]]:
    match = re.match(r'dbLoadRecords\(\s*"([^"]+)"\s*,\s*"([^"]*)"\s*\)\s*$', line)
    if not match:
        return None
    db_file = _normalize_value(match.group(1).strip())
    payload = match.group(2).strip()
    pairs, _malformed = _parse_macro_payload(payload)
    return db_file, {key: value for key, value in pairs}


def _extract_ecmc_config_invocation(line: str) -> Optional[Tuple[str, str]]:
    stripped = _strip_inline_comment(line).strip()
    wrapper = _extract_command_name(stripped)
    if wrapper not in {"ecmcConfig", "ecmcConfigOrDie"}:
        return None

    remaining = _strip_leading_macro_prefixes(stripped)
    if not remaining.startswith(wrapper):
        return None

    token, _ = _read_token(remaining, len(wrapper))
    command_text = _strip_wrapper_pairs(_normalize_value(token))
    if not command_text:
        return None
    return wrapper, command_text


def _parse_macro_payload(payload: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    pairs: List[Tuple[str, str]] = []
    malformed: List[str] = []
    if not payload:
        return pairs, malformed

    for part in _split_top_level(payload):
        token = part.strip()
        if not token:
            continue
        if "=" not in token:
            malformed.append(token)
            continue
        key, value = token.split("=", 1)
        key = key.strip().upper()
        value = _normalize_value(value)
        if not key or not re.fullmatch(r"[A-Z0-9_]+", key):
            malformed.append(token)
            continue
        pairs.append((key, value))
    return pairs, malformed


def _is_yaml_loader_script(module_script_name: str) -> bool:
    return module_script_name in {"loadYamlAxis.cmd", "loadYamlEnc.cmd", "loadYamlPlc.cmd"}


def _yaml_loader_schema_kind(module_script_name: str) -> str:
    mapping = {
        "loadYamlAxis.cmd": "axis",
        "loadYamlEnc.cmd": "encoder",
        "loadYamlPlc.cmd": "plc",
    }
    return mapping.get(module_script_name, "")


def _iter_macro_references(text: str) -> List[Tuple[str, bool]]:
    refs: List[Tuple[str, bool]] = []
    index = 0
    length = len(text)

    while index < length - 1:
        opener = text[index : index + 2]
        if opener not in {"${", "$("}:
            index += 1
            continue

        closer = "}" if opener == "${" else ")"
        depth = 1
        cursor = index + 2
        while cursor < length and depth > 0:
            token = text[cursor : cursor + 2]
            if token == opener:
                depth += 1
                cursor += 2
                continue
            if text[cursor] == closer:
                depth -= 1
                cursor += 1
                continue
            cursor += 1

        if depth != 0:
            index += 2
            continue

        body = text[index + 2 : cursor - 1]
        name_match = re.match(r"([A-Z0-9_]+)", body)
        if not name_match:
            index = cursor
            continue

        name = name_match.group(1).strip()
        has_default = False
        remainder = body[name_match.end() :]
        if remainder.startswith("="):
            has_default = True
            refs.extend(_iter_macro_references(remainder[1:]))

        refs.append((name, has_default))
        index = cursor

    return refs


@lru_cache(maxsize=512)
def _scan_file_macro_usage(text: str) -> FileMacroUsage:
    used: Set[str] = set()
    required: Set[str] = set()

    for raw_line in text.splitlines():
        line = _strip_inline_comment(raw_line)
        if not line.strip():
            continue
        for name, has_default in _iter_macro_references(line):
            used.add(name)
            if not has_default:
                required.add(name)
    return FileMacroUsage(used=used, required=required)


def _read_text_from_buffers(path: Path, buffer_lookup: Optional[Dict[Path, str]]) -> Optional[str]:
    resolved = path.resolve()
    if buffer_lookup and resolved in buffer_lookup:
        return buffer_lookup[resolved]
    if resolved.exists():
        return _read_text(resolved)
    return None


@lru_cache(maxsize=2048)
def _resolve_plc_include_reference_cached(
    include_name: str,
    current_dir_str: str,
    include_path_strs: Tuple[str, ...],
) -> Optional[str]:
    cleaned = _normalize_value(include_name)
    if not cleaned or "${" in cleaned or "$(" in cleaned:
        return None

    current_dir = Path(current_dir_str)
    include_paths = [Path(path_str) for path_str in include_path_strs]
    candidate = Path(cleaned)
    if candidate.is_absolute():
        return str(candidate.resolve())

    search_dirs = [current_dir.resolve(), *[path.resolve() for path in include_paths]]
    for directory in search_dirs:
        candidates = [directory / cleaned]
        if directory.name != "plc_lib":
            candidates.append(directory / "plc_lib" / cleaned)
        for resolved_candidate in candidates:
            resolved = resolved_candidate.resolve()
            if resolved.exists():
                return str(resolved)
        if "/" not in cleaned and "\\" not in cleaned:
            recursive_roots = [directory]
            if directory.name != "plc_lib":
                recursive_roots.append(directory / "plc_lib")
            for recursive_root in recursive_roots:
                if not recursive_root.exists() or not recursive_root.is_dir():
                    continue
                matches = sorted(recursive_root.rglob(cleaned))
                if matches:
                    return str(matches[0].resolve())

    return str((current_dir / cleaned).resolve())


def _resolve_plc_include_reference(include_name: str, current_dir: Path, include_paths: List[Path]) -> Optional[Path]:
    resolved = _resolve_plc_include_reference_cached(
        include_name,
        str(current_dir.resolve()),
        tuple(str(path.resolve()) for path in include_paths),
    )
    return Path(resolved) if resolved else None


def _freeze_buffer_lookup(buffer_lookup: Optional[Dict[Path, str]]) -> Tuple[Tuple[str, str], ...]:
    if not buffer_lookup:
        return ()
    return tuple(sorted((str(path.resolve()), text) for path, text in buffer_lookup.items()))


@lru_cache(maxsize=256)
def _scan_plc_tree_cached(
    plc_path_str: str,
    include_path_strs: Tuple[str, ...],
    macro_scope_items: Tuple[Tuple[str, str], ...],
    buffer_items: Tuple[Tuple[str, str], ...],
    active_stack_strs: Tuple[str, ...],
) -> Tuple[List[ExpandedTextLine], FileMacroUsage, FileMacroUsage, List[ValidationIssue]]:
    plc_path = Path(plc_path_str)
    include_paths = [Path(path_str) for path_str in include_path_strs]
    macro_scope = dict(macro_scope_items)
    buffer_lookup = {Path(path_str): text for path_str, text in buffer_items}
    active_stack = tuple(Path(path_str) for path_str in active_stack_strs)
    return _scan_plc_tree_impl(plc_path, include_paths, macro_scope, buffer_lookup, active_stack)


def _scan_plc_tree_impl(
    plc_path: Path,
    include_paths: List[Path],
    macro_scope: Dict[str, str],
    buffer_lookup: Optional[Dict[Path, str]],
    active_stack: Tuple[Path, ...] = (),
) -> Tuple[List[ExpandedTextLine], FileMacroUsage, FileMacroUsage, List[ValidationIssue]]:
    resolved = plc_path.resolve()
    if resolved in active_stack:
        return (
            [],
            FileMacroUsage(used=set(), required=set()),
            FileMacroUsage(used=set(), required=set()),
            [
                ValidationIssue(
                    severity="warning",
                    source=resolved,
                    line=1,
                    message=f"Recursive PLC include detected for {resolved.name}",
                    target=resolved,
                )
            ],
        )

    text = _read_text_from_buffers(resolved, buffer_lookup)
    if text is None:
        return (
            [],
            FileMacroUsage(used=set(), required=set()),
            FileMacroUsage(used=set(), required=set()),
            [
                ValidationIssue(
                    severity="error",
                    source=resolved,
                    line=1,
                    message=f"Cannot inspect PLC include because file is missing: {resolved.name}",
                    target=resolved,
                )
            ],
        )

    expanded_lines: List[ExpandedTextLine] = []
    raw_used: Set[str] = set()
    raw_required: Set[str] = set()
    unresolved_used: Set[str] = set()
    unresolved_required: Set[str] = set()
    issues: List[ValidationIssue] = []
    pending_substitute: Dict[str, str] = {}
    child_stack = (*active_stack, resolved)

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_inline_comment(raw_line)
        stripped = line.strip()
        if not stripped:
            continue

        substitute_match = re.match(r'substitute\s+"([^"]*)"\s*$', stripped)
        if substitute_match:
            substitute_pairs, malformed = _parse_macro_payload(substitute_match.group(1))
            pending_substitute = {key: value for key, value in substitute_pairs}
            for malformed_token in malformed:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        source=resolved,
                        line=line_no,
                        message=f"Malformed PLC substitute segment: {malformed_token}",
                        target=resolved,
                    )
                )
            continue

        include_match = re.match(r'include\s+"([^"]+)"\s*$', stripped)
        if include_match:
            include_target = _resolve_plc_include_reference(include_match.group(1), resolved.parent, include_paths)
            if include_target is None:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        source=resolved,
                        line=line_no,
                        message=f"Unable to resolve PLC include path: {include_match.group(1)}",
                    )
                )
            elif not include_target.exists():
                issues.append(
                    ValidationIssue(
                        severity="error",
                        source=resolved,
                        line=line_no,
                        message=f"Missing PLC include: {include_match.group(1)}",
                        target=include_target,
                    )
                )
            else:
                child_scope = dict(macro_scope)
                child_scope.update(pending_substitute)
                child_lines, child_raw_usage, child_unresolved_usage, child_issues = _scan_plc_tree_cached(
                    str(include_target.resolve()),
                    tuple(str(path.resolve()) for path in include_paths),
                    tuple(sorted(child_scope.items())),
                    _freeze_buffer_lookup(buffer_lookup),
                    tuple(str(path.resolve()) for path in child_stack),
                )
                expanded_lines.extend(child_lines)
                raw_used.update(child_raw_usage.used)
                raw_required.update(child_raw_usage.required)
                unresolved_used.update(child_unresolved_usage.used)
                unresolved_required.update(child_unresolved_usage.required)
                issues.extend(child_issues)
            pending_substitute = {}
            continue

        expanded_line = _expand_text_macros(raw_line, macro_scope)
        expanded_lines.append(ExpandedTextLine(source=resolved, line=line_no, text=expanded_line))
        raw_usage = _scan_file_macro_usage(raw_line)
        unresolved_usage = _scan_file_macro_usage(expanded_line)
        raw_used.update(raw_usage.used)
        raw_required.update(raw_usage.required)
        unresolved_used.update(unresolved_usage.used)
        unresolved_required.update(unresolved_usage.required)

    return (
        expanded_lines,
        FileMacroUsage(used=raw_used, required=raw_required),
        FileMacroUsage(used=unresolved_used, required=unresolved_required),
        issues,
    )


def _scan_plc_tree(
    plc_path: Path,
    include_paths: List[Path],
    macro_scope: Dict[str, str],
    buffer_lookup: Optional[Dict[Path, str]],
    active_stack: Tuple[Path, ...] = (),
) -> Tuple[List[ExpandedTextLine], FileMacroUsage, FileMacroUsage, List[ValidationIssue]]:
    return _scan_plc_tree_cached(
        str(plc_path.resolve()),
        tuple(str(path.resolve()) for path in include_paths),
        tuple(sorted(macro_scope.items())),
        _freeze_buffer_lookup(buffer_lookup),
        tuple(str(path.resolve()) for path in active_stack),
    )


def _entry_template_matches(entry_name: str, entry_template: str) -> bool:
    pattern = re.escape(entry_template)
    pattern = MACRO_REF_RE.sub(r"[A-Za-z0-9_:-]+", pattern)
    return bool(re.fullmatch(pattern, entry_name))


def _hardware_entry_exists(entry_name: str, hw_desc: str, inventory: RepositoryInventory) -> Tuple[bool, bool]:
    if entry_name in GENERIC_EC_ENTRY_NAMES:
        return True, True

    if hw_desc not in inventory.hardware_entries:
        return False, False

    entry_templates = inventory.hardware_entries[hw_desc]
    return any(_entry_template_matches(entry_name, template) for template in entry_templates), True


def _validate_expanded_ec_links(
    lines: List[ExpandedTextLine],
    current_master_id: int,
    slave_hw_desc_by_id: Dict[int, str],
    inventory: RepositoryInventory,
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    for expanded_line in lines:
        line = _strip_inline_comment(expanded_line.text)
        if not line.strip():
            continue

        for match in EC_LINK_RE.finditer(line):
            master_id = int(match.group("master"))
            slave_id = int(match.group("slave"))
            entry_name = match.group("entry")

            if entry_name == "mm":
                continue

            if master_id != current_master_id:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        source=expanded_line.source,
                        line=expanded_line.line,
                        message=(
                            f"EtherCAT link 'ec{master_id}.s{slave_id}.{entry_name}' uses master {master_id}, "
                            f"but the active master is {current_master_id}"
                        ),
                        target=expanded_line.source,
                    )
                )
                continue

            hw_desc = slave_hw_desc_by_id.get(slave_id)
            if hw_desc is None:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        source=expanded_line.source,
                        line=expanded_line.line,
                        message=(
                            f"EtherCAT link 'ec{master_id}.s{slave_id}.{entry_name}' refers to slave {slave_id}, "
                            "but that slave was not added with addSlave.cmd or configureSlave.cmd before this load"
                        ),
                        target=expanded_line.source,
                    )
                )
                continue

            exists, known_hw = _hardware_entry_exists(entry_name, hw_desc, inventory)
            if not known_hw:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        source=expanded_line.source,
                        line=expanded_line.line,
                        message=(
                            f"Cannot validate EtherCAT entry '{entry_name}' for HW_DESC '{hw_desc}' because no "
                            "hardware entry inventory is available"
                        ),
                        target=expanded_line.source,
                    )
                )
            elif not exists:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        source=expanded_line.source,
                        line=expanded_line.line,
                        message=(
                            f"EtherCAT entry '{entry_name}' is not defined by HW_DESC '{hw_desc}' "
                            f"for slave {slave_id}"
                        ),
                        target=expanded_line.source,
                    )
                )

    return issues


def _component_validation_context(
    expanded_key_map: Dict[str, str],
    env_values: Dict[str, str],
    slave_hw_desc_by_id: Dict[int, str],
    inventory: RepositoryInventory,
) -> Dict[str, object]:
    component_slave_id = _parse_int_value(expanded_key_map.get("COMP_S_ID", env_values.get("ECMC_EC_SLAVE_NUM", "")))
    slave_hw_desc = slave_hw_desc_by_id.get(component_slave_id, "") if component_slave_id is not None else ""
    explicit_ec_comp_type = _normalize_value(expanded_key_map.get("EC_COMP_TYPE", env_values.get("ECMC_EC_COMP_TYPE", ""))) or ""
    fallback_hw_type = _normalize_value(env_values.get("ECMC_EC_HWTYPE", "")) or ""
    if fallback_hw_type:
        fallback_hw_type = inventory.hardware_component_types.get(fallback_hw_type, fallback_hw_type)
    expected_ec_comp_type = inventory.hardware_component_types.get(slave_hw_desc, slave_hw_desc) if slave_hw_desc else ""
    resolved_ec_comp_type = explicit_ec_comp_type or expected_ec_comp_type or fallback_hw_type
    support_map = inventory.component_support.get(resolved_ec_comp_type, {}) if resolved_ec_comp_type else {}
    return {
        "component_slave_id": component_slave_id,
        "slave_hw_desc": slave_hw_desc,
        "explicit_ec_comp_type": explicit_ec_comp_type,
        "expected_ec_comp_type": expected_ec_comp_type,
        "resolved_ec_comp_type": resolved_ec_comp_type,
        "support_map": support_map,
    }


@lru_cache(maxsize=512)
def _parse_simple_yaml_paths(text: str) -> List[ParsedMappingLine]:
    entries: List[ParsedMappingLine] = []
    stack: List[Tuple[int, str]] = []

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_inline_comment(raw_line)
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped.startswith("- "):
            continue
        if ":" not in stripped:
            continue

        key, remainder = stripped.split(":", 1)
        key = key.strip().strip("'\"")
        if not key:
            continue

        while stack and indent <= stack[-1][0]:
            stack.pop()

        path_parts = [item[1] for item in stack] + [key]
        path = ".".join(path_parts)
        value = remainder.strip() or None
        entries.append(ParsedMappingLine(path=path, value=value, line=line_no))

        if value is None:
            stack.append((indent, key))

    return entries


def _schema_key_type_matches(value: str, declared_types: str) -> bool:
    if "${" in value or "$(" in value or "{{" in value:
        return True

    options = {item.strip().lower() for item in declared_types.split() if item.strip()}
    lowered = _normalize_value(value).strip().lower()
    if "string" in options:
        return True
    if "boolean" in options and lowered in {"true", "false", "yes", "no", "0", "1"}:
        return True
    if "integer" in options and re.fullmatch(r"[-+]?\d+", value.strip()):
        return True
    if "float" in options and re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value.strip()):
        return True
    return not options


def _get_schema_selector_default(schema: Dict[str, object], selector_path: str) -> Optional[str]:
    for schema_name, schema_info in schema.items():
        if not isinstance(schema_info, dict):
            continue
        schema_map = schema_info.get("schema")
        if not isinstance(schema_map, dict):
            continue
        selector_info = schema_map.get(selector_path)
        if isinstance(selector_info, dict) and "default" in selector_info:
            return str(selector_info["default"])
    return None


def _normalize_schema_selector_value(selector_path: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip("'\" ").lower()
    if selector_path == "axis.type":
        aliases = {
            "joint": "1",
            "endeffector": "2",
        }
        return aliases.get(normalized, normalized)
    return normalized


def _select_grand_schema_variant(
    schema_kind: str,
    parsed_entries: List[ParsedMappingLine],
    ecb_schema: Dict[str, object],
) -> Dict[str, List[str]]:
    grand_schema = ecb_schema.get("grandSchema", {})
    kind_entries = grand_schema.get(schema_kind, {}) if isinstance(grand_schema, dict) else {}
    if not isinstance(kind_entries, dict):
        return {"required": [], "optional": []}

    values = {entry.path: entry.value for entry in parsed_entries if entry.value is not None}
    if len(kind_entries) == 1:
        only_value = next(iter(kind_entries.values()))
        if isinstance(only_value, dict):
            return {
                "required": str(only_value.get("required", "")).split(),
                "optional": str(only_value.get("optional", "")).split(),
            }
        return {"required": [], "optional": []}

    for selector, section_data in kind_entries.items():
        if not isinstance(section_data, dict) or "=" not in selector:
            continue
        selector_path, expected_value = selector.split("=", 1)
        actual_value = values.get(selector_path)
        if actual_value is None:
            actual_value = _get_schema_selector_default(ecb_schema, selector_path)
        normalized_actual = _normalize_schema_selector_value(selector_path, actual_value)
        normalized_expected = _normalize_schema_selector_value(selector_path, expected_value)
        if normalized_actual is not None and normalized_actual == normalized_expected:
            return {
                "required": str(section_data.get("required", "")).split(),
                "optional": str(section_data.get("optional", "")).split(),
            }
    return {"required": [], "optional": []}


def _validate_ecb_yaml(
    yaml_path: Path,
    yaml_text: str,
    schema_kind: str,
    ecb_schema: Optional[Dict[str, object]],
) -> List[ValidationIssue]:
    if ecb_schema is None:
        return []

    parsed_entries = _parse_simple_yaml_paths(yaml_text)
    if not parsed_entries:
        return [
            ValidationIssue(
                severity="warning",
                source=yaml_path,
                line=1,
                message=f"Unable to parse YAML structure for ECB schema validation ({schema_kind})",
                target=yaml_path,
            )
        ]

    active = _select_grand_schema_variant(schema_kind, parsed_entries, ecb_schema)
    schema_names = active["required"] + active["optional"]
    schema_defs: Dict[str, Dict[str, object]] = {}
    allow_any_prefixes: Set[str] = set()
    required_keys: Dict[str, Tuple[str, str]] = {}
    required_identifiers: Dict[str, str] = {}

    for schema_name in schema_names:
        schema_info = ecb_schema.get(schema_name)
        if not isinstance(schema_info, dict):
            continue
        identifier = str(schema_info.get("identifier", "")).strip()
        if identifier:
            if schema_name in active["required"]:
                required_identifiers[identifier] = schema_name
        if schema_info.get("allowAnySubkey") and identifier:
            allow_any_prefixes.add(identifier)
        schema_map = schema_info.get("schema")
        if not isinstance(schema_map, dict):
            continue
        for key_path, key_info in schema_map.items():
            if not isinstance(key_info, dict):
                continue
            schema_defs[str(key_path)] = key_info
            if key_info.get("required"):
                required_keys[str(key_path)] = (schema_name, identifier)

    issues: List[ValidationIssue] = []
    line_by_path = {entry.path: entry.line for entry in parsed_entries}
    value_by_path = {entry.path: entry.value for entry in parsed_entries if entry.value is not None}
    top_level_keys = {entry.path for entry in parsed_entries if "." not in entry.path}

    for identifier, schema_name in required_identifiers.items():
        if identifier not in top_level_keys:
            issues.append(
                ValidationIssue(
                    severity="error",
                    source=yaml_path,
                    line=1,
                    message=f"Missing required YAML section '{identifier}' for {schema_kind} ({schema_name})",
                    target=yaml_path,
                )
            )

    for required_key, (schema_name, identifier) in required_keys.items():
        identifier_present = any(
            existing_path == identifier or existing_path.startswith(f"{identifier}.")
            for existing_path in line_by_path
        )
        if identifier_present and required_key not in value_by_path:
            issues.append(
                ValidationIssue(
                    severity="error",
                    source=yaml_path,
                    line=line_by_path.get(required_key.rsplit(".", 1)[0], 1),
                    message=f"Missing required YAML key '{required_key}' for {schema_kind} ({schema_name})",
                    target=yaml_path,
                )
            )

    for entry in parsed_entries:
        if entry.value is None:
            continue
        exact = schema_defs.get(entry.path)
        if exact is None:
            if not any(entry.path == prefix or entry.path.startswith(f"{prefix}.") for prefix in allow_any_prefixes):
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        source=yaml_path,
                        line=entry.line,
                        message=f"YAML key '{entry.path}' is not defined in ECB schema '{schema_kind}'",
                        target=yaml_path,
                    )
                )
            continue

        declared_types = str(exact.get("type", "")).strip()
        if declared_types and not _schema_key_type_matches(entry.value, declared_types):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    source=yaml_path,
                    line=entry.line,
                    message=f"YAML value for '{entry.path}' does not match ECB schema type '{declared_types}'",
                    target=yaml_path,
                )
            )

    return issues


def _scan_known_commands(ecmccfg_root: Path) -> Set[str]:
    known = set(KNOWN_IOCSH_COMMANDS) | set(SCRIPT_EXEC_MARKERS)
    candidate_files: List[Path] = []

    startup_cmd = ecmccfg_root / "startup.cmd"
    if startup_cmd.exists():
        candidate_files.append(startup_cmd)

    for rel_dir in ("scripts", "general", "motion", "naming", "hardware", "examples"):
        base = ecmccfg_root / rel_dir
        if not base.exists():
            continue
        candidate_files.extend(sorted(base.rglob("*.cmd")))
        candidate_files.extend(sorted(base.rglob("*.script")))

    for candidate in candidate_files:
        try:
            text = _read_text(candidate)
        except Exception:
            continue
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith(COMMENT_PREFIXES):
                continue
            line = _strip_inline_comment(raw_line)
            if not line.strip():
                continue
            command_name = _extract_command_name(line)
            if command_name:
                known.add(command_name)

    return known


def _looks_like_local_path(value: str) -> bool:
    if not value or "${" in value or "$(" in value:
        return False
    if value.startswith("-"):
        return False
    if value.startswith(("./", "../", "/")):
        return True
    if "/" in value:
        return True
    suffix = Path(value).suffix.lower()
    return suffix in PATH_SUFFIXES


def _resolve_reference(value: str, base_dir: Path) -> Optional[Path]:
    cleaned = _normalize_value(value)
    if not _looks_like_local_path(cleaned):
        return None
    path = Path(cleaned)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _resolve_inc_paths(value: str, base_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for chunk in value.split(":"):
        candidate = chunk.strip()
        if not candidate or candidate == ".":
            paths.append(base_dir.resolve())
            continue
        resolved = _resolve_reference(candidate, base_dir)
        if resolved is not None:
            paths.append(resolved)
    return paths


def _extract_plc_symbol_inventory(lines: List[ExpandedTextLine]) -> Dict[str, Set[str]]:
    inventory = {"static": set(), "global": set()}
    in_var_block = False
    for expanded_line in lines:
        text = expanded_line.text.split("#", 1)[0]
        text = text.split("//", 1)[0].strip()
        if not text:
            continue
        upper_text = text.upper()
        if upper_text == "VAR":
            in_var_block = True
            continue
        if upper_text == "END_VAR":
            in_var_block = False
            continue
        if in_var_block:
            declaration_match = PLC_VAR_DECL_RE.match(text)
            if declaration_match:
                inventory[declaration_match.group("scope")].add(declaration_match.group("name"))
                continue
        for match in PLC_SYMBOL_RE.finditer(text):
            inventory[match.group("scope")].add(match.group("name"))
    return inventory


def _parse_plc_asyn_name(asyn_name: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    plc_match = re.search(r"\bplcs\.plc(?P<plc>\d+)\.(?P<scope>static|global)\.(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b", asyn_name)
    if plc_match:
        return _parse_int_value(plc_match.group("plc")), plc_match.group("scope"), plc_match.group("name")
    global_match = re.search(r"\bplcs\.(?P<scope>global)\.(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b", asyn_name)
    if global_match:
        return None, global_match.group("scope"), global_match.group("name")
    return None, None, None


def _is_project_script(path: Path) -> bool:
    return path.suffix.lower() in SCRIPT_EXTENSIONS


def _is_internal_helper_script(path: Path, inventory: RepositoryInventory) -> bool:
    if inventory.ecmccfg_root is None:
        return False
    try:
        resolved = path.resolve()
        scripts_root = (inventory.ecmccfg_root.resolve() / "scripts").resolve()
        if scripts_root not in resolved.parents:
            return False
    except Exception:
        return False
    return resolved.name in inventory.module_scripts


@lru_cache(maxsize=2048)
def _read_text_cached(resolved_path: str, mtime_ns: int, size: int) -> str:
    return Path(resolved_path).read_text(encoding="utf-8", errors="ignore")


def _read_text(path: Path) -> str:
    resolved = path.resolve()
    stat_result = resolved.stat()
    return _read_text_cached(str(resolved), stat_result.st_mtime_ns, stat_result.st_size)


def _validate_single_file(
    path: Path,
    text: str,
    inventory: RepositoryInventory,
    buffer_lookup: Optional[Dict[Path, str]],
) -> Tuple[List[ValidationIssue], List[FileReference], List[Path]]:
    issues: List[ValidationIssue] = []
    references: List[FileReference] = []
    nested_scripts: List[Path] = []
    base_dir = path.parent
    env_values: Dict[str, str] = {}
    if inventory.ecmccfg_root is not None:
        config_root = inventory.ecmccfg_root.resolve() / "scripts"
        if not config_root.exists():
            config_root = inventory.ecmccfg_root.resolve()
        root_str = str(config_root)
        if not root_str.endswith("/"):
            root_str = f"{root_str}/"
        env_values["ecmccfg_DIR"] = root_str
        env_values["ECMC_CONFIG_ROOT"] = root_str
    current_master_id = 0
    next_slave_id = 0
    slave_hw_desc_by_id: Dict[int, str] = {}
    plc_symbols_by_id: Dict[int, Dict[str, Set[str]]] = {}
    plc_file_by_id: Dict[int, Path] = {}
    plc_global_symbols: Set[str] = set()

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(COMMENT_PREFIXES):
            continue

        line = _strip_inline_comment(raw_line)
        if not line.strip():
            continue

        command_name = _extract_command_name(line)
        if command_name and command_name not in inventory.known_commands:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    source=path,
                    line=line_no,
                    message=f"Command '{command_name}' is not recognized as a valid startup command",
                )
            )

        if command_name == "require":
            for key, value in _parse_require_macro_pairs(line):
                expanded_value = _expand_text_macros(value, env_values)
                env_values[key] = expanded_value
                if key == "MASTER_ID":
                    master_id = _parse_int_value(expanded_value)
                    if master_id is not None:
                        current_master_id = master_id
                        env_values["MASTER_ID"] = str(master_id)
                        env_values["ECMC_EC_MASTER_ID"] = str(master_id)

        env_assignment = _extract_epics_env_assignment(line)
        if env_assignment is not None:
            env_name, env_value = env_assignment
            expanded_env_value = _expand_text_macros(env_value, env_values)
            env_values[env_name] = expanded_env_value
            numeric_env_value = _parse_int_value(expanded_env_value)
            if env_name in {"MASTER_ID", "ECMC_EC_MASTER_ID"} and numeric_env_value is not None:
                current_master_id = numeric_env_value
                env_values["MASTER_ID"] = str(numeric_env_value)
                env_values["ECMC_EC_MASTER_ID"] = str(numeric_env_value)
            elif env_name == "SLAVE_ID" and numeric_env_value is not None:
                next_slave_id = numeric_env_value

        unset_name = _extract_epics_env_unset(line)
        if unset_name:
            env_values.pop(unset_name, None)

        command_target = _extract_script_target(line)
        module_script_name = _extract_module_script_name(command_target)
        macro_payload = _extract_script_call_macro_text(line)
        payload_pairs, malformed_macros = _parse_macro_payload(macro_payload)
        if malformed_macros and module_script_name:
            for malformed in malformed_macros:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        source=path,
                        line=line_no,
                        message=f"Malformed macro segment for '{module_script_name}': {malformed}",
                    )
                )

        key_values = payload_pairs if macro_payload else _iter_key_values(line)
        key_map = {key: value for key, value in key_values if value}
        expanded_key_map = {key: _expand_text_macros(value, env_values) for key, value in key_map.items()}

        if module_script_name == "addMaster.cmd":
            master_id = _parse_int_value(expanded_key_map.get("MASTER_ID", env_values.get("MASTER_ID", "0")))
            if master_id is not None:
                current_master_id = master_id
                env_values["MASTER_ID"] = str(master_id)
                env_values["ECMC_EC_MASTER_ID"] = str(master_id)

        hw_desc = expanded_key_map.get("HW_DESC", env_values.get("HW_DESC", ""))
        if module_script_name in {"addSlave.cmd", "configureSlave.cmd"}:
            requested_slave_id = expanded_key_map.get("SLAVE_ID", env_values.get("SLAVE_ID", str(next_slave_id)))
            assigned_slave_id = _parse_int_value(requested_slave_id)
            if assigned_slave_id is None:
                assigned_slave_id = next_slave_id
            next_slave_id = assigned_slave_id + 1
            env_values["ECMC_EC_SLAVE_NUM"] = str(assigned_slave_id)
            env_values["SLAVE_ID"] = str(next_slave_id)
            if hw_desc:
                env_values["HW_DESC"] = hw_desc
                slave_hw_desc_by_id[assigned_slave_id] = hw_desc

        if hw_desc and "${" not in hw_desc and "$(" not in hw_desc and hw_desc not in inventory.hardware_descs:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    source=path,
                    line=line_no,
                    message=f"HW_DESC '{hw_desc}' was not found in hardware/",
                )
            )

        for key, value in key_values:
            expanded_value = _expand_text_macros(value, env_values)
            if key == "FILE" or key == "LOCAL_CONFIG":
                resolved = _resolve_reference(expanded_value, base_dir)
                if resolved is None:
                    continue
                exists = resolved.exists()
                references.append(FileReference(path, resolved, key.lower(), line_no, exists))
                if not exists:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            source=path,
                            line=line_no,
                            message=f"Missing {key.lower()} reference: {expanded_value}",
                            target=resolved,
                        )
                    )
                elif _is_project_script(resolved):
                    nested_scripts.append(resolved)
            elif key == "CONFIG":
                resolved = _resolve_reference(expanded_value, base_dir)
                if resolved is not None:
                    exists = resolved.exists()
                    references.append(FileReference(path, resolved, "config", line_no, exists))
                    if not exists:
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                source=path,
                                line=line_no,
                                message=f"Missing config reference: {expanded_value}",
                                target=resolved,
                            )
                        )
                    elif _is_project_script(resolved):
                        nested_scripts.append(resolved)
                    continue

                if (
                    hw_desc
                    and "${" not in hw_desc
                    and "$(" not in hw_desc
                    and "${" not in expanded_value
                    and "$(" not in expanded_value
                ):
                    config_name = f"ecmc{hw_desc}{expanded_value}.cmd"
                    config_matches = inventory.hardware_configs.get(config_name, [])
                    config_target = config_matches[0] if config_matches else Path(config_name)
                    exists = bool(config_matches)
                    references.append(FileReference(path, config_target, "hardware-config", line_no, exists))
                    if not exists:
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                source=path,
                                line=line_no,
                                message=f"Missing hardware config '{config_name}' in hardware/",
                                target=config_target,
                            )
                        )
            elif key == "INC":
                for inc_path in _resolve_inc_paths(expanded_value, base_dir):
                    exists = inc_path.exists()
                    references.append(FileReference(path, inc_path, "include-path", line_no, exists))
                    if not exists:
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                source=path,
                                line=line_no,
                                message=f"Missing include path: {inc_path}",
                                target=inc_path,
                            )
                        )

        yaml_macro_usage: Optional[FileMacroUsage] = None
        yaml_target = None
        if module_script_name and _is_yaml_loader_script(module_script_name):
            file_value = expanded_key_map.get("FILE", "")
            yaml_target = _resolve_reference(file_value, base_dir) if file_value else None
            if yaml_target is not None:
                yaml_text = _read_text_from_buffers(yaml_target, buffer_lookup)
                if yaml_text is not None:
                    yaml_macro_usage = _scan_file_macro_usage(yaml_text)
                    schema_kind = _yaml_loader_schema_kind(module_script_name)
                    if schema_kind:
                        issues.extend(_validate_ecb_yaml(yaml_target, yaml_text, schema_kind, inventory.ecb_schema))
                    yaml_scope = dict(env_values)
                    yaml_scope.update(expanded_key_map)
                    expanded_yaml_text = _expand_text_macros(yaml_text, yaml_scope)
                    yaml_lines = [
                        ExpandedTextLine(source=yaml_target, line=index, text=expanded_line)
                        for index, expanded_line in enumerate(expanded_yaml_text.splitlines(), start=1)
                    ]
                    issues.extend(
                        _validate_expanded_ec_links(yaml_lines, current_master_id, slave_hw_desc_by_id, inventory)
                    )
                else:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            source=path,
                            line=line_no,
                            message=f"Cannot inspect YAML macros because file is missing: {file_value}",
                            target=yaml_target,
                        )
                    )

        plc_raw_macro_usage: Optional[FileMacroUsage] = None
        plc_unresolved_macro_usage: Optional[FileMacroUsage] = None
        plc_target = None
        plc_macro_pairs: List[Tuple[str, str]] = []
        current_plc_id = None
        if module_script_name == "loadPLCFile.cmd":
            file_value = expanded_key_map.get("FILE", "")
            plc_target = _resolve_reference(file_value, base_dir) if file_value else None
            plc_payload = expanded_key_map.get("PLC_MACROS", "")
            current_plc_id = _parse_int_value(_normalize_value(expanded_key_map.get("PLC_ID", env_values.get("ECMC_PLC_ID", "0"))) or "0")
            if plc_payload:
                plc_macro_pairs, malformed_plc_macros = _parse_macro_payload(plc_payload)
                for malformed in malformed_plc_macros:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            source=path,
                            line=line_no,
                            message=f"Malformed PLC_MACROS segment for 'loadPLCFile.cmd': {malformed}",
                            target=plc_target,
                        )
                    )
            if plc_target is not None:
                if _read_text_from_buffers(plc_target, buffer_lookup) is None:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            source=path,
                            line=line_no,
                            message=f"Cannot inspect PLC macros because file is missing: {file_value}",
                            target=plc_target,
                        )
                    )
                else:
                    plc_scope = dict(env_values)
                    plc_scope.update({key: value for key, value in plc_macro_pairs})
                    plc_id = _normalize_value(expanded_key_map.get("PLC_ID", env_values.get("ECMC_PLC_ID", "0"))) or "0"
                    plc_scope["SELF_ID"] = plc_id
                    plc_scope["SELF"] = f"plc{plc_id}"
                    plc_scope["M_ID"] = str(current_master_id)
                    plc_scope["M"] = f"ec{current_master_id}"
                    plc_include_paths = _resolve_inc_paths(expanded_key_map.get("INC", ""), base_dir)
                    if not plc_include_paths:
                        plc_include_paths = [plc_target.parent.resolve()]
                    plc_lines, plc_raw_macro_usage, plc_unresolved_macro_usage, plc_tree_issues = _scan_plc_tree(
                        plc_target,
                        plc_include_paths,
                        plc_scope,
                        buffer_lookup,
                    )
                    if current_plc_id is not None:
                        plc_symbols_by_id[current_plc_id] = _extract_plc_symbol_inventory(plc_lines)
                        plc_file_by_id[current_plc_id] = plc_target
                        plc_global_symbols |= plc_symbols_by_id[current_plc_id]["global"]
                    issues.extend(plc_tree_issues)
                    issues.extend(
                        _validate_expanded_ec_links(plc_lines, current_master_id, slave_hw_desc_by_id, inventory)
                    )

        if module_script_name == "applyComponent.cmd" and (
            inventory.component_definitions or inventory.component_support
        ):
            component_name = expanded_key_map.get("COMP", "").strip()
            component_context = _component_validation_context(expanded_key_map, env_values, slave_hw_desc_by_id, inventory)
            component_slave_id = component_context["component_slave_id"]
            slave_hw_desc = str(component_context["slave_hw_desc"])
            explicit_ec_comp_type = str(component_context["explicit_ec_comp_type"])
            expected_ec_comp_type = str(component_context["expected_ec_comp_type"])
            resolved_ec_comp_type = str(component_context["resolved_ec_comp_type"])
            support_map = dict(component_context["support_map"])
            definition = inventory.component_definitions.get(component_name) if component_name else None

            if component_name and inventory.component_definitions and definition is None:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        source=path,
                        line=line_no,
                        message=f"Component '{component_name}' was not found in ecmccomp",
                    )
                )

            if (
                component_slave_id is not None
                and component_slave_id not in slave_hw_desc_by_id
                and not explicit_ec_comp_type
            ):
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        source=path,
                        line=line_no,
                        message=(
                            "Cannot verify component support because slave {} has not been added with "
                            "addSlave.cmd or configureSlave.cmd before this component"
                        ).format(component_slave_id),
                    )
                )

            if explicit_ec_comp_type and expected_ec_comp_type and explicit_ec_comp_type != expected_ec_comp_type:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        source=path,
                        line=line_no,
                        message=(
                            "EC_COMP_TYPE '{}' does not match slave {} HW_DESC '{}' "
                            "(expected '{}')"
                        ).format(
                            explicit_ec_comp_type,
                            component_slave_id if component_slave_id is not None else "?",
                            slave_hw_desc or "<unknown>",
                            expected_ec_comp_type,
                        ),
                    )
                )

            if resolved_ec_comp_type and inventory.component_support and not support_map:
                slave_details = []
                if component_slave_id is not None:
                    slave_details.append(f"slave {component_slave_id}")
                if slave_hw_desc:
                    slave_details.append(f"HW_DESC '{slave_hw_desc}'")
                suffix = " for {}".format(", ".join(slave_details)) if slave_details else ""
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        source=path,
                        line=line_no,
                        message=(
                            "No component support was found in ecmccomp for EC_COMP_TYPE '{}'{}"
                        ).format(resolved_ec_comp_type, suffix),
                    )
                )

            if definition is not None and support_map and definition.comp_type not in support_map:
                slave_details = []
                if component_slave_id is not None:
                    slave_details.append(f"slave {component_slave_id}")
                if slave_hw_desc:
                    slave_details.append(f"HW_DESC '{slave_hw_desc}'")
                suffix = " ({})".format(", ".join(slave_details)) if slave_details else ""
                issues.append(
                    ValidationIssue(
                        severity="error",
                        source=path,
                        line=line_no,
                        message=(
                            "Component '{}' of type '{}' is not supported by EC_COMP_TYPE '{}'{}"
                        ).format(
                            component_name,
                            definition.comp_type,
                            resolved_ec_comp_type or "<unset>",
                            suffix,
                        ),
                    )
                )

            macros_payload = expanded_key_map.get("MACROS", "").strip()
            if definition is not None and macros_payload:
                macro_pairs, malformed_component_macros = _parse_macro_payload(macros_payload)
                for malformed in malformed_component_macros:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            source=path,
                            line=line_no,
                            message=f"Malformed MACROS segment for 'applyComponent.cmd': {malformed}",
                        )
                    )
                support = support_map.get(definition.comp_type) if support_map else None
                allowed_macros = support.supported_macros if support is not None else set()
                invalid_macros = sorted(key for key, _value in macro_pairs if allowed_macros and key not in allowed_macros)
                if invalid_macros:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            source=path,
                            line=line_no,
                            message=(
                                "Component macros {} are not supported for '{}' on EC_COMP_TYPE '{}'"
                            ).format(
                                ", ".join(invalid_macros),
                                component_name,
                                resolved_ec_comp_type or "<unset>",
                            ),
                        )
                    )

        if module_script_name == "configureAxis.cmd":
            cfg_macros_payload = expanded_key_map.get("CFG_MACROS", "").strip()
            if cfg_macros_payload:
                _cfg_macro_pairs, malformed_cfg_macros = _parse_macro_payload(cfg_macros_payload)
                for malformed in malformed_cfg_macros:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            source=path,
                            line=line_no,
                            message=f"Malformed CFG_MACROS segment for 'configureAxis.cmd': {malformed}",
                        )
                    )

        if module_script_name == "loadSubstConfig.cmd":
            subst_macros_payload = expanded_key_map.get("MACROS", "").strip()
            if subst_macros_payload:
                _subst_macro_pairs, malformed_subst_macros = _parse_macro_payload(subst_macros_payload)
                for malformed in malformed_subst_macros:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            source=path,
                            line=line_no,
                            message=f"Malformed MACROS segment for 'loadSubstConfig.cmd': {malformed}",
                        )
                    )

        if module_script_name == "pvtControllerConfig.cmd":
            trigger_entry = expanded_key_map.get("TRG_EC_ENTRY", "").strip()
            if trigger_entry:
                issues.extend(
                    _validate_expanded_ec_links(
                        [ExpandedTextLine(source=path, line=line_no, text=trigger_entry)],
                        current_master_id,
                        slave_hw_desc_by_id,
                        inventory,
                    )
                )
            trigger_duration = expanded_key_map.get("TRG_DUR_S", "").strip()
            if (
                trigger_duration
                and "${" not in trigger_duration
                and "$(" not in trigger_duration
                and _parse_float_value(trigger_duration) is None
            ):
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        source=path,
                        line=line_no,
                        message="TRG_DUR_S '{}' is not a valid numeric value for 'pvtControllerConfig.cmd'".format(
                            trigger_duration
                        ),
                    )
                )

        if module_script_name == "loadLUTFile.cmd":
            lut_id_value = expanded_key_map.get("LUT_ID", "").strip()
            if (
                lut_id_value
                and "${" not in lut_id_value
                and "$(" not in lut_id_value
                and _parse_int_value(lut_id_value) is None
            ):
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        source=path,
                        line=line_no,
                        message="LUT_ID '{}' is not a valid integer for 'loadLUTFile.cmd'".format(lut_id_value),
                    )
                )

        if module_script_name:
            module_matches = inventory.module_scripts.get(module_script_name, [])
            if not module_matches:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        source=path,
                        line=line_no,
                        message=f"Script '{module_script_name}' was not found in scripts/",
                        target=Path(module_script_name),
                    )
                )
            else:
                macro_spec = inventory.module_macro_specs.get(module_script_name, MacroSpec(set(), set()))
                command_macro_usage = inventory.module_macro_usage.get(module_script_name, FileMacroUsage(set(), set()))
                command_defined_keys = set(macro_spec.allowed) | command_macro_usage.used
                passed_keys = {key for key, _value in payload_pairs}
                yaml_used_keys = yaml_macro_usage.used if yaml_macro_usage is not None else set()
                yaml_required_keys = yaml_macro_usage.required if yaml_macro_usage is not None else set()

                allowed_keys = set(command_defined_keys)
                if _is_yaml_loader_script(module_script_name):
                    allowed_keys |= yaml_used_keys

                if not _is_yaml_loader_script(module_script_name):
                    for unknown_key in sorted(passed_keys - allowed_keys):
                        if allowed_keys:
                            issues.append(
                                ValidationIssue(
                                    severity="warning",
                                    source=path,
                                    line=line_no,
                                    message=f"Macro '{unknown_key}' is not defined for '{module_script_name}'",
                                )
                            )
                for missing_key in sorted(macro_spec.required - passed_keys):
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            source=path,
                            line=line_no,
                            message=f"Missing required macro '{missing_key}' for '{module_script_name}'",
                        )
                    )
                if _is_yaml_loader_script(module_script_name) and yaml_macro_usage is not None:
                    defined_keys = passed_keys | set(env_values.keys())
                    for missing_key in sorted(yaml_required_keys - defined_keys):
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                source=path,
                                line=line_no,
                                message=(
                                    f"Missing YAML macro '{missing_key}' required by "
                                    f"{yaml_target.name if yaml_target is not None else 'loaded YAML'} "
                                    f"for '{module_script_name}'"
                                ),
                                target=yaml_target,
                            )
                        )
                    extra_passed_keys = passed_keys - command_defined_keys
                    for unused_key in sorted(extra_passed_keys - yaml_used_keys):
                        issues.append(
                            ValidationIssue(
                                severity="warning",
                                source=path,
                                line=line_no,
                                message=(
                                    f"Macro '{unused_key}' is passed to '{module_script_name}' but not used "
                                    f"in {yaml_target.name if yaml_target is not None else 'the loaded YAML file'}"
                                ),
                                target=yaml_target,
                            )
                        )
                if module_script_name == "loadPLCFile.cmd" and plc_unresolved_macro_usage is not None:
                    reserved_plc_keys = {"SELF_ID", "SELF", "M_ID", "M"}
                    passed_plc_keys = {key for key, _value in plc_macro_pairs}
                    defined_plc_keys = passed_plc_keys | reserved_plc_keys | set(env_values.keys())
                    for missing_key in sorted(plc_unresolved_macro_usage.required - defined_plc_keys):
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                source=path,
                                line=line_no,
                                message=(
                                    f"Missing PLC macro '{missing_key}' required by "
                                    f"{plc_target.name if plc_target is not None else 'the loaded PLC file'} "
                                    "for 'loadPLCFile.cmd'"
                                ),
                                target=plc_target,
                            )
                        )
                    raw_used_keys = plc_raw_macro_usage.used if plc_raw_macro_usage is not None else set()
                    for unused_key in sorted(passed_plc_keys - raw_used_keys - command_defined_keys):
                        issues.append(
                            ValidationIssue(
                                severity="warning",
                                source=path,
                                line=line_no,
                                message=(
                                    f"PLC macro '{unused_key}' is passed to 'loadPLCFile.cmd' but not used "
                                    f"in {plc_target.name if plc_target is not None else 'the loaded PLC file'}"
                                ),
                                target=plc_target,
                            )
                        )

        dbload_call = _extract_dbloadrecords_call(line)
        if dbload_call is not None:
            db_file, db_macros = dbload_call
            if db_file in {"ecmcPlcAnalog.db", "ecmcPlcBinary.db"}:
                expanded_db_macros = {key: _expand_text_macros(value, env_values) for key, value in db_macros.items()}
                plc_id, scope, variable_name = _parse_plc_asyn_name(expanded_db_macros.get("ASYN_NAME", ""))
                if scope == "static" and variable_name:
                    if plc_id is None:
                        issues.append(
                            ValidationIssue(
                                severity="warning",
                                source=path,
                                line=line_no,
                                message="Cannot determine PLC_ID from ASYN_NAME for PLC variable export",
                            )
                        )
                    elif plc_id not in plc_symbols_by_id:
                        issues.append(
                            ValidationIssue(
                                severity="warning",
                                source=path,
                                line=line_no,
                                message="Cannot verify PLC variable because PLC {} has not been loaded yet in this file".format(plc_id),
                            )
                        )
                    elif variable_name not in plc_symbols_by_id[plc_id]["static"]:
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                source=path,
                                line=line_no,
                                message="PLC static variable '{}' was not found in PLC {}".format(variable_name, plc_id),
                                target=plc_file_by_id.get(plc_id),
                            )
                        )
                elif scope == "global" and variable_name:
                    if variable_name not in plc_global_symbols:
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                source=path,
                                line=line_no,
                                message="PLC global variable '{}' was not found in loaded PLC files".format(variable_name),
                            )
                        )

        resolved_command = _resolve_reference(_expand_text_macros(command_target, env_values), base_dir)
        if resolved_command is not None and not module_script_name:
            exists = resolved_command.exists()
            references.append(FileReference(path, resolved_command, "script", line_no, exists))
            if not exists:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        source=path,
                        line=line_no,
                        message=f"Missing nested script: {command_target}",
                        target=resolved_command,
                    )
                )
            elif _is_project_script(resolved_command):
                nested_scripts.append(resolved_command)

    return issues, references, nested_scripts


def _format_tree_details(expanded_key_map: Dict[str, str], preferred: List[str]) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for key in preferred:
        value = expanded_key_map.get(key, "")
        if value:
            rows.append((key, value))
            seen.add(key)
    for key in sorted(k for k, value in expanded_key_map.items() if value and k not in seen):
        rows.append((key, expanded_key_map[key]))
    return rows


def _format_applied_macro_details(
    macro_scope: Dict[str, str],
    macro_usage: Optional[FileMacroUsage],
    preferred: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    if macro_usage is None:
        return []

    rows: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    preferred = preferred or []

    def value_for(key: str) -> str:
        if key in macro_scope and macro_scope[key] != "":
            return macro_scope[key]
        if key in macro_usage.required:
            return "<unresolved>"
        return "<file default>"

    for key in preferred:
        if key not in macro_usage.used or key in seen:
            continue
        rows.append((key, value_for(key)))
        seen.add(key)

    for key in sorted(k for k in macro_usage.used if k not in seen):
        rows.append((key, value_for(key)))

    return rows


def _format_command_macro_details(
    module_script_name: str,
    expanded_key_map: Dict[str, str],
    inventory: RepositoryInventory,
    preferred: List[str],
) -> List[Tuple[str, str]]:
    fallback_rows = _format_tree_details(expanded_key_map, preferred)
    macro_usage = inventory.module_macro_usage.get(module_script_name)
    if macro_usage is None or not macro_usage.used:
        return fallback_rows

    rows: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for key in preferred:
        if key in macro_usage.used and key in expanded_key_map and expanded_key_map[key]:
            rows.append((key, expanded_key_map[key]))
            seen.add(key)
    for key in sorted(k for k, value in expanded_key_map.items() if value and k in macro_usage.used and k not in seen):
        rows.append((key, expanded_key_map[key]))
    return rows


def _quote_startup_value(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        return text
    if "," in text or " " in text:
        return "'" + text.replace("'", "\\'") + "'"
    return text


def _render_startup_command(script_name: str, items: List[Tuple[str, str]]) -> str:
    rendered = []
    for key, value in items:
        cleaned = str(value).strip()
        if not cleaned:
            continue
        rendered.append("{}={}".format(key, _quote_startup_value(cleaned)))
    payload = ", ".join(rendered)
    if not payload:
        return "${SCRIPTEXEC} ${ecmccfg_DIR}%s\n" % script_name
    return '${SCRIPTEXEC} ${ecmccfg_DIR}%s "%s"\n' % (script_name, payload)


def _parse_extra_macro_items(raw_text: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for token in _split_top_level(raw_text or ""):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip()
        value = _normalize_value(value.strip())
        if key:
            items.append((key, value))
    return items


def _lookup_ecmc_command_definitions(
    command_text: str,
    inventory: RepositoryInventory,
) -> List[EcmcCommandDefinition]:
    command_name = _extract_ecmc_inner_command_name(command_text)
    if not command_name:
        return []
    return list(inventory.ecmc_command_definitions.get(command_name, []))


def _normalize_ecmc_command_text(command_text: str) -> str:
    cleaned = _strip_wrapper_pairs(_normalize_value(command_text)).strip()
    if not cleaned:
        return ""
    if not cleaned.startswith("Cfg."):
        return cleaned
    match = re.match(r"Cfg\.(?P<name>[A-Za-z_][A-Za-z0-9_?]*)\((?P<args>.*)\)$", cleaned)
    if not match:
        return cleaned
    arguments = [token.strip() for token in _split_top_level(match.group("args"))]
    return "Cfg.{}({})".format(match.group("name"), ",".join(arguments))


def _extract_startup_objects_from_file(
    path: Path,
    text: str,
    inventory: RepositoryInventory,
    buffer_lookup: Optional[Dict[Path, str]] = None,
) -> Tuple[List[StartupObject], List[Tuple[Path, int]]]:
    objects: List[StartupObject] = []
    nested_scripts: List[Tuple[Path, int]] = []
    base_dir = path.parent
    env_values: Dict[str, str] = {}
    if inventory.ecmccfg_root is not None:
        config_root = inventory.ecmccfg_root.resolve() / "scripts"
        if not config_root.exists():
            config_root = inventory.ecmccfg_root.resolve()
        root_str = str(config_root)
        if not root_str.endswith("/"):
            root_str = f"{root_str}/"
        env_values["ecmccfg_DIR"] = root_str
        env_values["ECMC_CONFIG_ROOT"] = root_str
    current_master_id = 0
    next_slave_id = 0
    last_slave_id = None
    last_axis_line = None
    last_plc_id = None

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(COMMENT_PREFIXES):
            continue

        line = _strip_inline_comment(raw_line)
        if not line.strip():
            continue

        command_name = _extract_command_name(line)
        ecmc_config_call = _extract_ecmc_config_invocation(line)
        if ecmc_config_call is not None:
            wrapper_name, raw_command_text = ecmc_config_call
            normalized_raw_command_text = _normalize_ecmc_command_text(raw_command_text)
            expanded_command_text = _normalize_ecmc_command_text(_expand_text_macros(normalized_raw_command_text, env_values))
            command_defs = _lookup_ecmc_command_definitions(expanded_command_text, inventory)
            inner_command_name = _extract_ecmc_inner_command_name(expanded_command_text) or _extract_ecmc_inner_command_name(
                normalized_raw_command_text
            )
            title_name = inner_command_name or expanded_command_text or normalized_raw_command_text
            detail_rows: List[Tuple[str, str]] = [
                ("WRAPPER", wrapper_name),
                ("COMMAND", normalized_raw_command_text),
            ]
            if expanded_command_text and expanded_command_text != normalized_raw_command_text:
                detail_rows.append(("EXPANDED_COMMAND", expanded_command_text))
            if inner_command_name:
                detail_rows.append(("NAME", inner_command_name))
            if command_defs:
                function_names = ", ".join(sorted({definition.function_name for definition in command_defs if definition.function_name}))
                syntaxes = " | ".join(definition.syntax for definition in command_defs[:4])
                briefs = " | ".join(
                    definition.brief for definition in command_defs if definition.brief
                )
                header_paths = " | ".join(
                    str(definition.header) for definition in command_defs if definition.header is not None
                )
                if function_names:
                    detail_rows.append(("FUNCTION", function_names))
                if syntaxes:
                    detail_rows.append(("SYNTAX", syntaxes))
                if briefs:
                    detail_rows.append(("DESCRIPTION", briefs))
                if header_paths:
                    detail_rows.append(("HEADER", header_paths))
            summary_text = expanded_command_text or raw_command_text
            objects.append(
                StartupObject(
                    kind="ecmc_command",
                    source=path,
                    line=line_no,
                    title=title_name,
                    summary="{}: {}".format(wrapper_name, summary_text),
                    details=detail_rows,
                )
            )
        if command_name == "require":
            module_name, version, require_macro_pairs = _parse_require_invocation(line)
            for key, value in require_macro_pairs:
                expanded_value = _expand_text_macros(value, env_values)
                env_values[key] = expanded_value
                if key == "MASTER_ID":
                    master_id = _parse_int_value(expanded_value)
                    if master_id is not None:
                        current_master_id = master_id
                        env_values["MASTER_ID"] = str(master_id)
                        env_values["ECMC_EC_MASTER_ID"] = str(master_id)
            linked_file_details: List[Tuple[str, str]] = []
            if inventory.ecmccfg_root is not None and module_name == "ecmccfg":
                startup_macro_map = {
                    key: _expand_text_macros(value, env_values)
                    for key, value in require_macro_pairs
                    if value
                }
                if version and "ECMC_VER" not in startup_macro_map:
                    startup_macro_map["ECMC_VER"] = version
                linked_file_details = _format_tree_details(
                    startup_macro_map,
                    ["ECMC_VER", "SYS", "ENG_MODE", "MASTER_ID", "MODE", "EC_RATE", "PVA", "INIT", "SCRIPTEXEC"],
                )

            require_details: List[Tuple[str, str]] = []
            if module_name:
                require_details.append(("MODULE", module_name))
            if version:
                require_details.append(("VERSION", version))
            summary_parts = []
            if module_name:
                summary_parts.append("MODULE={}".format(module_name))
            if version:
                summary_parts.append("VERSION={}".format(version))
            if not summary_parts:
                summary_parts.append("require")
            objects.append(
                StartupObject(
                    kind="require",
                    source=path,
                    line=line_no,
                    title="Require {}".format(module_name or "module"),
                    summary=", ".join(summary_parts),
                    details=require_details,
                    linked_file_details=linked_file_details,
                )
            )

        env_assignment = _extract_epics_env_assignment(line)
        if env_assignment is not None:
            env_name, env_value = env_assignment
            expanded_env_value = _expand_text_macros(env_value, env_values)
            env_values[env_name] = expanded_env_value
            objects.append(
                StartupObject(
                    kind="macro",
                    source=path,
                    line=line_no,
                    title="Macro {}".format(env_name),
                    summary="{}={}".format(env_name, expanded_env_value or env_value),
                    details=[("NAME", env_name), ("VALUE", env_value)],
                )
            )
            numeric_env_value = _parse_int_value(expanded_env_value)
            if env_name in {"MASTER_ID", "ECMC_EC_MASTER_ID"} and numeric_env_value is not None:
                current_master_id = numeric_env_value
                env_values["MASTER_ID"] = str(numeric_env_value)
                env_values["ECMC_EC_MASTER_ID"] = str(numeric_env_value)
            elif env_name == "SLAVE_ID" and numeric_env_value is not None:
                next_slave_id = numeric_env_value
            elif env_name in {"ECMC_PLC_ID", "PLC_ID"} and numeric_env_value is not None:
                last_plc_id = numeric_env_value

        unset_name = _extract_epics_env_unset(line)
        if unset_name:
            env_values.pop(unset_name, None)

        command_target = _extract_script_target(line)
        module_script_name = _extract_module_script_name(command_target)
        macro_payload = _extract_script_call_macro_text(line)
        payload_pairs, _malformed_macros = _parse_macro_payload(macro_payload)
        key_values = payload_pairs if macro_payload else _iter_key_values(line)
        expanded_key_map = {key: _expand_text_macros(value, env_values) for key, value in key_values if value}

        if module_script_name == "addMaster.cmd":
            master_id = _parse_int_value(expanded_key_map.get("MASTER_ID", env_values.get("MASTER_ID", "0")))
            if master_id is not None:
                current_master_id = master_id
                env_values["MASTER_ID"] = str(master_id)
                env_values["ECMC_EC_MASTER_ID"] = str(master_id)
            objects.append(
                StartupObject(
                    kind="master",
                    source=path,
                    line=line_no,
                    title="Master {}".format(env_values.get("MASTER_ID", str(current_master_id or 0))),
                    summary="MASTER_ID={}".format(env_values.get("MASTER_ID", str(current_master_id or 0))),
                    details=_format_tree_details(expanded_key_map, ["MASTER_ID"]),
                )
            )

        if module_script_name == "setRecordUpdateRate.cmd":
            rate_ms = expanded_key_map.get("RATE_MS", env_values.get("ECMC_EC_SAMPLE_RATE_MS", ""))
            if rate_ms:
                env_values["ECMC_SAMPLE_RATE_MS"] = rate_ms
            detail_rows = _format_tree_details(expanded_key_map, ["RATE_MS"])
            command_rows = _format_command_macro_details(
                module_script_name,
                expanded_key_map,
                inventory,
                ["RATE_MS"],
            )
            objects.append(
                StartupObject(
                    kind="record_update_rate",
                    source=path,
                    line=line_no,
                    title="Set Record Rate",
                    summary="RATE_MS={}".format(rate_ms or "<default>"),
                    details=detail_rows,
                    command_details=command_rows,
                )
            )

        if module_script_name == "restoreRecordUpdateRate.cmd":
            restored_rate = env_values.get("ECMC_SAMPLE_RATE_MS_ORIGINAL", env_values.get("ECMC_EC_SAMPLE_RATE_MS", ""))
            if restored_rate:
                env_values["ECMC_SAMPLE_RATE_MS"] = restored_rate
            objects.append(
                StartupObject(
                    kind="restore_record_update_rate",
                    source=path,
                    line=line_no,
                    title="Restore Record Rate",
                    summary="Restore startup record rate",
                    details=[],
                    command_details=[],
                )
            )

        if module_script_name == "applyConfig.cmd":
            env_values["ECMC_EC_APPLY_CFG_DONE"] = "1"
            objects.append(
                StartupObject(
                    kind="apply_config",
                    source=path,
                    line=line_no,
                    title="Apply Config",
                    summary="Apply EtherCAT bus configuration",
                    details=[],
                    command_details=[],
                )
            )

        hw_desc = expanded_key_map.get("HW_DESC", env_values.get("HW_DESC", ""))
        if module_script_name in {"addSlave.cmd", "configureSlave.cmd"}:
            requested_slave_id = expanded_key_map.get("SLAVE_ID", env_values.get("SLAVE_ID", str(next_slave_id)))
            assigned_slave_id = _parse_int_value(requested_slave_id)
            if assigned_slave_id is None:
                assigned_slave_id = next_slave_id
            next_slave_id = assigned_slave_id + 1
            last_slave_id = assigned_slave_id
            env_values["ECMC_EC_SLAVE_NUM"] = str(assigned_slave_id)
            env_values["SLAVE_ID"] = str(next_slave_id)
            if hw_desc:
                env_values["HW_DESC"] = hw_desc

            config_target = None
            config_value = expanded_key_map.get("CONFIG", "")
            if config_value:
                config_target = _resolve_reference(config_value, base_dir)
                if config_target is None and hw_desc and "${" not in hw_desc and "$(" not in hw_desc:
                    config_name = "ecmc{}{}.cmd".format(hw_desc, config_value)
                    config_matches = inventory.hardware_configs.get(config_name, [])
                    if config_matches:
                        config_target = config_matches[0]
            summary = "SLAVE_ID={}, HW_DESC={}".format(assigned_slave_id, hw_desc or "<unset>")
            detail_rows = _format_tree_details(
                expanded_key_map,
                ["SLAVE_ID", "HW_DESC", "CONFIG", "LOCAL_CONFIG", "MACROS", "SUBST_FILE"],
            )
            command_rows = _format_command_macro_details(
                module_script_name,
                expanded_key_map,
                inventory,
                ["SLAVE_ID", "HW_DESC", "CONFIG", "LOCAL_CONFIG", "MACROS", "SUBST_FILE"],
            )
            objects.append(
                StartupObject(
                    kind="slave",
                    source=path,
                    line=line_no,
                    title="Slave {} {}".format(assigned_slave_id, hw_desc).strip(),
                    summary=summary,
                    slave_id=assigned_slave_id,
                    linked_file=config_target,
                    details=detail_rows,
                    command_details=command_rows,
                )
            )

        if module_script_name == "applySlaveConfig.cmd":
            parent_slave_id = _parse_int_value(
                expanded_key_map.get("SLAVE_ID", env_values.get("ECMC_EC_SLAVE_NUM", ""))
            )
            if parent_slave_id is None:
                parent_slave_id = last_slave_id
            if hw_desc:
                env_values["HW_DESC"] = hw_desc
            config_target = None
            local_config = expanded_key_map.get("LOCAL_CONFIG", "")
            config_value = expanded_key_map.get("CONFIG", "")
            if local_config:
                config_target = _resolve_reference(local_config, base_dir)
            elif config_value:
                config_target = _resolve_reference(config_value, base_dir)
                if config_target is None and hw_desc and "${" not in hw_desc and "$(" not in hw_desc:
                    config_name = "ecmc{}{}.cmd".format(hw_desc, config_value)
                    config_matches = inventory.hardware_configs.get(config_name, [])
                    if config_matches:
                        config_target = config_matches[0]
            summary_target = local_config or config_value or "<unset>"
            detail_rows = _format_tree_details(
                expanded_key_map,
                ["CONFIG", "LOCAL_CONFIG", "SLAVE_ID", "HW_DESC"],
            )
            command_rows = _format_command_macro_details(
                module_script_name,
                expanded_key_map,
                inventory,
                ["CONFIG", "LOCAL_CONFIG", "SLAVE_ID", "HW_DESC"],
            )
            objects.append(
                StartupObject(
                    kind="slave_config",
                    source=path,
                    line=line_no,
                    title="Slave Config {}".format(Path(summary_target).name if local_config else summary_target).strip(),
                    summary="CONFIG={}".format(summary_target),
                    parent_slave_id=parent_slave_id,
                    linked_file=config_target,
                    details=detail_rows,
                    command_details=command_rows,
                )
            )

        if module_script_name and _is_yaml_loader_script(module_script_name):
            file_value = expanded_key_map.get("FILE", "")
            yaml_target = _resolve_reference(file_value, base_dir) if file_value else None
            yaml_macro_usage = None
            yaml_scope: Dict[str, str] = {}
            if yaml_target is not None:
                yaml_text = _read_text_from_buffers(yaml_target, buffer_lookup)
                if yaml_text is not None:
                    yaml_macro_usage = _scan_file_macro_usage(yaml_text)
                    yaml_scope = dict(env_values)
                    yaml_scope.update(expanded_key_map)
            schema_kind = _yaml_loader_schema_kind(module_script_name) or "yaml"
            title_name = Path(file_value).name if file_value else module_script_name
            if schema_kind == "axis":
                axis_id = expanded_key_map.get("AXIS_ID", "?")
                axis_name = expanded_key_map.get("AX_NAME", title_name)
                title_name = "Axis {}: {}".format(axis_id, axis_name).strip()
            elif schema_kind == "encoder":
                title_name = "Encoder {}".format(title_name)
            elif schema_kind == "plc":
                title_name = "PLC {}".format(title_name)
            summary = "FILE={}".format(file_value or "<unset>")
            if "AX_NAME" in expanded_key_map:
                summary = "AX_NAME={}, {}".format(expanded_key_map["AX_NAME"], summary)
            elif "AXIS_ID" in expanded_key_map:
                summary = "AXIS_ID={}, {}".format(expanded_key_map["AXIS_ID"], summary)
            detail_rows = _format_tree_details(
                expanded_key_map,
                ["FILE", "AX_NAME", "AXIS_ID", "DRV_SID", "ENC_SID", "DRV_CH", "ENC_CH", "DEV", "PREFIX"],
            )
            command_rows = _format_command_macro_details(
                module_script_name,
                expanded_key_map,
                inventory,
                ["FILE", "DEV", "PREFIX", "AXIS_ID", "DRV_SID", "ENC_SID", "DRV_CH", "ENC_CH", "AX_NAME"],
            )
            linked_file_rows = _format_applied_macro_details(
                yaml_scope,
                yaml_macro_usage,
                ["AX_NAME", "AXIS_ID", "DRV_SID", "ENC_SID", "DRV_CH", "ENC_CH", "DEV", "PREFIX"],
            )
            objects.append(
                StartupObject(
                    kind=schema_kind,
                    source=path,
                    line=line_no,
                    title=title_name,
                    summary=summary,
                    parent_axis_line=last_axis_line if schema_kind == "encoder" else None,
                    linked_file=yaml_target,
                    details=detail_rows,
                    command_details=command_rows,
                    linked_file_details=linked_file_rows,
                )
            )
            if schema_kind == "axis":
                last_axis_line = line_no

        if module_script_name == "configureAxis.cmd":
            config_value = expanded_key_map.get("CONFIG", "")
            config_target = _resolve_reference(config_value, base_dir) if config_value else None
            detail_rows = _format_tree_details(
                expanded_key_map,
                ["CONFIG", "DEV", "CLEAR_VARS_CMD", "CFG_MACROS"],
            )
            command_rows = _format_command_macro_details(
                module_script_name,
                expanded_key_map,
                inventory,
                ["CONFIG", "DEV", "CLEAR_VARS_CMD", "CFG_MACROS"],
            )
            axis_name = Path(config_value).name if config_value else "axis"
            summary = "CONFIG={}".format(config_value or "<unset>")
            if expanded_key_map.get("DEV"):
                summary = "DEV={}, {}".format(expanded_key_map["DEV"], summary)
            objects.append(
                StartupObject(
                    kind="configured_axis",
                    source=path,
                    line=line_no,
                    title="Configure Axis {}".format(axis_name),
                    summary=summary,
                    linked_file=config_target,
                    details=detail_rows,
                    command_details=command_rows,
                )
            )

        if module_script_name == "loadPLCFile.cmd":
            file_value = expanded_key_map.get("FILE", "")
            plc_target = _resolve_reference(file_value, base_dir) if file_value else None
            plc_id = _normalize_value(expanded_key_map.get("PLC_ID", env_values.get("ECMC_PLC_ID", "0"))) or "0"
            parsed_plc_id = _parse_int_value(plc_id)
            if parsed_plc_id is not None:
                last_plc_id = parsed_plc_id
                env_values["ECMC_PLC_ID"] = str(parsed_plc_id)
            plc_scope: Dict[str, str] = {}
            plc_macro_usage = None
            plc_payload = expanded_key_map.get("PLC_MACROS", "")
            plc_macro_pairs: List[Tuple[str, str]] = []
            if plc_payload:
                plc_macro_pairs, _malformed_plc_macros = _parse_macro_payload(plc_payload)
            if plc_target is not None:
                plc_text = _read_text_from_buffers(plc_target, buffer_lookup)
                if plc_text is not None:
                    plc_scope = dict(env_values)
                    plc_scope.update({key: value for key, value in plc_macro_pairs})
                    plc_scope["SELF_ID"] = plc_id
                    plc_scope["SELF"] = "plc{}".format(plc_id)
                    plc_scope["M_ID"] = str(current_master_id)
                    plc_scope["M"] = "ec{}".format(current_master_id)
                    plc_macro_usage = _scan_file_macro_usage(plc_text)
            summary = "PLC_ID={}, FILE={}".format(plc_id, file_value or "<unset>")
            detail_rows = _format_tree_details(
                expanded_key_map,
                ["FILE", "PLC_ID", "SAMPLE_RATE_MS", "PLC_MACROS", "INC", "DESC"],
            )
            command_rows = _format_command_macro_details(
                module_script_name,
                expanded_key_map,
                inventory,
                ["FILE", "PLC_ID", "SAMPLE_RATE_MS", "PLC_MACROS", "INC", "DESC"],
            )
            linked_file_rows = _format_applied_macro_details(
                plc_scope,
                plc_macro_usage,
                ["SELF_ID", "SELF", "M_ID", "M"] + [key for key, _value in plc_macro_pairs],
            )
            objects.append(
                StartupObject(
                    kind="plc",
                    source=path,
                    line=line_no,
                    title="PLC {}".format(plc_id),
                    summary=summary,
                    parent_plc_id=parsed_plc_id,
                    linked_file=plc_target,
                    details=detail_rows,
                    command_details=command_rows,
                    linked_file_details=linked_file_rows,
                )
            )

        dbload_call = _extract_dbloadrecords_call(line)
        if dbload_call is not None:
            db_file, db_macros = dbload_call
            expanded_db_macros = {key: _expand_text_macros(value, env_values) for key, value in db_macros.items()}
            plc_var_kind = ""
            if db_file == "ecmcPlcAnalog.db":
                plc_var_kind = "plcvar_analog"
            elif db_file == "ecmcPlcBinary.db":
                plc_var_kind = "plcvar_binary"
            if plc_var_kind:
                asyn_name = expanded_db_macros.get("ASYN_NAME", "")
                rec_name = expanded_db_macros.get("REC_NAME", "") or expanded_db_macros.get("NAME", "")
                plc_id = last_plc_id
                plc_match = re.search(r"\bplc(?:s\.)?plc(\d+)\.", asyn_name)
                if plc_match:
                    plc_id = _parse_int_value(plc_match.group(1))
                summary = "ASYN_NAME={}".format(asyn_name or "<unset>")
                if rec_name:
                    summary = "REC_NAME={}, {}".format(rec_name, summary)
                objects.append(
                    StartupObject(
                        kind=plc_var_kind,
                        source=path,
                        line=line_no,
                        title=("PLC Analog " if plc_var_kind == "plcvar_analog" else "PLC Binary ")
                        + (rec_name or asyn_name or "variable"),
                        summary=summary,
                        parent_plc_id=plc_id,
                    details=_format_tree_details(
                            expanded_db_macros,
                            ["P", "PORT", "ASYN_NAME", "REC_NAME", "TSE", "T_SMP_MS"],
                        ),
                    )
                )

        if module_script_name == "addMasterSlaveSM.cmd":
            sm_id = _normalize_value(expanded_key_map.get("SM_ID", env_values.get("SM_ID", "0"))) or "0"
            parsed_sm_id = _parse_int_value(sm_id)
            if parsed_sm_id is not None:
                env_values["ECMC_SM_ID"] = str(parsed_sm_id)
                env_values["SM_ID"] = str(parsed_sm_id + 1)
            detail_rows = _format_tree_details(
                expanded_key_map,
                ["NAME", "MST_GRP_NAME", "SLV_GRP_NAME", "MST_DISABLE", "SLV_DISABLE", "SM_ID"],
            )
            command_rows = _format_command_macro_details(
                module_script_name,
                expanded_key_map,
                inventory,
                ["NAME", "MST_GRP_NAME", "SLV_GRP_NAME", "MST_DISABLE", "SLV_DISABLE", "SM_ID"],
            )
            objects.append(
                StartupObject(
                    kind="master_slave_sm",
                    source=path,
                    line=line_no,
                    title="Master/Slave SM {}".format(expanded_key_map.get("NAME", sm_id or "unnamed")),
                    summary="NAME={}, MST_GRP_NAME={}, SLV_GRP_NAME={}".format(
                        expanded_key_map.get("NAME", "<unset>"),
                        expanded_key_map.get("MST_GRP_NAME", "<unset>"),
                        expanded_key_map.get("SLV_GRP_NAME", "<unset>"),
                    ),
                    details=detail_rows,
                    command_details=command_rows,
                )
            )

        if module_script_name == "loadSubstConfig.cmd":
            file_value = expanded_key_map.get("FILE", "")
            subst_target = _resolve_reference(file_value, base_dir) if file_value else None
            detail_rows = _format_tree_details(expanded_key_map, ["FILE", "MACROS"])
            command_rows = _format_command_macro_details(
                module_script_name,
                expanded_key_map,
                inventory,
                ["FILE", "MACROS"],
            )
            objects.append(
                StartupObject(
                    kind="subst_config",
                    source=path,
                    line=line_no,
                    title="Subst Config {}".format(Path(file_value).name if file_value else "file"),
                    summary="FILE={}".format(file_value or "<unset>"),
                    linked_file=subst_target,
                    details=detail_rows,
                    command_details=command_rows,
                )
            )

        if module_script_name == "pvtControllerConfig.cmd":
            detail_rows = _format_tree_details(
                expanded_key_map,
                ["TRG_EC_ENTRY", "TRG_DUR_S", "NAXES", "NPOINTS", "NREADBACK", "NPULSES", "MAX_SIZE", "SOFT_TRG_FLNK"],
            )
            command_rows = _format_command_macro_details(
                module_script_name,
                expanded_key_map,
                inventory,
                ["TRG_EC_ENTRY", "TRG_DUR_S", "NAXES", "NPOINTS", "NREADBACK", "NPULSES", "MAX_SIZE", "SOFT_TRG_FLNK"],
            )
            summary_parts: List[str] = []
            if expanded_key_map.get("TRG_EC_ENTRY"):
                summary_parts.append("TRG_EC_ENTRY={}".format(expanded_key_map["TRG_EC_ENTRY"]))
            if expanded_key_map.get("NAXES"):
                summary_parts.append("NAXES={}".format(expanded_key_map["NAXES"]))
            objects.append(
                StartupObject(
                    kind="pvt_controller",
                    source=path,
                    line=line_no,
                    title="PVT Controller",
                    summary=", ".join(summary_parts) if summary_parts else "Profile move controller",
                    details=detail_rows,
                    command_details=command_rows,
                )
            )

        if module_script_name == "loadPlugin.cmd":
            file_value = expanded_key_map.get("FILE", "")
            plugin_target = _resolve_reference(file_value, base_dir) if file_value else None
            plugin_id = _normalize_value(expanded_key_map.get("PLUGIN_ID", env_values.get("PLUGIN_ID", "0"))) or "0"
            summary = "PLUGIN_ID={}, FILE={}".format(plugin_id, file_value or "<unset>")
            detail_rows = _format_tree_details(expanded_key_map, ["FILE", "PLUGIN_ID", "CONFIG", "REPORT"])
            command_rows = _format_command_macro_details(
                module_script_name,
                expanded_key_map,
                inventory,
                ["FILE", "PLUGIN_ID", "CONFIG", "REPORT"],
            )
            objects.append(
                StartupObject(
                    kind="plugin",
                    source=path,
                    line=line_no,
                    title="Plugin {}".format(plugin_id),
                    summary=summary,
                    linked_file=plugin_target,
                    details=detail_rows,
                    command_details=command_rows,
                )
            )

        if module_script_name == "loadLUTFile.cmd":
            file_value = expanded_key_map.get("FILE", "")
            lut_target = _resolve_reference(file_value, base_dir) if file_value else None
            lut_id = _normalize_value(expanded_key_map.get("LUT_ID", env_values.get("LUT_ID", "0"))) or "0"
            parsed_lut_id = _parse_int_value(lut_id)
            if parsed_lut_id is not None:
                env_values["ECMC_LUT_ID"] = str(parsed_lut_id)
                env_values["LUT_ID"] = str(parsed_lut_id + 1)
            detail_rows = _format_tree_details(expanded_key_map, ["FILE", "LUT_ID"])
            command_rows = _format_command_macro_details(
                module_script_name,
                expanded_key_map,
                inventory,
                ["FILE", "LUT_ID"],
            )
            objects.append(
                StartupObject(
                    kind="lut",
                    source=path,
                    line=line_no,
                    title="LUT {}".format(lut_id),
                    summary="LUT_ID={}, FILE={}".format(lut_id, file_value or "<unset>"),
                    linked_file=lut_target,
                    details=detail_rows,
                    command_details=command_rows,
                )
            )

        if module_script_name == "addDataStorage.cmd":
            ds_id = _normalize_value(expanded_key_map.get("DS_ID", env_values.get("DS_ID", "0"))) or "0"
            ds_size = expanded_key_map.get("DS_SIZE", "")
            summary = "DS_ID={}, DS_SIZE={}".format(ds_id, ds_size or "<unset>")
            detail_rows = _format_tree_details(
                expanded_key_map,
                ["DS_SIZE", "DS_ID", "DS_TYPE", "SAMPLE_RATE_MS", "DS_DEBUG", "DESC"],
            )
            command_rows = _format_command_macro_details(
                module_script_name,
                expanded_key_map,
                inventory,
                ["DS_SIZE", "DS_ID", "DS_TYPE", "SAMPLE_RATE_MS", "DS_DEBUG", "DESC"],
            )
            objects.append(
                StartupObject(
                    kind="datastorage",
                    source=path,
                    line=line_no,
                    title="DataStorage {}".format(ds_id),
                    summary=summary,
                    details=detail_rows,
                    command_details=command_rows,
                )
            )

        if module_script_name == "addEcSdoRT.cmd":
            slave_id = _parse_int_value(expanded_key_map.get("SLAVE_ID", env_values.get("ECMC_EC_SLAVE_NUM", "")))
            title_name = expanded_key_map.get("NAME", "") or "unnamed"
            summary = "SLAVE_ID={}, INDEX={}, SUBINDEX={}".format(
                slave_id if slave_id is not None else "?",
                expanded_key_map.get("INDEX", ""),
                expanded_key_map.get("SUBINDEX", ""),
            )
            detail_rows = _format_tree_details(
                expanded_key_map,
                ["SLAVE_ID", "INDEX", "SUBINDEX", "DT", "NAME", "P_SCRIPT"],
            )
            command_rows = _format_command_macro_details(
                module_script_name,
                expanded_key_map,
                inventory,
                ["SLAVE_ID", "INDEX", "SUBINDEX", "DT", "NAME", "P_SCRIPT"],
            )
            objects.append(
                StartupObject(
                    kind="ecsdo",
                    source=path,
                    line=line_no,
                    title="EcSdo {}".format(title_name),
                    summary=summary,
                    parent_slave_id=slave_id,
                    details=detail_rows,
                    command_details=command_rows,
                )
            )

        if module_script_name == "addEcDataItem.cmd":
            slave_id = _parse_int_value(
                expanded_key_map.get("STRT_ENTRY_S_ID", env_values.get("ECMC_EC_SLAVE_NUM", ""))
            )
            title_name = expanded_key_map.get("NAME", "") or "unnamed"
            summary = "SLAVE_ID={}, ENTRY={}".format(
                slave_id if slave_id is not None else "?",
                expanded_key_map.get("STRT_ENTRY_NAME", ""),
            )
            detail_rows = _format_tree_details(
                expanded_key_map,
                [
                    "STRT_ENTRY_S_ID",
                    "STRT_ENTRY_NAME",
                    "OFFSET_BYTE",
                    "OFFSET_BITS",
                    "RW",
                    "DT",
                    "NAME",
                    "P_SCRIPT",
                    "REC_FIELDS",
                    "REC_TYPE",
                    "INIT_VAL",
                    "LOAD_RECS",
                ],
            )
            command_rows = _format_command_macro_details(
                module_script_name,
                expanded_key_map,
                inventory,
                [
                    "STRT_ENTRY_S_ID",
                    "STRT_ENTRY_NAME",
                    "OFFSET_BYTE",
                    "OFFSET_BITS",
                    "RW",
                    "DT",
                    "NAME",
                    "P_SCRIPT",
                    "REC_FIELDS",
                    "REC_TYPE",
                    "INIT_VAL",
                    "LOAD_RECS",
                ],
            )
            objects.append(
                StartupObject(
                    kind="ecdataitem",
                    source=path,
                    line=line_no,
                    title="EcDataItem {}".format(title_name),
                    summary=summary,
                    parent_slave_id=slave_id,
                    details=detail_rows,
                    command_details=command_rows,
                )
            )

        if module_script_name == "applyComponent.cmd":
            component_name = expanded_key_map.get("COMP", "") or expanded_key_map.get("EC_COMP_TYPE", "")
            parent_slave_id = _parse_int_value(
                expanded_key_map.get("COMP_S_ID", env_values.get("ECMC_EC_SLAVE_NUM", ""))
            )
            if parent_slave_id is None:
                parent_slave_id = last_slave_id
            channel_id = expanded_key_map.get("CH_ID", "1") or "1"
            summary_parts = []
            if component_name:
                summary_parts.append("COMP={}".format(component_name))
            if parent_slave_id is not None:
                summary_parts.append("SLAVE_ID={}".format(parent_slave_id))
            summary_parts.append("CH_ID={}".format(channel_id))
            detail_rows = _format_tree_details(
                expanded_key_map,
                ["COMP", "EC_COMP_TYPE", "COMP_S_ID", "CH_ID", "MACROS"],
            )
            command_rows = _format_command_macro_details(
                module_script_name,
                expanded_key_map,
                inventory,
                ["COMP", "EC_COMP_TYPE", "COMP_S_ID", "CH_ID", "MACROS"],
            )
            objects.append(
                StartupObject(
                    kind="component",
                    source=path,
                    line=line_no,
                    title="Component {}".format(component_name or "unnamed"),
                    summary=", ".join(summary_parts),
                    parent_slave_id=parent_slave_id,
                    details=detail_rows,
                    command_details=command_rows,
                )
            )

        resolved_command = _resolve_reference(_expand_text_macros(command_target, env_values), base_dir)
        if (
            resolved_command is not None
            and _is_project_script(resolved_command)
            and not module_script_name
            and not _is_internal_helper_script(resolved_command, inventory)
        ):
            nested_scripts.append((resolved_command, line_no))

    return objects, nested_scripts


def build_startup_tree(
    startup_path: Path,
    startup_text: str,
    inventory: RepositoryInventory,
    buffer_lookup: Optional[Dict[Path, str]] = None,
) -> StartupTreeModel:
    files: List[StartupFileNode] = []
    queued: List[Tuple[Optional[Path], Path, int]] = [(None, startup_path.resolve(), 1)]
    seen: Set[Path] = set()
    buffer_lookup = buffer_lookup or {}

    while queued:
        parent_path, current, parent_line = queued.pop(0)
        if current in seen:
            continue
        seen.add(current)

        if current == startup_path.resolve():
            text = startup_text
        elif current in buffer_lookup:
            text = buffer_lookup[current]
        elif current.exists():
            text = _read_text(current)
        else:
            text = ""

        objects, nested_scripts = _extract_startup_objects_from_file(current, text, inventory, buffer_lookup)
        files.append(StartupFileNode(path=current, parent_path=parent_path, parent_line=parent_line, objects=objects))
        for nested_path, line_no in nested_scripts:
            resolved_nested = nested_path.resolve()
            if resolved_nested not in seen:
                queued.append((current, resolved_nested, line_no))

    return StartupTreeModel(files=files)


def _project_macro_map_from_tree(startup_tree: StartupTreeModel) -> Dict[str, str]:
    macro_map: Dict[str, str] = {}
    for file_node in startup_tree.files:
        for obj in file_node.objects:
            if obj.kind == "macro":
                detail_map = dict(obj.details)
                name = detail_map.get("NAME", "").strip()
                value = detail_map.get("VALUE", "").strip()
                if name and value:
                    macro_map[name] = value
            elif obj.kind == "require":
                for key, value in obj.linked_file_details:
                    if key and value:
                        macro_map[key] = value
    return macro_map


def _validate_axis_identity_uniqueness(
    startup_tree: StartupTreeModel,
    inventory: RepositoryInventory,
    buffer_lookup: Optional[Dict[Path, str]] = None,
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    axis_id_seen: Dict[str, Tuple[StartupObject, Path]] = {}
    epics_name_seen: Dict[str, Tuple[StartupObject, Path]] = {}
    global_macros = _project_macro_map_from_tree(startup_tree)

    for file_node in startup_tree.files:
        for obj in file_node.objects:
            if obj.kind != "axis" or obj.linked_file is None:
                continue
            yaml_text = _read_text_from_buffers(obj.linked_file, buffer_lookup)
            if yaml_text is None:
                continue

            macro_scope = dict(global_macros)
            for key, value in obj.command_details:
                macro_scope[key] = value
            for key, value in obj.linked_file_details:
                macro_scope[key] = value

            expanded_yaml = _expand_text_macros(yaml_text, macro_scope)
            parsed_entries = _parse_simple_yaml_paths(expanded_yaml)
            value_by_path = {entry.path: entry.value for entry in parsed_entries if entry.value is not None}
            axis_id = _normalize_value(value_by_path.get("axis.id", "") or "")
            epics_name = _normalize_value(value_by_path.get("epics.name", "") or "")

            if axis_id:
                previous = axis_id_seen.get(axis_id)
                if previous is not None:
                    previous_obj, previous_target = previous
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            source=obj.source,
                            line=obj.line,
                            message=(
                                "Duplicate axis.id '{}' for '{}' and '{}'"
                            ).format(axis_id, previous_obj.title, obj.title),
                            target=obj.linked_file,
                        )
                    )
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            source=previous_obj.source,
                            line=previous_obj.line,
                            message=(
                                "Duplicate axis.id '{}' for '{}' and '{}'"
                            ).format(axis_id, previous_obj.title, obj.title),
                            target=previous_target,
                        )
                    )
                else:
                    axis_id_seen[axis_id] = (obj, obj.linked_file)

            if epics_name:
                previous = epics_name_seen.get(epics_name)
                if previous is not None:
                    previous_obj, previous_target = previous
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            source=obj.source,
                            line=obj.line,
                            message=(
                                "Duplicate epics.name '{}' for '{}' and '{}'"
                            ).format(epics_name, previous_obj.title, obj.title),
                            target=obj.linked_file,
                        )
                    )
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            source=previous_obj.source,
                            line=previous_obj.line,
                            message=(
                                "Duplicate epics.name '{}' for '{}' and '{}'"
                            ).format(epics_name, previous_obj.title, obj.title),
                            target=previous_target,
                        )
                    )
                else:
                    epics_name_seen[epics_name] = (obj, obj.linked_file)

    return issues


def _validate_extended_startup_object_relationships(
    startup_tree: StartupTreeModel,
    inventory: RepositoryInventory,
    buffer_lookup: Optional[Dict[Path, str]] = None,
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    global_macros = _project_macro_map_from_tree(startup_tree)
    axis_groups_by_source: Dict[Path, Set[str]] = {}
    pvt_axis_count_by_source: Dict[Path, int] = {}

    for file_node in startup_tree.files:
        source_path = file_node.path.resolve()
        axis_groups_by_source.setdefault(source_path, set())
        pvt_axis_count_by_source.setdefault(source_path, 0)
        for obj in file_node.objects:
            if obj.kind != "axis" or obj.linked_file is None:
                continue
            yaml_text = _read_text_from_buffers(obj.linked_file, buffer_lookup)
            if yaml_text is None:
                continue
            macro_scope = dict(global_macros)
            for key, value in obj.command_details:
                macro_scope[key] = value
            for key, value in obj.linked_file_details:
                macro_scope[key] = value
            expanded_yaml = _expand_text_macros(yaml_text, macro_scope)
            parsed_entries = _parse_simple_yaml_paths(expanded_yaml)
            value_by_path = {entry.path: entry.value for entry in parsed_entries if entry.value is not None}
            axis_group = _normalize_value(value_by_path.get("axis.group", "") or "")
            if axis_group:
                axis_groups_by_source[source_path].add(axis_group)
            if any(
                entry.path in {"axis.pvt", "epics.motorRecord.pvt"}
                or entry.path.startswith("axis.pvt.")
                or entry.path.startswith("epics.motorRecord.pvt.")
                for entry in parsed_entries
            ):
                pvt_axis_count_by_source[source_path] += 1

    for file_node in startup_tree.files:
        source_path = file_node.path.resolve()
        known_groups = axis_groups_by_source.get(source_path, set())
        pvt_axis_count = pvt_axis_count_by_source.get(source_path, 0)
        for obj in file_node.objects:
            detail_map = dict(obj.details)
            if obj.kind == "master_slave_sm":
                master_group = detail_map.get("MST_GRP_NAME", "").strip()
                slave_group = detail_map.get("SLV_GRP_NAME", "").strip()
                if not known_groups:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            source=obj.source,
                            line=obj.line,
                            message=(
                                "Cannot verify master/slave groups for '{}' because no axis.group values were found "
                                "in axis YAML files loaded from this startup file"
                            ).format(obj.title),
                        )
                    )
                    continue
                if master_group and master_group not in known_groups:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            source=obj.source,
                            line=obj.line,
                            message="MST_GRP_NAME '{}' was not found in any loaded axis.group value".format(master_group),
                        )
                    )
                if slave_group and slave_group not in known_groups:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            source=obj.source,
                            line=obj.line,
                            message="SLV_GRP_NAME '{}' was not found in any loaded axis.group value".format(slave_group),
                        )
                    )
            elif obj.kind == "pvt_controller" and pvt_axis_count <= 0:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        source=obj.source,
                        line=obj.line,
                        message=(
                            "PVT controller is configured but no loaded axis YAML in this startup file declares PVT support"
                        ),
                    )
                )

    return issues


def validate_project(
    startup_path: Path,
    startup_text: str,
    inventory: RepositoryInventory,
    buffer_lookup: Optional[Dict[Path, str]] = None,
) -> ValidationResult:
    issues: List[ValidationIssue] = []
    references: List[FileReference] = []
    visited: List[Path] = []
    queued: List[Path] = [startup_path.resolve()]
    seen: Set[Path] = set()
    buffer_lookup = buffer_lookup or {}

    while queued:
        current = queued.pop(0)
        if current in seen:
            continue
        seen.add(current)
        visited.append(current)

        if current == startup_path.resolve():
            text = startup_text
        elif current in buffer_lookup:
            text = buffer_lookup[current]
        elif current.exists():
            text = _read_text(current)
        else:
            issues.append(
                ValidationIssue(
                    severity="error",
                    source=current,
                    line=1,
                    message=f"Referenced file does not exist: {current}",
                    target=current,
                )
            )
            continue

        file_issues, file_refs, nested_scripts = _validate_single_file(current, text, inventory, buffer_lookup)
        issues.extend(file_issues)
        references.extend(file_refs)
        for nested in nested_scripts:
            resolved_nested = nested.resolve()
            if resolved_nested not in seen:
                queued.append(resolved_nested)

    startup_tree = build_startup_tree(
        startup_path=startup_path,
        startup_text=startup_text,
        inventory=inventory,
        buffer_lookup=buffer_lookup,
    )
    issues.extend(_validate_axis_identity_uniqueness(startup_tree, inventory, buffer_lookup))
    issues.extend(_validate_extended_startup_object_relationships(startup_tree, inventory, buffer_lookup))

    return ValidationResult(issues=issues, references=references, visited_files=visited)


class ValidatorApp:
    def __init__(self, root, initial_startup: Optional[Path] = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title("ecmc Engineering Studio")
        self.root.geometry("1480x900")

        self.ecmccfg_root = _find_ecmccfg_root(initial_startup)
        self.inventory = _build_repository_inventory(self.ecmccfg_root)

        self.startup_var = tk.StringVar(value=str(initial_startup) if initial_startup else "")
        self.editor_file_var = tk.StringVar(value="(no file selected)")
        self.editor_search_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")
        self.editor_auto_complete_var = tk.BooleanVar(value=True)
        self.editor_macro_hover_var = tk.BooleanVar(value=True)
        self.tree_view_mode_var = tk.StringVar(value="Flow")
        self.tree_filter_var = tk.StringVar(value="All")
        self.tree_sort_var = tk.StringVar(value="Flow")
        self.tree_search_var = tk.StringVar(value="")
        self.tree_compact_var = tk.BooleanVar(value=False)
        self.tree_overview_var = tk.StringVar(value="No objects loaded")
        self.tree_view_toggle_var = tk.StringVar(value="Show Objects")
        self.tree_error_var = tk.StringVar(value="0 errors")
        self.tree_warning_var = tk.StringVar(value="0 warnings")
        self.tree_unsaved_var = tk.StringVar(value="0 unsaved")
        self.tree_missing_var = tk.StringVar(value="0 missing")
        self.inventory_var = tk.StringVar(
            value=(
                f"{len(self.inventory.module_scripts)} scripts  •  "
                f"{len(self.inventory.hardware_descs)} HW_DESC"
            )
        )

        self.file_buffers: Dict[Path, str] = {}
        self.current_edit_path: Optional[Path] = None
        self.latest_result: Optional[ValidationResult] = None
        self.latest_startup_tree: Optional[StartupTreeModel] = None
        self.startup_item_map: Dict[str, Tuple[str, object]] = {}
        self.validation_issue_tree = None
        self.validation_summary_var = tk.StringVar(value="No validation results yet.")
        self.issue_item_map: Dict[str, ValidationIssue] = {}
        self.param_notebook = None
        self.selection_header_var = tk.StringVar(value="No object selected")
        self.object_action_hint_var = tk.StringVar(value="Select an object to edit, open a linked file, or add a macro.")
        self.help_summary_var = tk.StringVar(value="Select an object to see targeted problems and suggestions.")
        self.context_tree = None
        self.context_open_button = None
        self.context_edit_button = None
        self.context_macro_button = None
        self.help_issue_tree = None
        self.help_issue_item_map: Dict[str, ValidationIssue] = {}
        self.help_issue_action_map: Dict[str, Tuple[str, object]] = {}
        self.help_suggestion_list = None
        self.help_suggestion_actions: List[Optional[str]] = []
        self.help_open_button = None
        self.help_edit_button = None
        self.help_macro_button = None
        self.help_validate_button = None
        self.quick_edit_key_var = tk.StringVar(value="")
        self.quick_edit_value_var = tk.StringVar(value="")
        self.quick_edit_hint_var = tk.StringVar(value="Select an object field to edit directly.")
        self.quick_edit_key_combo = None
        self.quick_edit_value_entry = None
        self.quick_edit_apply_button = None
        self.quick_edit_open_button = None
        self.quick_edit_frame = None
        self.quick_edit_fields: List[str] = []
        self.resolved_text = None
        self.object_action_frame = None
        self.edit_object_button = None
        self.add_macro_button = None
        self.open_object_file_button = None
        self.copy_object_button = None
        self.paste_object_button = None
        self.delete_object_button = None
        self.move_up_button = None
        self.move_down_button = None
        self.file_browser_tree = None
        self.file_browser_item_paths: Dict[str, Path] = {}
        self.file_browser_path_items: Dict[Path, str] = {}
        self.file_browser_loaded_dirs: Set[Path] = set()
        self.file_browser_root: Optional[Path] = None
        self.file_browser_root_item: Optional[str] = None
        self.tree_error_button = None
        self.tree_warning_button = None
        self.tree_unsaved_button = None
        self.tree_missing_button = None
        self.tree_view_toggle_button = None
        self.selection_kind_badge = None
        self.selection_state_badge = None
        self.selection_source_badge = None
        self.help_state_badge = None
        self.help_summary_label = None
        self.editor_scrollbar = None
        self._editor_target_line_clear_job = None
        self._editor_update_job = None
        self._editor_tree_sync_job = None
        self._editor_mouse_interacting_until = 0.0
        self._search_update_job = None
        self._tree_filter_update_job = None
        self.editor_search_entry = None
        self.tree_search_entry = None
        self._editor_search_placeholder_visible = False
        self._tree_search_placeholder_visible = False
        self.object_tree_items: Dict[Tuple[Path, int, str, str], str] = {}
        self.linked_file_tree_items: Dict[Path, List[str]] = {}
        self.linked_file_item_by_object_key: Dict[Tuple[Path, int, str, str], str] = {}
        self.file_tree_items: Dict[Path, str] = {}
        self._editor_tree_highlight_item: Optional[str] = None
        self._suppress_tree_open_on_select = False
        self._last_editor_sync_location: Optional[Tuple[Path, int]] = None
        self.current_edit_linked_object_key: Optional[Tuple[Path, int, str, str]] = None
        self.editor_completion_popup = None
        self.editor_completion_listbox = None
        self.editor_completion_label_var = tk.StringVar(value="")
        self.editor_completion_candidates: List[str] = []
        self.editor_completion_range: Optional[Tuple[str, str]] = None
        self._editor_completion_should_open = False
        self._editor_completion_forced = False
        self.editor_macro_tooltip = None
        self.editor_macro_tooltip_label = None
        self._editor_macro_hover_job = None
        self._editor_macro_hover_token = None
        self._editor_macro_hover_event = None
        self.log_pane = None
        self.log_frame = None
        self.log_toggle_button = None
        self.log_text = None
        self.log_visible = False
        self._last_logged_status = ""
        self.copied_object_text: str = ""
        self.copied_object_kind: str = ""
        self.copied_object_top_level: bool = False

        self._configure_ui_style()
        self._build_ui()
        self.tree_view_mode_var.trace_add("write", self._schedule_tree_view_update)
        self.tree_view_mode_var.trace_add("write", self._on_tree_view_mode_changed)
        self.tree_filter_var.trace_add("write", self._schedule_tree_view_update)
        self.tree_sort_var.trace_add("write", self._schedule_tree_view_update)
        self.tree_search_var.trace_add("write", self._schedule_tree_view_update)
        self.tree_compact_var.trace_add("write", self._schedule_tree_view_update)
        self._update_tree_view_toggle_text()
        self.status_var.trace_add("write", self._on_status_message_changed)
        self._append_log_message(self.status_var.get())

        if initial_startup is not None:
            self._open_startup(initial_startup.resolve(), validate_now=False)

    def _configure_ui_style(self) -> None:
        style = self.ttk.Style()
        try:
            if "clam" in style.theme_names():
                style.theme_use("clam")
        except Exception:
            pass

        self.root.configure(background="#f3efe7")
        style.configure(".", font=("Helvetica", 12))
        style.configure("Toolbar.TButton", padding=(8, 6))
        style.configure("ObjectAction.TButton", font=("Helvetica", 10), padding=(4, 2))
        style.configure("TreeChip.TButton", font=("Helvetica", 10, "bold"), padding=(8, 2), relief="flat")
        style.map("TreeChip.TButton", relief=[("pressed", "sunken")])
        style.configure("ErrorChip.TButton", background="#f8d7da", foreground="#8b1e2d")
        style.map(
            "ErrorChip.TButton",
            background=[("active", "#f3c2c8"), ("disabled", "#efe6e7")],
            foreground=[("disabled", "#a78a8f")],
        )
        style.configure("WarningChip.TButton", background="#fff0c2", foreground="#8a5a00")
        style.map(
            "WarningChip.TButton",
            background=[("active", "#fbe6a5"), ("disabled", "#f2ece0")],
            foreground=[("disabled", "#a99672")],
        )
        style.configure("UnsavedChip.TButton", background="#dcecff", foreground="#0b5cad")
        style.map(
            "UnsavedChip.TButton",
            background=[("active", "#c6e0ff"), ("disabled", "#e7edf4")],
            foreground=[("disabled", "#8ba0b5")],
        )
        style.configure("MissingChip.TButton", background="#eadcf7", foreground="#6e3ea3")
        style.map(
            "MissingChip.TButton",
            background=[("active", "#dec9f1"), ("disabled", "#efe8f4")],
            foreground=[("disabled", "#a294b3")],
        )
        style.configure("ViewToggle.TButton", font=("Helvetica", 10, "bold"), padding=(8, 2))
        style.configure("Section.TLabel", font=("Helvetica", 13, "bold"))
        style.configure("Muted.TLabel", foreground="#5f6a77")
        style.configure("HeaderNote.TLabel", foreground="#5f6a77", font=("Helvetica", 11))
        style.configure("Treeview", rowheight=26)
        style.configure("Treeview.Heading", font=("Helvetica", 11, "bold"))
        style.map(
            "Treeview",
            background=[("selected", "#cfe0f6")],
            foreground=[("selected", "#1d2a35")],
        )
        style.configure("TNotebook.Tab", padding=(10, 6))

    def _build_ui(self) -> None:
        tk = self.tk
        ttk = self.ttk

        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Startup", style="Section.TLabel").grid(row=0, column=0, sticky=tk.W)
        startup_entry = ttk.Entry(top, textvariable=self.startup_var, width=96)
        startup_entry.grid(row=0, column=1, sticky=tk.EW, padx=6)
        startup_entry.bind("<Return>", lambda _event: self._load_startup_from_entry())
        ttk.Button(top, text="Browse...", style="Toolbar.TButton", command=self._browse_startup).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(top, text="Open", style="Toolbar.TButton", command=self._load_startup_from_entry).grid(row=0, column=3, padx=(0, 6))
        ttk.Button(top, text="Save File", style="Toolbar.TButton", command=self._save_current_file).grid(row=0, column=4, padx=(0, 6))
        ttk.Button(top, text="Save All", style="Toolbar.TButton", command=self._save_all_files).grid(row=0, column=5, padx=(0, 6))
        ttk.Button(top, text="Refresh", style="Toolbar.TButton", command=self._refresh_startup_tree).grid(row=0, column=6, padx=(0, 6))
        ttk.Button(top, text="Validate", style="Toolbar.TButton", command=self._validate_current_project).grid(row=0, column=7)
        self.log_toggle_button = ttk.Button(top, text="Results", style="Toolbar.TButton", command=self._show_latest_results)
        self.log_toggle_button.grid(row=0, column=8, padx=(6, 0))
        ttk.Label(top, textvariable=self.inventory_var, style="HeaderNote.TLabel").grid(row=1, column=1, sticky=tk.W, padx=6, pady=(6, 0))
        top.columnconfigure(1, weight=1)

        center = ttk.Panedwindow(self.root, orient=tk.VERTICAL)
        center.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.log_pane = center

        body = ttk.Panedwindow(center, orient=tk.HORIZONTAL)
        center.add(body, weight=5)

        left = ttk.Panedwindow(body, orient=tk.VERTICAL)
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=5)

        object_frame = ttk.Frame(left)
        lower_left = ttk.Frame(left)
        left.add(object_frame, weight=3)
        left.add(lower_left, weight=2)

        object_header = ttk.Frame(object_frame)
        object_header.pack(fill=tk.X)
        ttk.Label(object_header, text="Objects", style="Section.TLabel").pack(side=tk.LEFT, anchor=tk.W)
        ttk.Label(object_header, textvariable=self.tree_overview_var, style="Muted.TLabel").pack(side=tk.RIGHT)
        overview_action_frame = ttk.Frame(object_frame)
        overview_action_frame.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(overview_action_frame, text="Status", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        self.tree_error_button = ttk.Button(
            overview_action_frame,
            textvariable=self.tree_error_var,
            style="ErrorChip.TButton",
            command=self._show_error_tree_items,
        )
        self.tree_error_button.pack(side=tk.LEFT)
        self.tree_warning_button = ttk.Button(
            overview_action_frame,
            textvariable=self.tree_warning_var,
            style="WarningChip.TButton",
            command=self._show_warning_tree_items,
        )
        self.tree_warning_button.pack(side=tk.LEFT, padx=(6, 0))
        self.tree_unsaved_button = ttk.Button(
            overview_action_frame,
            textvariable=self.tree_unsaved_var,
            style="UnsavedChip.TButton",
            command=self._show_unsaved_tree_items,
        )
        self.tree_unsaved_button.pack(side=tk.LEFT, padx=(6, 0))
        self.tree_missing_button = ttk.Button(
            overview_action_frame,
            textvariable=self.tree_missing_var,
            style="MissingChip.TButton",
            command=self._show_missing_tree_items,
        )
        self.tree_missing_button.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(
            overview_action_frame,
            text="Reset View",
            style="ObjectAction.TButton",
            command=self._reset_tree_view_controls,
        ).pack(side=tk.RIGHT)
        ttk.Button(
            overview_action_frame,
            text="Collapse All",
            style="ObjectAction.TButton",
            command=self._collapse_startup_tree,
        ).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(
            overview_action_frame,
            text="Expand Issues",
            style="ObjectAction.TButton",
            command=self._expand_issue_tree_nodes,
        ).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(
            overview_action_frame,
            text="Unsaved",
            style="ObjectAction.TButton",
            command=self._show_unsaved_tree_items,
        ).pack(side=tk.RIGHT, padx=(12, 0))
        ttk.Button(
            overview_action_frame,
            text="Issues",
            style="ObjectAction.TButton",
            command=self._show_issue_tree_items,
        ).pack(side=tk.RIGHT, padx=(6, 0))
        filter_frame = ttk.Frame(object_frame)
        filter_frame.pack(fill=tk.X, pady=(4, 4))
        ttk.Label(filter_frame, text="View", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        tree_view_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.tree_view_mode_var,
            values=("Flow", "Objects"),
            state="readonly",
            width=8,
        )
        tree_view_combo.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(filter_frame, text="Show", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        tree_filter_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.tree_filter_var,
            values=("All", "Slaves", "Axes", "PLCs", "Macros", "ECMC", "Errors", "Warnings", "Missing", "Issues", "Unsaved"),
            state="readonly",
            width=10,
        )
        tree_filter_combo.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(filter_frame, text="Sort", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        tree_sort_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.tree_sort_var,
            values=("Flow", "Name", "ID"),
            state="readonly",
            width=7,
        )
        tree_sort_combo.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(filter_frame, text="Search", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        tree_search_entry = tk.Entry(filter_frame, textvariable=self.tree_search_var, width=24, relief=tk.FLAT, highlightthickness=1)
        tree_search_entry.pack(side=tk.LEFT)
        tree_search_entry.configure(highlightbackground="#d0cabf", highlightcolor="#8fb7e1")
        tree_search_entry.bind("<Return>", self._focus_first_tree_match)
        tree_search_entry.bind("<FocusIn>", self._on_tree_search_focus_in)
        tree_search_entry.bind("<FocusOut>", self._on_tree_search_focus_out)
        self.tree_search_entry = tree_search_entry
        self._set_tree_search_placeholder()
        ttk.Button(filter_frame, text="Clear", style="ObjectAction.TButton", command=lambda: self.tree_search_var.set("")).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Checkbutton(filter_frame, text="Compact", variable=self.tree_compact_var).pack(side=tk.RIGHT)
        ttk.Label(
            object_frame,
            text="Order: top to bottom • ▣ file • ↗ linked file • ≡ macros • badges: ● unsaved  ! error  ? warning  × missing",
            style="Muted.TLabel",
        ).pack(
            anchor=tk.W, pady=(0, 6)
        )
        self.startup_tree = ttk.Treeview(
            object_frame,
            columns=("summary",),
            show="tree headings",
            selectmode="extended",
        )
        self.startup_tree.heading("#0", text="Object", anchor=tk.W)
        self.startup_tree.heading("summary", text="Summary", anchor=tk.W)
        self.startup_tree.column("#0", width=330, stretch=True, anchor=tk.W)
        self.startup_tree.column("summary", width=330, stretch=True, anchor=tk.W)
        self.startup_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        object_scroll = ttk.Scrollbar(object_frame, orient=tk.VERTICAL, command=self.startup_tree.yview)
        object_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.startup_tree.configure(yscrollcommand=object_scroll.set)
        self._configure_startup_tree_tags()
        self.startup_tree.bind("<<TreeviewSelect>>", self._on_startup_tree_selected)
        self.startup_tree.bind("<Double-1>", self._edit_selected_tree_entry)
        self.startup_tree.bind("<Return>", self._edit_selected_tree_entry)
        self.startup_tree.bind("<Button-2>", self._on_startup_tree_right_click)
        self.startup_tree.bind("<Button-3>", self._on_startup_tree_right_click)
        self.startup_tree.bind("<Control-Button-1>", self._on_startup_tree_right_click)
        self.startup_tree.bind("<Control-c>", self._on_tree_copy, add="+")
        self.startup_tree.bind("<Control-v>", self._on_tree_paste, add="+")
        self.startup_tree.bind("<Command-c>", self._on_tree_copy, add="+")
        self.startup_tree.bind("<Command-v>", self._on_tree_paste, add="+")
        self.root.bind_all("<Escape>", self._on_escape, add="+")

        self.startup_menu = tk.Menu(self.root, tearoff=0)

        self.object_action_frame = ttk.Frame(lower_left)
        self.object_action_frame.pack(fill=tk.X, pady=(0, 6))
        self.edit_object_button = ttk.Button(
            self.object_action_frame,
            text="Edit",
            command=self._edit_selected_tree_entry,
            style="ObjectAction.TButton",
        )
        self.edit_object_button.pack(side=tk.LEFT)
        self.add_macro_button = ttk.Button(
            self.object_action_frame,
            text="Macro",
            command=self._add_inline_macro,
            style="ObjectAction.TButton",
        )
        self.add_macro_button.pack(side=tk.LEFT, padx=(6, 0))
        self.open_object_file_button = ttk.Button(
            self.object_action_frame,
            text="Open",
            command=self._open_selected_object_file,
            style="ObjectAction.TButton",
        )
        self.open_object_file_button.pack(side=tk.LEFT, padx=(6, 0))
        self.move_up_button = ttk.Button(
            self.object_action_frame,
            text="↑",
            command=self._move_selected_object_up,
            style="ObjectAction.TButton",
        )
        self.move_up_button.pack(side=tk.LEFT, padx=(6, 0))
        self.move_down_button = ttk.Button(
            self.object_action_frame,
            text="↓",
            command=self._move_selected_object_down,
            style="ObjectAction.TButton",
        )
        self.move_down_button.pack(side=tk.LEFT, padx=(6, 0))
        self.copy_object_button = ttk.Button(
            self.object_action_frame,
            text="Copy",
            command=self._copy_selected_object,
            style="ObjectAction.TButton",
        )
        self.copy_object_button.pack(side=tk.LEFT, padx=(6, 0))
        self.paste_object_button = ttk.Button(
            self.object_action_frame,
            text="Paste",
            command=self._paste_copied_object,
            style="ObjectAction.TButton",
        )
        self.paste_object_button.pack(side=tk.LEFT, padx=(6, 0))
        self.delete_object_button = ttk.Button(
            self.object_action_frame,
            text="Remove",
            command=self._remove_selected_object,
            style="ObjectAction.TButton",
        )
        self.delete_object_button.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(lower_left, textvariable=self.object_action_hint_var, style="Muted.TLabel").pack(fill=tk.X, pady=(0, 6))

        self.param_notebook = ttk.Notebook(lower_left)
        self.param_notebook.pack(fill=tk.BOTH, expand=True)

        param_frame = ttk.Frame(self.param_notebook)
        self.param_notebook.add(param_frame, text="Details")
        ttk.Label(param_frame, textvariable=self.selection_header_var, style="Section.TLabel").pack(
            side=tk.TOP, anchor=tk.W, pady=(0, 6)
        )
        selection_badge_frame = ttk.Frame(param_frame)
        selection_badge_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
        self.selection_kind_badge = tk.Label(
            selection_badge_frame,
            text="Selection",
            padx=8,
            pady=2,
            background="#e8edf2",
            foreground="#425466",
        )
        self.selection_kind_badge.pack(side=tk.LEFT)
        self.selection_state_badge = tk.Label(
            selection_badge_frame,
            text="State",
            padx=8,
            pady=2,
            background="#eef3e8",
            foreground="#45603f",
        )
        self.selection_state_badge.pack(side=tk.LEFT, padx=(6, 0))
        self.selection_source_badge = tk.Label(
            selection_badge_frame,
            text="Source",
            padx=8,
            pady=2,
            background="#f2ede5",
            foreground="#6a5b4d",
        )
        self.selection_source_badge.pack(side=tk.LEFT, padx=(6, 0))
        param_tree_frame = ttk.Frame(param_frame)
        param_tree_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.param_tree = ttk.Treeview(param_tree_frame, columns=("value",), show="tree headings")
        self.param_tree.heading("#0", text="Parameter", anchor=tk.W)
        self.param_tree.heading("value", text="Value", anchor=tk.W)
        self.param_tree.column("#0", width=190, stretch=False, anchor=tk.W)
        self.param_tree.column("value", width=360, stretch=True, anchor=tk.W)
        self.param_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        param_scroll = ttk.Scrollbar(param_tree_frame, orient=tk.VERTICAL, command=self.param_tree.yview)
        param_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.param_tree.configure(yscrollcommand=param_scroll.set)
        self.param_tree.tag_configure("param-primary", foreground="#21384f", font=("Helvetica", 11, "bold"))
        self.param_tree.tag_configure("param-meta", foreground="#6a7280")
        self.param_tree.bind("<Double-1>", self._edit_selected_parameter)
        self.param_tree.bind("<Return>", self._edit_selected_parameter)
        self.param_tree.bind("<<TreeviewSelect>>", self._on_param_tree_selected)

        quick_edit_frame = ttk.Frame(param_frame, padding=(0, 0, 0, 0))
        self.quick_edit_frame = quick_edit_frame
        ttk.Label(quick_edit_frame, text="Quick Edit", style="Muted.TLabel").grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        self.quick_edit_key_combo = ttk.Combobox(
            quick_edit_frame,
            textvariable=self.quick_edit_key_var,
            state="readonly",
            width=18,
        )
        self.quick_edit_key_combo.grid(row=0, column=1, sticky=tk.EW, padx=(0, 8))
        self.quick_edit_key_combo.bind("<<ComboboxSelected>>", self._on_quick_edit_field_changed)
        ttk.Label(quick_edit_frame, text="Value").grid(row=0, column=2, sticky=tk.W, padx=(0, 6))
        self.quick_edit_value_entry = ttk.Entry(quick_edit_frame, textvariable=self.quick_edit_value_var, width=24)
        self.quick_edit_value_entry.grid(row=0, column=3, sticky=tk.EW)
        self.quick_edit_value_entry.bind("<Return>", self._apply_quick_edit)
        self.quick_edit_open_button = ttk.Button(
            quick_edit_frame,
            text="Full Edit",
            command=self._edit_selected_object,
            style="ObjectAction.TButton",
        )
        self.quick_edit_open_button.grid(row=0, column=4, sticky=tk.W, padx=(8, 0))
        self.quick_edit_apply_button = ttk.Button(
            quick_edit_frame,
            text="Apply",
            command=self._apply_quick_edit,
            style="ObjectAction.TButton",
        )
        self.quick_edit_apply_button.grid(row=0, column=5, sticky=tk.E, padx=(8, 0))
        ttk.Label(quick_edit_frame, textvariable=self.quick_edit_hint_var, style="Muted.TLabel").grid(
            row=1, column=0, columnspan=6, sticky=tk.W, pady=(4, 0)
        )
        quick_edit_frame.columnconfigure(1, weight=1)
        quick_edit_frame.columnconfigure(3, weight=2)
        quick_edit_frame.pack_forget()

        context_frame = ttk.Frame(self.param_notebook)
        self.param_notebook.add(context_frame, text="Context")
        context_header = ttk.Frame(context_frame)
        context_header.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(context_header, text="Current Context", style="Section.TLabel").pack(side=tk.LEFT)
        self.context_open_button = ttk.Button(
            context_header,
            text="Open",
            command=self._open_selected_object_file,
            style="ObjectAction.TButton",
        )
        self.context_open_button.pack(side=tk.RIGHT)
        self.context_macro_button = ttk.Button(
            context_header,
            text="Macro",
            command=self._add_inline_macro,
            style="ObjectAction.TButton",
        )
        self.context_macro_button.pack(side=tk.RIGHT, padx=(6, 0))
        self.context_edit_button = ttk.Button(
            context_header,
            text="Edit",
            command=self._edit_selected_object,
            style="ObjectAction.TButton",
        )
        self.context_edit_button.pack(side=tk.RIGHT, padx=(6, 0))
        self.context_tree = ttk.Treeview(context_frame, columns=("value",), show="tree headings")
        self.context_tree.heading("#0", text="Context", anchor=tk.W)
        self.context_tree.heading("value", text="Value", anchor=tk.W)
        self.context_tree.column("#0", width=180, stretch=False, anchor=tk.W)
        self.context_tree.column("value", width=360, stretch=True, anchor=tk.W)
        self.context_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        context_scroll = ttk.Scrollbar(context_frame, orient=tk.VERTICAL, command=self.context_tree.yview)
        context_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.context_tree.configure(yscrollcommand=context_scroll.set)
        self.context_tree.tag_configure("context-primary", foreground="#21384f", font=("Helvetica", 11, "bold"))
        self.context_tree.tag_configure("context-meta", foreground="#6a7280")
        self.context_tree.tag_configure("context-support", foreground="#35586d")

        resolved_frame = ttk.Frame(self.param_notebook)
        self.param_notebook.add(resolved_frame, text="Resolved")
        self.resolved_text = tk.Text(
            resolved_frame,
            wrap=tk.NONE,
            state=tk.DISABLED,
            relief=tk.FLAT,
            background="#faf7f0",
            foreground="#313131",
        )
        self.resolved_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        resolved_scroll = ttk.Scrollbar(resolved_frame, orient=tk.VERTICAL, command=self.resolved_text.yview)
        resolved_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.resolved_text.configure(yscrollcommand=resolved_scroll.set)

        help_frame = ttk.Frame(self.param_notebook)
        self.param_notebook.add(help_frame, text="Help")
        help_header = ttk.Frame(help_frame)
        help_header.pack(fill=tk.X, pady=(0, 6))
        self.help_validate_button = ttk.Button(
            help_header,
            text="Validate",
            style="ObjectAction.TButton",
            command=self._validate_current_project,
        )
        self.help_validate_button.pack(side=tk.RIGHT)
        self.help_open_button = ttk.Button(
            help_header,
            text="Open",
            style="ObjectAction.TButton",
            command=self._open_selected_object_file,
        )
        self.help_open_button.pack(side=tk.RIGHT, padx=(6, 0))
        self.help_macro_button = ttk.Button(
            help_header,
            text="Macro",
            style="ObjectAction.TButton",
            command=self._add_inline_macro,
        )
        self.help_macro_button.pack(side=tk.RIGHT, padx=(6, 0))
        self.help_edit_button = ttk.Button(
            help_header,
            text="Edit",
            style="ObjectAction.TButton",
            command=self._edit_selected_object,
        )
        self.help_edit_button.pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Label(help_header, text="Problems & Suggestions", style="Section.TLabel").pack(side=tk.LEFT)
        self.help_state_badge = tk.Label(
            help_header,
            text="Info",
            padx=8,
            pady=2,
            background="#e8edf2",
            foreground="#425466",
        )
        self.help_state_badge.pack(side=tk.LEFT, padx=(10, 0))
        self.help_summary_label = ttk.Label(help_header, textvariable=self.help_summary_var, style="Muted.TLabel")
        self.help_summary_label.pack(side=tk.LEFT, padx=(8, 0))

        help_body = ttk.Panedwindow(help_frame, orient=tk.VERTICAL)
        help_body.pack(fill=tk.BOTH, expand=True)

        help_issues_frame = ttk.Frame(help_body)
        self.help_issue_tree = ttk.Treeview(
            help_issues_frame,
            columns=("severity", "message", "action"),
            show="headings",
            height=5,
        )
        self.help_issue_tree.heading("severity", text="Level", anchor=tk.W)
        self.help_issue_tree.heading("message", text="Problem", anchor=tk.W)
        self.help_issue_tree.heading("action", text="Action", anchor=tk.W)
        self.help_issue_tree.column("severity", width=75, stretch=False, anchor=tk.W)
        self.help_issue_tree.column("message", width=320, stretch=True, anchor=tk.W)
        self.help_issue_tree.column("action", width=100, stretch=False, anchor=tk.W)
        self.help_issue_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.help_issue_tree.bind("<Double-1>", self._open_selected_help_issue)
        self.help_issue_tree.bind("<Return>", self._open_selected_help_issue)
        help_issue_scroll = ttk.Scrollbar(help_issues_frame, orient=tk.VERTICAL, command=self.help_issue_tree.yview)
        help_issue_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.help_issue_tree.configure(yscrollcommand=help_issue_scroll.set)

        help_suggestions_frame = ttk.Frame(help_body)
        self.help_suggestion_list = tk.Listbox(
            help_suggestions_frame,
            exportselection=False,
            activestyle="none",
            height=6,
        )
        self.help_suggestion_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.help_suggestion_list.bind("<Double-1>", self._run_selected_help_suggestion)
        self.help_suggestion_list.bind("<Return>", self._run_selected_help_suggestion)
        help_suggestion_scroll = ttk.Scrollbar(
            help_suggestions_frame,
            orient=tk.VERTICAL,
            command=self.help_suggestion_list.yview,
        )
        help_suggestion_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.help_suggestion_list.configure(yscrollcommand=help_suggestion_scroll.set)

        help_body.add(help_issues_frame, weight=3)
        help_body.add(help_suggestions_frame, weight=2)

        file_browser_frame = ttk.Frame(self.param_notebook)
        self.param_notebook.add(file_browser_frame, text="Files")
        self.file_browser_tree = ttk.Treeview(file_browser_frame, columns=("type",), show="tree headings")
        self.file_browser_tree.heading("#0", text="Path", anchor=tk.W)
        self.file_browser_tree.heading("type", text="Type", anchor=tk.W)
        self.file_browser_tree.column("#0", width=280, stretch=True, anchor=tk.W)
        self.file_browser_tree.column("type", width=90, stretch=False, anchor=tk.W)
        self.file_browser_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        file_browser_scroll = ttk.Scrollbar(file_browser_frame, orient=tk.VERTICAL, command=self.file_browser_tree.yview)
        file_browser_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_browser_tree.configure(yscrollcommand=file_browser_scroll.set)
        self.file_browser_tree.bind("<<TreeviewSelect>>", self._on_file_browser_selected)
        self.file_browser_tree.bind("<Double-1>", self._on_file_browser_selected)
        self.file_browser_tree.bind("<<TreeviewOpen>>", self._on_file_browser_open)

        ttk.Label(right, text="Editor", style="Section.TLabel").pack(anchor=tk.W)
        editor_header = ttk.Frame(right)
        editor_header.pack(fill=tk.X, pady=(4, 4))
        ttk.Label(editor_header, textvariable=self.editor_file_var).pack(side=tk.LEFT)
        search_frame = ttk.Frame(editor_header)
        search_frame.pack(side=tk.RIGHT)
        ttk.Button(editor_header, text="Revert", command=self._reload_current_from_disk).pack(side=tk.RIGHT)
        ttk.Checkbutton(
            editor_header,
            text="Macro Hover",
            variable=self.editor_macro_hover_var,
            command=self._on_editor_assist_setting_changed,
        ).pack(side=tk.RIGHT, padx=(0, 8))
        ttk.Checkbutton(
            editor_header,
            text="EC Help",
            variable=self.editor_auto_complete_var,
            command=self._on_editor_assist_setting_changed,
        ).pack(side=tk.RIGHT, padx=(0, 8))
        ttk.Label(search_frame, text="Find", style="Muted.TLabel").pack(side=tk.RIGHT, padx=(0, 6))
        ttk.Button(search_frame, text="Next", command=lambda: self._goto_search_match(forward=True)).pack(side=tk.RIGHT)
        ttk.Button(search_frame, text="Prev", command=lambda: self._goto_search_match(forward=False)).pack(
            side=tk.RIGHT, padx=(0, 4)
        )
        search_entry = tk.Entry(search_frame, textvariable=self.editor_search_var, width=26, relief=tk.FLAT, highlightthickness=1)
        search_entry.pack(side=tk.RIGHT, padx=(0, 6))
        search_entry.bind("<KeyRelease>", self._on_search_changed)
        search_entry.bind("<Return>", lambda _event: self._goto_search_match(forward=True))
        search_entry.bind("<FocusIn>", self._on_editor_search_focus_in)
        search_entry.bind("<FocusOut>", self._on_editor_search_focus_out)
        self.editor_search_entry = search_entry
        self._set_editor_search_placeholder()
        self.root.bind_all("<Control-f>", self._focus_editor_search, add="+")
        self.root.bind_all("<Control-Shift-S>", self._on_ctrl_shift_s, add="+")
        self.root.bind_all("<Command-Shift-S>", self._on_ctrl_shift_s, add="+")
        self.startup_tree.bind("<Delete>", self._on_tree_delete, add="+")
        self.startup_tree.bind("<BackSpace>", self._on_tree_delete, add="+")

        editor_body = ttk.Frame(right)
        editor_body.pack(fill=tk.BOTH, expand=True)
        self.editor_gutter = tk.Text(
            editor_body,
            width=5,
            padx=6,
            takefocus=0,
            wrap=tk.NONE,
            state=tk.DISABLED,
            background="#f4f1ea",
            foreground="#7a6f63",
            relief=tk.FLAT,
        )
        self.editor_gutter.pack(side=tk.LEFT, fill=tk.Y)

        self.editor_text = tk.Text(editor_body, wrap=tk.NONE, undo=True)
        self.editor_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.editor_scrollbar = ttk.Scrollbar(editor_body, orient=tk.VERTICAL, command=self._on_editor_scrollbar)
        self.editor_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll = ttk.Scrollbar(right, orient=tk.HORIZONTAL, command=self.editor_text.xview)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.editor_text.configure(yscrollcommand=self._on_editor_yview, xscrollcommand=x_scroll.set)
        self.editor_text.tag_configure("current_line", background="#fff3bf")
        self.editor_text.tag_configure("tree_target_line", background="#9fd0ff", foreground="#10243a")
        self.editor_text.tag_configure("match_bracket", background="#d7ecff", foreground="#003b73")
        self.editor_text.tag_configure("search_match", background="#fff1a8")
        self.editor_text.tag_configure("search_current", background="#ffd166")
        self.editor_text.tag_configure("syntax_comment", foreground="#6b7280")
        self.editor_text.tag_configure("syntax_string", foreground="#0b7a75")
        self.editor_text.tag_configure("syntax_macro", foreground="#7c3aed")
        self.editor_text.tag_configure("syntax_keyword", foreground="#b45309")
        self.editor_text.tag_configure("syntax_number", foreground="#1d4ed8")
        self.editor_text.tag_configure("syntax_key", foreground="#047857")
        self.editor_gutter.tag_configure("tree_target_line", background="#c7e1ff", foreground="#17324d")
        self.editor_text.bind("<<Modified>>", self._on_editor_modified)
        self.editor_text.bind("<KeyRelease>", self._on_editor_cursor_changed, add="+")
        self.editor_text.bind("<Motion>", self._on_editor_mouse_motion, add="+")
        self.editor_text.bind("<Leave>", self._on_editor_mouse_leave, add="+")
        self.editor_text.bind("<ButtonPress-1>", self._on_editor_mouse_press, add="+")
        self.editor_text.bind("<ButtonPress-2>", self._on_editor_mouse_press, add="+")
        self.editor_text.bind("<ButtonPress-3>", self._on_editor_mouse_press, add="+")
        self.editor_text.bind("<ButtonRelease-1>", self._on_editor_clicked, add="+")
        self.editor_text.bind("<ButtonRelease-2>", self._on_editor_clicked, add="+")
        self.editor_text.bind("<ButtonRelease-3>", self._on_editor_clicked, add="+")
        self.editor_text.bind("<Control-space>", self._show_editor_completion_on_demand)
        self.editor_text.bind("<Tab>", self._on_editor_tab)
        self.editor_text.bind("<Shift-Tab>", self._unindent_selection)
        try:
            self.editor_text.bind("<ISO_Left_Tab>", self._unindent_selection)
        except Exception:
            pass
        self.editor_text.bind("<Down>", self._on_editor_completion_down, add="+")
        self.editor_text.bind("<Up>", self._on_editor_completion_up, add="+")
        self.editor_text.bind("<Return>", self._on_editor_completion_accept, add="+")
        self.editor_text.bind("<Escape>", self._on_editor_completion_escape, add="+")
        self.root.bind_all("<Control-s>", self._on_ctrl_s, add="+")
        self.root.bind_all("<Command-s>", self._on_ctrl_s, add="+")

        self.log_frame = ttk.Frame(center)
        log_header = ttk.Frame(self.log_frame, padding=(8, 6))
        log_header.pack(fill=tk.X)
        ttk.Label(log_header, text="Validation & Activity", style="Section.TLabel").pack(side=tk.LEFT)
        ttk.Label(log_header, textvariable=self.validation_summary_var).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(log_header, text="Clear", style="Toolbar.TButton", command=self._clear_activity_log).pack(side=tk.RIGHT)
        ttk.Button(log_header, text="Hide", style="Toolbar.TButton", command=self._hide_log_panel).pack(side=tk.RIGHT, padx=(0, 6))

        log_body = ttk.Panedwindow(self.log_frame, orient=tk.VERTICAL)
        log_body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        validation_frame = ttk.Frame(log_body)
        ttk.Label(validation_frame, text="Validation", style="Section.TLabel").pack(anchor=tk.W, pady=(0, 4))
        issue_container = ttk.Frame(validation_frame)
        issue_container.pack(fill=tk.BOTH, expand=True)
        issue_tree = self.ttk.Treeview(
            issue_container,
            columns=("severity", "location", "message"),
            show="headings",
        )
        issue_tree.heading("severity", text="Severity", anchor=self.tk.W)
        issue_tree.heading("location", text="Location", anchor=self.tk.W)
        issue_tree.heading("message", text="Message", anchor=self.tk.W)
        issue_tree.column("severity", width=90, stretch=False, anchor=self.tk.W)
        issue_tree.column("location", width=240, stretch=False, anchor=self.tk.W)
        issue_tree.column("message", width=760, stretch=True, anchor=self.tk.W)
        issue_tree.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)
        issue_tree.bind("<<TreeviewSelect>>", self._on_issue_selected)
        issue_tree.tag_configure("error", foreground="#8b0000")
        issue_tree.tag_configure("warning", foreground="#8a5a00")
        issue_tree.tag_configure("info", foreground="#005a9c")
        issue_scroll = self.ttk.Scrollbar(issue_container, orient=self.tk.VERTICAL, command=issue_tree.yview)
        issue_scroll.pack(side=self.tk.RIGHT, fill=self.tk.Y)
        issue_tree.configure(yscrollcommand=issue_scroll.set)
        self.validation_issue_tree = issue_tree

        activity_frame = ttk.Frame(log_body)
        ttk.Label(activity_frame, text="Activity", style="Section.TLabel").pack(anchor=tk.W, pady=(0, 4))
        activity_container = ttk.Frame(activity_frame)
        activity_container.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(
            activity_container,
            height=8,
            wrap=tk.WORD,
            state=tk.DISABLED,
            background="#f7f3eb",
            foreground="#313131",
            relief=tk.FLAT,
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        activity_scroll = ttk.Scrollbar(activity_container, orient=tk.VERTICAL, command=self.log_text.yview)
        activity_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=activity_scroll.set)

        log_body.add(validation_frame, weight=3)
        log_body.add(activity_frame, weight=2)

        bottom = ttk.Frame(self.root, padding=(8, 4))
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(bottom, textvariable=self.status_var).pack(side=tk.LEFT)
        ttk.Button(bottom, text="Quit", command=self.root.destroy).pack(side=tk.RIGHT)

    def _configure_startup_tree_tags(self) -> None:
        tag_styles = {
            "node-file": {"foreground": "#445063", "font": ("Helvetica", 12, "bold")},
            "node-category": {"foreground": "#506070", "font": ("Helvetica", 11, "bold")},
            "node-link": {"foreground": "#0b5cad"},
            "node-detail": {"foreground": "#485266"},
            "editor-current": {"background": "#d8e9ff"},
            "group-command": {"foreground": "#7a3e00", "font": ("Helvetica", 11, "bold")},
            "group-linked": {"foreground": "#005f73", "font": ("Helvetica", 11, "bold")},
            "status-dirty": {"background": "#eef5ff"},
            "status-warning": {"background": "#fff4d8"},
            "status-error": {"background": "#fde8e8"},
            "status-missing": {"background": "#fde8e8"},
            "kind-macro": {"foreground": "#7c3a8c"},
            "kind-require": {"foreground": "#8b5e00"},
            "kind-master": {"foreground": "#8a5300"},
            "kind-slave": {"foreground": "#9c4221"},
            "kind-slave_config": {"foreground": "#c77700"},
            "kind-apply_config": {"foreground": "#6d4c41"},
            "kind-axis": {"foreground": "#0b5cad"},
            "kind-configured_axis": {"foreground": "#1c78c0"},
            "kind-encoder": {"foreground": "#00796b"},
            "kind-plc": {"foreground": "#2e7d32"},
            "kind-plugin": {"foreground": "#6a1b9a"},
            "kind-datastorage": {"foreground": "#6d4c41"},
            "kind-component": {"foreground": "#ad1457"},
            "kind-ecsdo": {"foreground": "#c62828"},
            "kind-ecdataitem": {"foreground": "#00838f"},
            "kind-master_slave_sm": {"foreground": "#7b1fa2"},
            "kind-subst_config": {"foreground": "#5d4037"},
            "kind-pvt_controller": {"foreground": "#00695c"},
            "kind-lut": {"foreground": "#2e7d32"},
            "kind-plcvar_analog": {"foreground": "#558b2f"},
            "kind-plcvar_binary": {"foreground": "#33691e"},
            "kind-ecmc_command": {"foreground": "#5c6bc0"},
            "kind-record_update_rate": {"foreground": "#1565c0"},
            "kind-restore_record_update_rate": {"foreground": "#6d4c41"},
        }
        for tag_name, options in tag_styles.items():
            self.startup_tree.tag_configure(tag_name, **options)

    def _tree_tags_for_object(self, obj: StartupObject) -> Tuple[str, ...]:
        return ("node-object", "kind-{}".format(obj.kind))

    def _tree_object_symbol(self, kind: str) -> str:
        return {
            "require": "⇢",
            "master": "◇",
            "macro": "※",
            "slave": "◉",
            "slave_config": "◔",
            "apply_config": "⇣",
            "axis": "◎",
            "configured_axis": "◍",
            "encoder": "◌",
            "plc": "▤",
            "plugin": "◈",
            "datastorage": "▥",
            "component": "◆",
            "ecsdo": "◍",
            "ecdataitem": "◐",
            "master_slave_sm": "⇄",
            "subst_config": "▦",
            "pvt_controller": "◳",
            "lut": "▧",
            "plcvar_analog": "∿",
            "plcvar_binary": "⊙",
            "ecmc_command": "≣",
            "record_update_rate": "◷",
            "restore_record_update_rate": "◴",
        }.get(kind, "•")

    def _prefixed_tree_text(self, prefix: str, text: str) -> str:
        return "{} {}".format(prefix, text)

    def _tree_text_logical_name(self, text: str) -> str:
        for prefix in ("· ", "↗ ", "≡ ", "▣ "):
            if text.startswith(prefix):
                return text[len(prefix) :]
        return text

    def _schedule_tree_view_update(self, *_args) -> None:
        if self._tree_filter_update_job is not None:
            try:
                self.root.after_cancel(self._tree_filter_update_job)
            except Exception:
                pass
        self._tree_filter_update_job = self.root.after(80, self._apply_tree_view_update)

    def _apply_tree_view_update(self) -> None:
        self._tree_filter_update_job = None
        if self.startup_tree is None or self.latest_startup_tree is None:
            return
        startup_value = self.startup_var.get().strip()
        if not startup_value:
            return
        startup_path = Path(startup_value).expanduser().resolve()
        tree_state = self._capture_startup_tree_state()
        self._populate_startup_tree(startup_path, self.latest_startup_tree, tree_state=tree_state)

    def _on_tree_view_mode_changed(self, *_args) -> None:
        self._update_tree_view_toggle_text()

    def _tree_filter_kind(self) -> str:
        return self.tree_filter_var.get().strip().lower() or "all"

    def _update_tree_view_toggle_text(self) -> None:
        mode = self.tree_view_mode_var.get().strip().lower()
        self.tree_view_toggle_var.set("Show Objects" if mode == "flow" else "Show Flow")

    def _toggle_tree_view_mode(self) -> None:
        current = self.tree_view_mode_var.get().strip().lower()
        self.tree_view_mode_var.set("Objects" if current == "flow" else "Flow")

    def _tree_view_mode(self) -> str:
        return self.tree_view_mode_var.get().strip().lower() or "flow"

    def _tree_sort_mode(self) -> str:
        return self.tree_sort_var.get().strip().lower() or "flow"

    def _tree_search_query(self) -> str:
        if self._tree_search_placeholder_visible:
            return ""
        return self.tree_search_var.get().strip().lower()

    def _set_tree_search_placeholder(self) -> None:
        if self.tree_search_entry is None:
            return
        if self.tree_search_var.get().strip():
            return
        self._tree_search_placeholder_visible = True
        self.tree_search_var.set("Filter objects")
        self.tree_search_entry.configure(fg="#7a7a7a", highlightbackground="#d0cabf", highlightcolor="#c1b7a6")

    def _clear_tree_search_placeholder(self) -> None:
        if self.tree_search_entry is None or not self._tree_search_placeholder_visible:
            return
        self._tree_search_placeholder_visible = False
        self.tree_search_var.set("")
        self.tree_search_entry.configure(fg="#222222", highlightbackground="#d0cabf", highlightcolor="#8fb7e1")

    def _on_tree_search_focus_in(self, _event=None) -> None:
        self._clear_tree_search_placeholder()

    def _on_tree_search_focus_out(self, _event=None) -> None:
        if not self.tree_search_var.get().strip():
            self._set_tree_search_placeholder()

    def _tree_category_for_object(self, obj: StartupObject) -> str:
        if obj.kind in {"slave", "slave_config", "component", "ecsdo", "ecdataitem"}:
            return "Slaves"
        if obj.kind in {"axis", "configured_axis", "encoder", "master_slave_sm", "pvt_controller", "lut"}:
            return "Axes"
        if obj.kind in {"plc", "plcvar_analog", "plcvar_binary", "plugin", "datastorage"}:
            return "PLCs"
        if obj.kind in {"macro", "require", "record_update_rate", "restore_record_update_rate"}:
            return "Macros"
        if obj.kind == "ecmc_command":
            return "ECMC"
        return "Other"

    def _tree_filter_kind_match(self, obj: StartupObject, filter_kind: str) -> bool:
        if filter_kind == "all":
            return True
        if filter_kind == "slaves":
            return obj.kind in {"slave", "slave_config", "component", "ecsdo", "ecdataitem"}
        if filter_kind == "axes":
            return obj.kind in {"axis", "configured_axis", "encoder", "master_slave_sm", "pvt_controller", "lut"}
        if filter_kind == "plcs":
            return obj.kind in {"plc", "plcvar_analog", "plcvar_binary", "plugin", "datastorage"}
        if filter_kind == "macros":
            return obj.kind in {"macro", "require", "record_update_rate", "restore_record_update_rate"}
        if filter_kind == "ecmc":
            return obj.kind == "ecmc_command"
        return True

    def _tree_search_text_for_object(self, obj: StartupObject) -> str:
        parts = [obj.kind, obj.title, obj.summary, self._tree_summary_for_object(obj)]
        if obj.linked_file is not None:
            parts.append(self._relative_display(obj.linked_file))
        for key, value in obj.details:
            parts.append(str(key))
            parts.append(str(value))
        for key, value in obj.command_details:
            parts.append(str(key))
            parts.append(str(value))
        for key, value in obj.linked_file_details:
            parts.append(str(key))
            parts.append(str(value))
        return " ".join(part for part in parts if part).lower()

    def _object_matches_tree_view(
        self,
        obj: StartupObject,
        filter_kind: str,
        search_query: str,
        object_severity: str,
        source_dirty: bool,
        linked_dirty: bool,
        linked_missing: bool,
    ) -> bool:
        if filter_kind == "errors":
            if object_severity != "error":
                return False
        elif filter_kind == "warnings":
            if object_severity != "warning":
                return False
        elif filter_kind == "missing":
            if not linked_missing:
                return False
        elif filter_kind == "issues":
            if not (object_severity or linked_missing):
                return False
        elif filter_kind == "unsaved":
            if not (source_dirty or linked_dirty):
                return False
        elif not self._tree_filter_kind_match(obj, filter_kind):
            return False
        if not search_query:
            return True
        return search_query in self._tree_search_text_for_object(obj)

    def _tree_sort_value_for_object(self, obj: StartupObject, sort_mode: str) -> Tuple[object, ...]:
        if sort_mode == "name":
            return (obj.title.lower(), obj.line)
        if sort_mode == "id":
            detail_map = self._object_detail_map(obj)
            for key in (
                "SLAVE_ID",
                "AXIS_ID",
                "PLC_ID",
                "ENC_SID",
                "COMP_S_ID",
                "PLUGIN_ID",
                "DS_ID",
                "LUT_ID",
            ):
                parsed = _parse_int_value(str(detail_map.get(key, "")))
                if parsed is not None:
                    return (0, parsed, obj.title.lower(), obj.line)
            return (1, obj.title.lower(), obj.line)
        return (obj.line,)

    def _tree_category_summary_text(self, count: int, errors: int, warnings: int, dirty: int, missing: int) -> str:
        parts = ["{} object(s)".format(count)]
        if errors:
            parts.append("{} err".format(errors))
        if warnings:
            parts.append("{} warn".format(warnings))
        if dirty:
            parts.append("{} dirty".format(dirty))
        if missing:
            parts.append("{} missing".format(missing))
        return " • ".join(parts)

    def _focus_first_tree_match(self, _event=None) -> str:
        if self.startup_tree is None:
            return "break"

        def iter_items(parent: str = ""):
            for item_id in self.startup_tree.get_children(parent):
                yield item_id
                yield from iter_items(item_id)

        for item_id in iter_items(""):
            entry = self.startup_item_map.get(item_id)
            if entry is None:
                continue
            if entry[0] not in {"object", "linked-file", "detail", "linked-detail"}:
                continue
            self.startup_tree.selection_set(item_id)
            self.startup_tree.focus(item_id)
            self.startup_tree.see(item_id)
            self._on_startup_tree_selected()
            break
        return "break"

    def _iter_startup_tree_items(self, parent: str = ""):
        if self.startup_tree is None:
            return
        for item_id in self.startup_tree.get_children(parent):
            yield item_id
            yield from self._iter_startup_tree_items(item_id)

    def _reset_tree_view_controls(self) -> None:
        self.tree_view_mode_var.set("Flow")
        self.tree_filter_var.set("All")
        self.tree_sort_var.set("Flow")
        self.tree_search_var.set("")
        self.tree_compact_var.set(False)

    def _show_issue_tree_items(self) -> None:
        self.tree_view_mode_var.set("Objects")
        self.tree_filter_var.set("Issues")
        self.tree_search_var.set("")
        self.tree_compact_var.set(True)

    def _show_error_tree_items(self) -> None:
        self.tree_view_mode_var.set("Objects")
        self.tree_filter_var.set("Errors")
        self.tree_search_var.set("")
        self.tree_compact_var.set(True)

    def _show_warning_tree_items(self) -> None:
        self.tree_view_mode_var.set("Objects")
        self.tree_filter_var.set("Warnings")
        self.tree_search_var.set("")
        self.tree_compact_var.set(True)

    def _show_unsaved_tree_items(self) -> None:
        self.tree_view_mode_var.set("Objects")
        self.tree_filter_var.set("Unsaved")
        self.tree_search_var.set("")
        self.tree_compact_var.set(True)

    def _show_missing_tree_items(self) -> None:
        self.tree_view_mode_var.set("Objects")
        self.tree_filter_var.set("Missing")
        self.tree_search_var.set("")
        self.tree_compact_var.set(True)

    def _collapse_startup_tree(self) -> None:
        if self.startup_tree is None:
            return
        for item_id in self._iter_startup_tree_items(""):
            self.startup_tree.item(item_id, open=False)

    def _expand_issue_tree_nodes(self) -> None:
        if self.startup_tree is None:
            return
        issue_tags = {"status-error", "status-warning", "status-missing"}
        for item_id in self._iter_startup_tree_items(""):
            tags = set(self.startup_tree.item(item_id, "tags") or ())
            if tags & issue_tags:
                current = item_id
                while current:
                    self.startup_tree.item(current, open=True)
                    current = self.startup_tree.parent(current)

    def _tree_summary_for_object(self, obj: StartupObject) -> str:
        detail_map = self._object_detail_map(obj)
        summary_pairs: List[Tuple[str, str]] = []

        def add_pair(key: str, value: Optional[str]) -> None:
            cleaned = str(value or "").strip()
            if cleaned:
                summary_pairs.append((key, cleaned))

        if obj.kind == "require":
            add_pair("module", detail_map.get("MODULE"))
            add_pair("version", detail_map.get("VERSION"))
        elif obj.kind == "macro":
            add_pair("value", detail_map.get("VALUE"))
        elif obj.kind == "slave":
            add_pair("id", detail_map.get("SLAVE_ID"))
            add_pair("hw", detail_map.get("HW_DESC"))
        elif obj.kind == "slave_config":
            add_pair("config", detail_map.get("CONFIG") or detail_map.get("LOCAL_CONFIG"))
            add_pair("slave", detail_map.get("SLAVE_ID"))
        elif obj.kind == "axis":
            add_pair("name", detail_map.get("AX_NAME"))
            add_pair("id", detail_map.get("AXIS_ID"))
            add_pair("file", detail_map.get("FILE"))
        elif obj.kind == "configured_axis":
            add_pair("config", detail_map.get("CONFIG"))
            add_pair("dev", detail_map.get("DEV"))
        elif obj.kind == "encoder":
            add_pair("file", detail_map.get("FILE"))
            add_pair("enc", detail_map.get("ENC_SID"))
        elif obj.kind == "plc":
            add_pair("plc", detail_map.get("PLC_ID"))
            add_pair("file", detail_map.get("FILE"))
        elif obj.kind == "master_slave_sm":
            add_pair("name", detail_map.get("NAME"))
            add_pair("master", detail_map.get("MST_GRP_NAME"))
            add_pair("slave", detail_map.get("SLV_GRP_NAME"))
        elif obj.kind == "subst_config":
            add_pair("file", detail_map.get("FILE"))
        elif obj.kind == "apply_config":
            add_pair("state", "bus apply")
        elif obj.kind == "pvt_controller":
            add_pair("trg", detail_map.get("TRG_EC_ENTRY"))
            add_pair("axes", detail_map.get("NAXES"))
        elif obj.kind == "lut":
            add_pair("lut", detail_map.get("LUT_ID"))
            add_pair("file", detail_map.get("FILE"))
        elif obj.kind == "component":
            add_pair("comp", detail_map.get("COMP"))
            add_pair("slave", detail_map.get("COMP_S_ID"))
            add_pair("ch", detail_map.get("CH_ID"))
        elif obj.kind == "plugin":
            add_pair("plugin", detail_map.get("PLUGIN_ID"))
            add_pair("file", detail_map.get("FILE"))
        elif obj.kind == "datastorage":
            add_pair("ds", detail_map.get("DS_ID"))
            add_pair("size", detail_map.get("DS_SIZE"))
        elif obj.kind == "ecsdo":
            add_pair("slave", detail_map.get("SLAVE_ID"))
            add_pair("idx", detail_map.get("INDEX"))
            add_pair("sub", detail_map.get("SUBINDEX"))
        elif obj.kind == "ecdataitem":
            add_pair("slave", detail_map.get("STRT_ENTRY_S_ID"))
            add_pair("entry", detail_map.get("STRT_ENTRY_NAME"))
        elif obj.kind in {"plcvar_analog", "plcvar_binary"}:
            add_pair("rec", detail_map.get("REC_NAME"))
            add_pair("asyn", detail_map.get("ASYN_NAME"))
        elif obj.kind == "ecmc_command":
            add_pair("cmd", detail_map.get("COMMAND") or detail_map.get("EXPANDED_COMMAND"))
        elif obj.kind == "record_update_rate":
            add_pair("rate", detail_map.get("RATE_MS"))

        if not summary_pairs and obj.linked_file is not None:
            add_pair("file", self._relative_display(obj.linked_file))
        if not summary_pairs:
            return obj.summary

        display_parts: List[str] = []
        for key, value in summary_pairs[:3]:
            trimmed = value
            if len(trimmed) > 44:
                trimmed = trimmed[:41] + "..."
            display_parts.append("{}={}".format(key, trimmed))
        return "  |  ".join(display_parts)

    def _selection_header_text(self, entry: Optional[Tuple[str, object]]) -> str:
        if entry is None:
            return "No object selected"
        entry_type, payload = entry
        if entry_type == "file":
            return "File | {}".format(self._relative_display(payload.path))
        if entry_type == "category-group":
            file_path, category, count = payload
            return "Group | {} | {} | {} object(s)".format(category, self._relative_display(file_path), count)
        obj = payload
        parts = [obj.kind.upper(), obj.title]
        summary = self._tree_summary_for_object(obj)
        if summary:
            parts.append(summary)
        return " | ".join(parts)

    def _path_is_dirty(self, path: Optional[Path]) -> bool:
        if path is None:
            return False
        resolved = path.resolve()
        if resolved not in self.file_buffers:
            return False
        buffered = self.file_buffers[resolved]
        if not resolved.exists():
            return bool(buffered)
        return _read_text(resolved) != buffered

    def _configure_badge(self, widget, text: str, tone: str) -> None:
        if widget is None:
            return
        palette = {
            "neutral": ("#e8edf2", "#425466"),
            "ok": ("#e6f3e6", "#2f5e34"),
            "info": ("#e5effc", "#1f4f85"),
            "warning": ("#fff1d6", "#8a5a00"),
            "error": ("#f9d9dd", "#8b1e2d"),
            "dirty": ("#ddeafe", "#0b5cad"),
            "missing": ("#ecdef9", "#6e3ea3"),
        }
        background, foreground = palette.get(tone, palette["neutral"])
        widget.configure(text=text, background=background, foreground=foreground)

    def _entry_badge_snapshot(self, entry: Optional[Tuple[str, object]]) -> Tuple[str, str, str, str, str]:
        if entry is None:
            editor_target = self._relative_display(self.current_edit_path) if self.current_edit_path is not None else "No file open"
            return ("None", "neutral", "No selection", "neutral", editor_target)

        entry_type, payload = entry
        issues = self._issues_for_entry(entry)
        error_count = sum(1 for issue in issues if issue.severity == "error")
        warning_count = sum(1 for issue in issues if issue.severity == "warning")
        dirty = False
        missing = False

        if entry_type == "file":
            source_text = self._relative_display(payload.path)
            kind_text = "FILE"
            dirty = self._path_is_dirty(payload.path)
        elif entry_type == "category-group":
            file_path, category, _count = payload
            source_text = self._relative_display(file_path)
            kind_text = category.upper()
        else:
            obj = payload
            kind_text = obj.kind.upper()
            source_text = "{}:{}".format(self._relative_display(obj.source), obj.line)
            dirty = self._path_is_dirty(obj.source) or self._path_is_dirty(obj.linked_file)
            missing = bool(obj.linked_file is not None and not obj.linked_file.exists())

        state_parts: List[str] = []
        tone = "ok"
        if error_count:
            state_parts.append("{} error{}".format(error_count, "" if error_count == 1 else "s"))
            tone = "error"
        elif warning_count:
            state_parts.append("{} warning{}".format(warning_count, "" if warning_count == 1 else "s"))
            tone = "warning"
        elif dirty:
            state_parts.append("Unsaved")
            tone = "dirty"
        elif missing:
            state_parts.append("Missing file")
            tone = "missing"
        else:
            state_parts.append("OK")
            tone = "ok"

        if dirty and "Unsaved" not in state_parts:
            state_parts.append("Unsaved")
        if missing and "Missing file" not in state_parts:
            state_parts.append("Missing file")
            if tone == "ok":
                tone = "missing"
        return (kind_text, "info", " • ".join(state_parts), tone, source_text)

    def _update_selection_badges(self, entry: Optional[Tuple[str, object]]) -> None:
        kind_text, kind_tone, state_text, state_tone, source_text = self._entry_badge_snapshot(entry)
        self._configure_badge(self.selection_kind_badge, kind_text, kind_tone)
        self._configure_badge(self.selection_state_badge, state_text, state_tone)
        self._configure_badge(self.selection_source_badge, source_text, "neutral")

    def _update_help_header(self, entry: Optional[Tuple[str, object]]) -> None:
        issues = self._issues_for_entry(entry)
        error_count = sum(1 for issue in issues if issue.severity == "error")
        warning_count = sum(1 for issue in issues if issue.severity == "warning")
        suggestions = self._help_suggestions_for_entry(entry)
        if error_count:
            self._configure_badge(self.help_state_badge, "{} error{}".format(error_count, "" if error_count == 1 else "s"), "error")
            self.help_summary_var.set("Fix the errors below first. Double-click a problem to jump to source.")
        elif warning_count:
            self._configure_badge(self.help_state_badge, "{} warning{}".format(warning_count, "" if warning_count == 1 else "s"), "warning")
            self.help_summary_var.set("Review warnings and suggestions below before saving.")
        elif entry is None:
            self._configure_badge(self.help_state_badge, "Info", "neutral")
            self.help_summary_var.set("Select an object to see targeted problems and suggestions.")
        else:
            self._configure_badge(self.help_state_badge, "OK", "ok")
            self.help_summary_var.set("{} suggestion{}".format(len(suggestions), "" if len(suggestions) == 1 else "s"))

    def _status_rank(self, severity: str) -> int:
        return {"error": 2, "warning": 1}.get(severity, 0)

    def _worst_status(self, left: str, right: str) -> str:
        return left if self._status_rank(left) >= self._status_rank(right) else right

    def _tree_status_badge(self, severity: str = "", dirty: bool = False, missing: bool = False) -> str:
        markers: List[str] = []
        if missing:
            markers.append("×")
        if severity == "error":
            markers.append("!")
        elif severity == "warning":
            markers.append("?")
        if dirty:
            markers.append("●")
        return "[{}] ".format("".join(markers)) if markers else ""

    def _tree_status_tags(self, severity: str = "", dirty: bool = False, missing: bool = False) -> Tuple[str, ...]:
        tags: List[str] = []
        if missing:
            tags.append("status-missing")
        if severity == "error":
            tags.append("status-error")
        elif severity == "warning":
            tags.append("status-warning")
        if dirty:
            tags.append("status-dirty")
        return tuple(tags)

    def _object_tree_key(self, obj: StartupObject) -> Tuple[Path, int, str, str]:
        return (obj.source.resolve(), obj.line, obj.kind, obj.title)

    def _tree_label_for_object(self, obj: StartupObject, flow_index: Optional[int]) -> str:
        title = self._prefixed_tree_text(self._tree_object_symbol(obj.kind), obj.title)
        if flow_index is None:
            return title
        return "{:02d} ↓ {}".format(flow_index, title)

    def _on_editor_scrollbar(self, *args) -> None:
        self.editor_text.yview(*args)
        self.editor_gutter.yview(*args)

    def _on_editor_yview(self, first: str, last: str) -> None:
        if self.editor_scrollbar is not None:
            self.editor_scrollbar.set(first, last)
        self.editor_gutter.yview_moveto(first)

    def _focus_editor_search(self, _event=None) -> str:
        if self.editor_search_entry is not None:
            self.editor_search_entry.focus_set()
            if self._editor_search_placeholder_visible:
                self._clear_editor_search_placeholder()
            self.editor_search_entry.selection_range(0, "end")
        return "break"

    def _set_editor_search_placeholder(self) -> None:
        if self.editor_search_entry is None:
            return
        if self.editor_search_var.get().strip():
            return
        self._editor_search_placeholder_visible = True
        self.editor_search_var.set("Find in current file")
        self.editor_search_entry.configure(fg="#7a7a7a", highlightbackground="#d0cabf", highlightcolor="#c1b7a6")

    def _clear_editor_search_placeholder(self) -> None:
        if self.editor_search_entry is None or not self._editor_search_placeholder_visible:
            return
        self._editor_search_placeholder_visible = False
        self.editor_search_var.set("")
        self.editor_search_entry.configure(fg="#222222", highlightbackground="#d0cabf", highlightcolor="#8fb7e1")

    def _effective_editor_search_query(self) -> str:
        if self._editor_search_placeholder_visible:
            return ""
        return self.editor_search_var.get().strip()

    def _on_editor_search_focus_in(self, _event=None) -> None:
        self._clear_editor_search_placeholder()

    def _on_editor_search_focus_out(self, _event=None) -> None:
        if not self.editor_search_var.get().strip():
            self._set_editor_search_placeholder()

    def _on_ctrl_shift_s(self, _event=None) -> str:
        self._save_all_files()
        return "break"

    def _on_tree_delete(self, _event=None) -> str:
        self._remove_selected_object()
        return "break"

    def _on_editor_assist_setting_changed(self) -> None:
        if not self.editor_macro_hover_var.get():
            self._hide_editor_macro_tooltip()
        if not self.editor_auto_complete_var.get():
            self._hide_editor_completion()
        self._schedule_editor_update()

    def _show_combobox_dropdown(self, combobox) -> None:
        try:
            combobox.tk.call("ttk::combobox::Post", str(combobox))
        except Exception:
            try:
                combobox.event_generate("<Down>")
            except Exception:
                pass

    def _hide_combobox_dropdown(self, combobox) -> None:
        try:
            combobox.tk.call("ttk::combobox::Unpost", str(combobox))
        except Exception:
            pass

    def _enable_combobox_filter(self, combobox, values: List[str], auto_post: bool = False) -> None:
        all_values = list(values)

        def refresh(_event=None, should_post: bool = False) -> None:
            current = combobox.get().strip().lower()
            if not current:
                filtered = all_values
            else:
                startswith_matches = [value for value in all_values if value.lower().startswith(current)]
                contains_matches = [
                    value for value in all_values if current in value.lower() and value not in startswith_matches
                ]
                filtered = startswith_matches + contains_matches
            combobox.configure(values=filtered)
            if auto_post and should_post and combobox.focus_get() == combobox and filtered:
                combobox.after_idle(lambda: self._show_combobox_dropdown(combobox))
            elif not filtered:
                self._hide_combobox_dropdown(combobox)

        def on_button_press(_event=None) -> None:
            refresh()
            if auto_post:
                combobox.after_idle(lambda: self._show_combobox_dropdown(combobox))

        def on_key_release(event=None) -> None:
            keysym = getattr(event, "keysym", "")
            if keysym in {"Up", "Down", "Return", "Escape", "Tab"}:
                return
            refresh(should_post=True)

        combobox.bind("<KeyRelease>", on_key_release, add="+")
        combobox.bind("<FocusIn>", refresh, add="+")
        combobox.bind("<ButtonPress-1>", on_button_press, add="+")
        combobox.bind("<FocusOut>", lambda _event: self._hide_combobox_dropdown(combobox), add="+")

    def _create_filtered_value_picker(self, parent, variable, values: List[str], width: int = 48, height: int = 8):
        tk = self.tk
        ttk = self.ttk

        frame = ttk.Frame(parent)
        entry = ttk.Entry(frame, textvariable=variable, width=width)
        entry.grid(row=0, column=0, sticky=tk.EW)

        list_frame = ttk.Frame(frame)
        list_frame.grid(row=1, column=0, sticky=tk.NSEW, pady=(4, 0))

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        listbox = tk.Listbox(
            list_frame,
            exportselection=False,
            height=height,
            activestyle="none",
        )
        listbox.grid(row=0, column=0, sticky=tk.NSEW)
        scrollbar.grid(row=0, column=1, sticky=tk.NS)
        listbox.configure(yscrollcommand=scrollbar.set)
        scrollbar.configure(command=listbox.yview)

        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        all_values = list(dict.fromkeys(str(value) for value in values if str(value).strip()))
        current_value = variable.get().strip()
        if current_value and current_value not in all_values:
            all_values.append(current_value)

        def filtered_values() -> List[str]:
            current = variable.get().strip().lower()
            if not current:
                return all_values
            startswith_matches = [value for value in all_values if value.lower().startswith(current)]
            contains_matches = [
                value for value in all_values if current in value.lower() and value not in startswith_matches
            ]
            return startswith_matches + contains_matches

        def populate(preferred: Optional[str] = None) -> None:
            filtered = filtered_values()
            current_selection = preferred
            selection = listbox.curselection()
            if not current_selection and selection:
                current_selection = str(listbox.get(selection[0]))
            listbox.delete(0, tk.END)
            for value in filtered:
                listbox.insert(tk.END, value)
            if not filtered:
                return
            target_value = current_selection or variable.get().strip()
            if target_value in filtered:
                index = filtered.index(target_value)
            else:
                index = 0
            listbox.selection_clear(0, tk.END)
            listbox.selection_set(index)
            listbox.activate(index)
            listbox.see(index)

        def choose_selected(_event=None) -> str:
            selection = listbox.curselection()
            if not selection:
                return "break"
            value = str(listbox.get(selection[0]))
            variable.set(value)
            entry.focus_set()
            entry.icursor(tk.END)
            populate(preferred=value)
            return "break"

        def move_selection(offset: int) -> str:
            if listbox.size() <= 0:
                return "break"
            selection = listbox.curselection()
            current_index = selection[0] if selection else 0
            next_index = max(0, min(listbox.size() - 1, current_index + offset))
            listbox.selection_clear(0, tk.END)
            listbox.selection_set(next_index)
            listbox.activate(next_index)
            listbox.see(next_index)
            return "break"

        def on_entry_key_release(event=None) -> None:
            keysym = getattr(event, "keysym", "")
            if keysym in {"Up", "Down", "Return", "Escape", "Tab"}:
                return
            populate()

        entry.bind("<KeyRelease>", on_entry_key_release, add="+")
        entry.bind("<Down>", lambda _event: move_selection(1), add="+")
        entry.bind("<Up>", lambda _event: move_selection(-1), add="+")
        entry.bind("<Return>", choose_selected, add="+")
        listbox.bind("<ButtonRelease-1>", choose_selected, add="+")
        listbox.bind("<Double-Button-1>", choose_selected, add="+")
        listbox.bind("<Return>", choose_selected, add="+")

        populate(preferred=current_value)
        return frame, entry

    def _ensure_editor_macro_tooltip(self) -> None:
        if self.editor_macro_tooltip is not None and self.editor_macro_tooltip.winfo_exists():
            return
        tooltip = self.tk.Toplevel(self.root)
        tooltip.withdraw()
        tooltip.overrideredirect(True)
        tooltip.transient(self.root)
        tooltip.configure(background="#d8c7a3")
        label = self.tk.Label(
            tooltip,
            text="",
            justify=self.tk.LEFT,
            anchor=self.tk.W,
            background="#fff8dd",
            padx=8,
            pady=4,
        )
        label.pack(fill=self.tk.BOTH, expand=True)
        self.editor_macro_tooltip = tooltip
        self.editor_macro_tooltip_label = label

    def _hide_editor_macro_tooltip(self) -> None:
        if self._editor_macro_hover_job is not None:
            try:
                self.root.after_cancel(self._editor_macro_hover_job)
            except Exception:
                pass
        self._editor_macro_hover_job = None
        self._editor_macro_hover_token = None
        self._editor_macro_hover_event = None
        if self.editor_macro_tooltip is not None and self.editor_macro_tooltip.winfo_exists():
            self.editor_macro_tooltip.withdraw()

    def _editor_macro_hover_context(self, event) -> Optional[Tuple[str, str]]:
        if self.current_edit_path is None:
            return None
        try:
            index = self.editor_text.index("@{},{}".format(event.x, event.y))
        except Exception:
            return None
        line_no, column_no = [int(part) for part in index.split(".")]
        line_text = self.editor_text.get("{}.0".format(line_no), "{}.0 lineend".format(line_no))
        for match in re.finditer(r"\$\{[^}\n]+\}|\$\([^\)\n]+\)", line_text):
            if not (match.start() <= column_no < match.end()):
                continue
            token = match.group(0)
            inner = token[2:-1]
            name, has_default, default = inner.partition("=")
            name = name.strip()
            if not name:
                return None
            macro_map = self._editor_known_macro_map()
            value = macro_map.get(name)
            if value is None or value == "":
                value = default.strip() if has_default else "<unset>"
            else:
                value = str(value)
            return token, "{} = {}".format(name, value)
        return None

    def _show_editor_macro_tooltip(self) -> None:
        self._editor_macro_hover_job = None
        if self._editor_macro_hover_event is None:
            return
        context = self._editor_macro_hover_context(self._editor_macro_hover_event)
        if context is None:
            self._hide_editor_macro_tooltip()
            return
        token, text = context
        self._editor_macro_hover_token = token
        self._ensure_editor_macro_tooltip()
        if self.editor_macro_tooltip is None or self.editor_macro_tooltip_label is None:
            return
        self.editor_macro_tooltip_label.configure(text=text)
        x = self.editor_text.winfo_rootx() + self._editor_macro_hover_event.x + 16
        y = self.editor_text.winfo_rooty() + self._editor_macro_hover_event.y + 20
        self.editor_macro_tooltip.geometry("+{}+{}".format(x, y))
        self.editor_macro_tooltip.deiconify()
        self.editor_macro_tooltip.lift()

    def _set_editor_content(self, content: str, line: Optional[int] = None) -> None:
        self._hide_editor_macro_tooltip()
        self._hide_editor_completion()
        self.editor_text.delete("1.0", "end")
        self.editor_text.insert("1.0", content)
        self.editor_text.edit_modified(False)
        self._highlight_editor_line(line)
        self._update_editor_visuals()

    def _update_editor_visuals(self) -> None:
        self._update_editor_gutter()
        self._highlight_current_editor_line()
        self._highlight_matching_bracket()
        self._update_search_highlight()
        self._highlight_editor_syntax()

    def _schedule_editor_update(self) -> None:
        if self._editor_update_job is not None:
            try:
                self.root.after_cancel(self._editor_update_job)
            except Exception:
                pass
        self._editor_update_job = self.root.after(60, self._run_editor_update)

    def _run_editor_update(self) -> None:
        self._editor_update_job = None
        self._update_editor_visuals()
        if self.editor_auto_complete_var.get() or self._editor_completion_forced:
            self._update_editor_completion(self._editor_completion_forced or self._editor_completion_should_open)
        else:
            self._hide_editor_completion()
        self._refresh_resolved_preview()
        self._editor_completion_forced = False
        self._editor_completion_should_open = False

    def _schedule_tree_sync_from_editor(self, delay_ms: int = 120) -> None:
        if self._editor_tree_sync_job is not None:
            try:
                self.root.after_cancel(self._editor_tree_sync_job)
            except Exception:
                pass
        self._editor_tree_sync_job = self.root.after(delay_ms, self._run_tree_sync_from_editor)

    def _cancel_tree_sync_from_editor(self) -> None:
        if self._editor_tree_sync_job is None:
            return
        try:
            self.root.after_cancel(self._editor_tree_sync_job)
        except Exception:
            pass
        self._editor_tree_sync_job = None

    def _run_tree_sync_from_editor(self) -> None:
        self._editor_tree_sync_job = None
        if time.monotonic() < self._editor_mouse_interacting_until:
            self._schedule_tree_sync_from_editor(120)
            return
        self._sync_tree_selection_from_editor()

    def _on_editor_modified(self, _event=None) -> None:
        if not self.editor_text.edit_modified():
            return
        self._hide_editor_macro_tooltip()
        if self.current_edit_path is not None:
            self.file_buffers[self.current_edit_path] = self.editor_text.get("1.0", "end-1c")
        self.editor_text.edit_modified(False)
        self._schedule_editor_update()

    def _on_editor_cursor_changed(self, _event=None) -> None:
        keysym = getattr(_event, "keysym", "")
        char = getattr(_event, "char", "")
        self._hide_editor_macro_tooltip()
        popup_open = bool(
            self.editor_completion_popup is not None
            and self.editor_completion_popup.winfo_viewable()
        )
        should_open = False
        if self.editor_auto_complete_var.get():
            if char == ".":
                should_open = True
            elif popup_open and (char.isalnum() or char == "_" or keysym in {"BackSpace", "Delete"}):
                should_open = True
        self._editor_completion_should_open = should_open
        self._schedule_editor_update()
        navigation_keys = {
            "Up",
            "Down",
            "Left",
            "Right",
            "Home",
            "End",
            "Prior",
            "Next",
            "Return",
            "KP_Enter",
        }
        if keysym in navigation_keys:
            self._schedule_tree_sync_from_editor(180)

    def _on_editor_mouse_press(self, _event=None) -> None:
        self._editor_mouse_interacting_until = time.monotonic() + 0.4
        self._cancel_tree_sync_from_editor()
        self._hide_editor_macro_tooltip()
        self._hide_editor_completion()

    def _on_editor_clicked(self, _event=None) -> None:
        self._editor_mouse_interacting_until = time.monotonic() + 0.35
        self._cancel_tree_sync_from_editor()
        self._editor_completion_should_open = False
        self._schedule_editor_update()

    def _on_editor_mouse_motion(self, event=None) -> None:
        if event is None or not self.editor_macro_hover_var.get():
            return
        context = self._editor_macro_hover_context(event)
        if context is None:
            self._hide_editor_macro_tooltip()
            return
        token, _text = context
        if self._editor_macro_hover_token == token and self.editor_macro_tooltip is not None:
            x = self.editor_text.winfo_rootx() + event.x + 16
            y = self.editor_text.winfo_rooty() + event.y + 20
            self.editor_macro_tooltip.geometry("+{}+{}".format(x, y))
            return
        self._hide_editor_macro_tooltip()
        self._editor_macro_hover_token = token
        self._editor_macro_hover_event = event
        self._editor_macro_hover_job = self.root.after(650, self._show_editor_macro_tooltip)

    def _on_editor_mouse_leave(self, _event=None) -> None:
        self._hide_editor_macro_tooltip()

    def _show_editor_completion_on_demand(self, _event=None) -> str:
        self._editor_completion_forced = True
        self._schedule_editor_update()
        return "break"

    def _update_editor_gutter(self) -> None:
        line_count = int(self.editor_text.index("end-1c").split(".")[0])
        gutter_text = "".join("{}\n".format(line_no) for line_no in range(1, max(1, line_count) + 1))
        self.editor_gutter.configure(state=self.tk.NORMAL)
        self.editor_gutter.delete("1.0", "end")
        self.editor_gutter.insert("1.0", gutter_text)
        self.editor_gutter.configure(state=self.tk.DISABLED)
        self.editor_gutter.yview_moveto(self.editor_text.yview()[0])

    def _highlight_current_editor_line(self) -> None:
        self.editor_text.tag_remove("current_line", "1.0", "end")
        line = self.editor_text.index("insert").split(".")[0]
        self.editor_text.tag_add("current_line", "{}.0".format(line), "{}.0 lineend".format(line))

    def _highlight_matching_bracket(self) -> None:
        self.editor_text.tag_remove("match_bracket", "1.0", "end")
        full_text = self.editor_text.get("1.0", "end-1c")
        if not full_text:
            return

        cursor_offset = len(self.editor_text.get("1.0", "insert"))
        candidate_offsets = []
        if 0 <= cursor_offset - 1 < len(full_text):
            candidate_offsets.append(cursor_offset - 1)
        if 0 <= cursor_offset < len(full_text):
            candidate_offsets.append(cursor_offset)

        pairs = {"(": ")", "[": "]", "{": "}", ")": "(", "]": "[", "}": "{"}
        openings = "([{"
        closings = ")]}"

        for offset in candidate_offsets:
            char = full_text[offset]
            if char not in pairs:
                continue
            if char in openings:
                match_offset = self._find_matching_bracket(full_text, offset, char, pairs[char], forward=True)
            else:
                match_offset = self._find_matching_bracket(full_text, offset, pairs[char], char, forward=False)
            if match_offset is None:
                continue
            start_a = "1.0+{}c".format(offset)
            end_a = "1.0+{}c".format(offset + 1)
            start_b = "1.0+{}c".format(match_offset)
            end_b = "1.0+{}c".format(match_offset + 1)
            self.editor_text.tag_add("match_bracket", start_a, end_a)
            self.editor_text.tag_add("match_bracket", start_b, end_b)
            return

    def _find_matching_bracket(
        self,
        text: str,
        start_offset: int,
        open_char: str,
        close_char: str,
        forward: bool,
    ) -> Optional[int]:
        depth = 0
        if forward:
            for offset in range(start_offset, len(text)):
                char = text[offset]
                if char == open_char:
                    depth += 1
                elif char == close_char:
                    depth -= 1
                    if depth == 0:
                        return offset
            return None
        for offset in range(start_offset, -1, -1):
            char = text[offset]
            if char == close_char:
                depth += 1
            elif char == open_char:
                depth -= 1
                if depth == 0:
                    return offset
        return None

    def _on_search_changed(self, _event=None) -> None:
        if self._search_update_job is not None:
            try:
                self.root.after_cancel(self._search_update_job)
            except Exception:
                pass
        self._search_update_job = self.root.after(60, self._run_search_update)

    def _run_search_update(self) -> None:
        self._search_update_job = None
        self._update_search_highlight()

    def _update_search_highlight(self) -> None:
        self.editor_text.tag_remove("search_match", "1.0", "end")
        self.editor_text.tag_remove("search_current", "1.0", "end")
        query = self._effective_editor_search_query()
        if not query:
            return
        start = "1.0"
        first_match = None
        while True:
            match_start = self.editor_text.search(query, start, stopindex="end", nocase=True)
            if not match_start:
                break
            match_end = "{}+{}c".format(match_start, len(query))
            self.editor_text.tag_add("search_match", match_start, match_end)
            if first_match is None:
                first_match = (match_start, match_end)
            start = match_end
        insert_index = self.editor_text.index("insert")
        current_start = self.editor_text.search(query, insert_index, stopindex="end", nocase=True)
        if not current_start:
            current_start = self.editor_text.search(query, "1.0", stopindex=insert_index, nocase=True)
        if current_start:
            current_end = "{}+{}c".format(current_start, len(query))
            self.editor_text.tag_add("search_current", current_start, current_end)

    def _goto_search_match(self, forward: bool = True) -> None:
        query = self._effective_editor_search_query()
        if not query:
            return
        insert_index = self.editor_text.index("insert")
        if forward:
            match_start = self.editor_text.search(query, "{}+1c".format(insert_index), stopindex="end", nocase=True)
            if not match_start:
                match_start = self.editor_text.search(query, "1.0", stopindex="end", nocase=True)
        else:
            match_start = self.editor_text.search(query, insert_index, stopindex="1.0", backwards=True, nocase=True)
            if not match_start:
                match_start = self.editor_text.search(query, "end-1c", stopindex="1.0", backwards=True, nocase=True)
        if not match_start:
            return
        match_end = "{}+{}c".format(match_start, len(query))
        self.editor_text.mark_set("insert", match_start)
        self.editor_text.tag_remove("sel", "1.0", "end")
        self.editor_text.tag_add("sel", match_start, match_end)
        self.editor_text.see(match_start)
        self._update_search_highlight()

    def _indent_selection(self, _event=None) -> str:
        ranges = self.editor_text.tag_ranges("sel")
        if ranges:
            start_line = int(str(ranges[0]).split(".")[0])
            end_line = int(str(ranges[1]).split(".")[0])
            if str(ranges[1]).endswith(".0") and end_line > start_line:
                end_line -= 1
            for line_no in range(start_line, end_line + 1):
                self.editor_text.insert("{}.0".format(line_no), "    ")
        else:
            self.editor_text.insert("insert", "    ")
        self._schedule_editor_update()
        return "break"

    def _unindent_selection(self, _event=None) -> str:
        ranges = self.editor_text.tag_ranges("sel")
        if ranges:
            start_line = int(str(ranges[0]).split(".")[0])
            end_line = int(str(ranges[1]).split(".")[0])
            if str(ranges[1]).endswith(".0") and end_line > start_line:
                end_line -= 1
        else:
            start_line = end_line = int(self.editor_text.index("insert").split(".")[0])
        for line_no in range(start_line, end_line + 1):
            line_start = "{}.0".format(line_no)
            line_text = self.editor_text.get(line_start, "{}.0 lineend".format(line_no))
            if line_text.startswith("    "):
                self.editor_text.delete(line_start, "{}.4".format(line_no))
            elif line_text.startswith("\t"):
                self.editor_text.delete(line_start, "{}.1".format(line_no))
        self._schedule_editor_update()
        return "break"

    def _on_editor_tab(self, _event=None) -> str:
        if self.editor_completion_popup is not None:
            return self._on_editor_completion_accept()
        return self._indent_selection()

    def _editor_language(self) -> str:
        if self.current_edit_path is None:
            return ""
        suffix = self.current_edit_path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            return "yaml"
        if suffix == ".plc":
            return "plc"
        if suffix in {".cmd", ".script", ".iocsh"}:
            return "startup"
        return ""

    def _highlight_editor_syntax(self) -> None:
        for tag_name in ("syntax_comment", "syntax_string", "syntax_macro", "syntax_keyword", "syntax_number", "syntax_key"):
            self.editor_text.tag_remove(tag_name, "1.0", "end")

        language = self._editor_language()
        if not language:
            return

        text = self.editor_text.get("1.0", "end-1c")
        if not text:
            return

        for match in re.finditer(r"#[^\n]*", text):
            self.editor_text.tag_add("syntax_comment", "1.0+{}c".format(match.start()), "1.0+{}c".format(match.end()))
        for match in re.finditer(r'"[^"\n]*"|\'[^\'\n]*\'', text):
            self.editor_text.tag_add("syntax_string", "1.0+{}c".format(match.start()), "1.0+{}c".format(match.end()))
        for match in re.finditer(r"\$\{[^}\n]+\}|\$\([^\)\n]+\)", text):
            self.editor_text.tag_add("syntax_macro", "1.0+{}c".format(match.start()), "1.0+{}c".format(match.end()))
        for match in re.finditer(r"\b(?:0x[0-9A-Fa-f]+|\d+(?:\.\d+)?)\b", text):
            self.editor_text.tag_add("syntax_number", "1.0+{}c".format(match.start()), "1.0+{}c".format(match.end()))

        if language == "yaml":
            for match in re.finditer(r"(?m)^(\s*[- ]*)?([A-Za-z_][A-Za-z0-9_-]*)(?=\s*:)", text):
                self.editor_text.tag_add(
                    "syntax_key",
                    "1.0+{}c".format(match.start(2)),
                    "1.0+{}c".format(match.end(2)),
                )
            return

        if language == "plc":
            keywords = (
                "VAR",
                "END_VAR",
                "IF",
                "THEN",
                "ELSE",
                "ELSIF",
                "END_IF",
                "FUNCTION",
                "FUNCTION_BLOCK",
                "PROGRAM",
                "TRUE",
                "FALSE",
            )
        else:
            keywords = (
                "epicsEnvSet",
                "epicsEnvUnset",
                "iocshLoad",
                "runScript",
                "require",
                "dbLoadRecords",
                "dbLoadDatabase",
                "dbLoadTemplate",
                "ecmcConfig",
                "ecmcConfigOrDie",
                "ecmcFileExist",
                "system",
                "on",
            )
        keyword_pattern = r"\b(?:{})\b".format("|".join(re.escape(keyword) for keyword in keywords))
        for match in re.finditer(keyword_pattern, text, re.IGNORECASE if language == "plc" else 0):
            self.editor_text.tag_add("syntax_keyword", "1.0+{}c".format(match.start()), "1.0+{}c".format(match.end()))

    def _startup_slave_hw_desc_map(self) -> Dict[int, str]:
        mapping: Dict[int, str] = {}
        if self.latest_startup_tree is None:
            return mapping
        for file_node in self.latest_startup_tree.files:
            for obj in file_node.objects:
                if obj.kind != "slave" or obj.slave_id is None:
                    continue
                hw_desc = dict(obj.details).get("HW_DESC", "").strip()
                if hw_desc:
                    mapping[obj.slave_id] = hw_desc
        return mapping

    def _startup_known_macro_map(self) -> Dict[str, str]:
        macro_map: Dict[str, str] = {}
        if self.latest_startup_tree is None:
            return macro_map

        for file_node in self.latest_startup_tree.files:
            for obj in file_node.objects:
                if obj.kind == "macro":
                    name = obj.title.replace("Macro ", "", 1).strip()
                    summary = obj.summary
                    if "=" in summary:
                        macro_map[name] = summary.split("=", 1)[1].strip()
                    continue
                if obj.kind == "require":
                    for key, value in obj.linked_file_details:
                        macro_map[key] = value
        return macro_map

    def _linked_object_for_current_editor(self) -> Optional[StartupObject]:
        if self.current_edit_path is None or self.latest_startup_tree is None:
            return None
        if self.current_edit_linked_object_key is not None:
            target_key = self.current_edit_linked_object_key
            for file_node in self.latest_startup_tree.files:
                for obj in file_node.objects:
                    if self._object_tree_key(obj) == target_key:
                        return obj
        target = self.current_edit_path.resolve()
        for file_node in self.latest_startup_tree.files:
            for obj in file_node.objects:
                if obj.linked_file is not None and obj.linked_file.resolve() == target:
                    return obj
        return None

    def _editor_known_macro_map(self) -> Dict[str, str]:
        macro_map = self._startup_known_macro_map()
        linked_obj = self._linked_object_for_current_editor()
        if linked_obj is None:
            return macro_map
        for key, value in linked_obj.linked_file_details:
            macro_map[key] = value
        for key, value in linked_obj.command_details:
            if key not in macro_map:
                macro_map[key] = value
        return macro_map

    def _resolve_editor_numeric_token(self, token: str, macro_map: Dict[str, str]) -> Optional[int]:
        cleaned = token.strip()
        if re.fullmatch(r"\d+", cleaned):
            return int(cleaned)
        expanded = _expand_text_macros(cleaned, macro_map).strip()
        if re.fullmatch(r"\d+", expanded):
            return int(expanded)
        return None

    def _resolve_editor_ec_path_base(
        self, text: str, macro_map: Dict[str, str]
    ) -> Optional[Tuple[int, int, str]]:
        cleaned = _strip_wrapper_pairs(_normalize_value(text))
        if not cleaned:
            return None
        expanded = _expand_text_macros(cleaned, macro_map).strip()
        match = re.search(r"\bec(?P<master>\d+)\.s(?P<slave>\d+)\b", expanded)
        if match is None:
            return None
        master_id = _parse_int_value(match.group("master"))
        slave_id = _parse_int_value(match.group("slave"))
        if master_id is None or slave_id is None:
            return None
        return master_id, slave_id, match.group(0)

    def _editor_plc_ec_aliases(self, macro_map: Dict[str, str]) -> Dict[str, Tuple[int, int, str]]:
        if self._editor_language() != "plc":
            return {}
        text = self.editor_text.get("1.0", "end-1c")
        aliases: Dict[str, Tuple[int, int, str]] = {}
        in_var_block = False
        for raw_line in text.splitlines():
            line = _strip_inline_comment(raw_line).strip()
            if not line:
                continue
            upper_line = line.upper()
            if upper_line == "VAR":
                in_var_block = True
                continue
            if upper_line == "END_VAR":
                in_var_block = False
                continue
            if not in_var_block:
                continue
            declaration_match = re.match(
                r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<value>[^;]+?)\s*;?\s*$",
                line,
            )
            if declaration_match is None:
                continue
            resolved = self._resolve_editor_ec_path_base(declaration_match.group("value"), macro_map)
            if resolved is None:
                continue
            aliases[declaration_match.group("name")] = resolved
        return aliases

    def _build_editor_completion_context(
        self,
        master_id: int,
        slave_id: int,
        entry_prefix: str,
        start_index: str,
        end_index: str,
        source_label: str = "",
    ) -> Optional[Dict[str, object]]:
        slave_hw_desc = self._startup_slave_hw_desc_map().get(slave_id, "")
        if not slave_hw_desc:
            return None

        available_entries = sorted(self.inventory.hardware_entries.get(slave_hw_desc, set()))
        if not available_entries:
            return None

        matches = [entry for entry in available_entries if entry.lower().startswith(entry_prefix.lower())]
        if not matches:
            return None

        return {
            "master_id": master_id,
            "slave_id": slave_id,
            "hw_desc": slave_hw_desc,
            "entry_prefix": entry_prefix,
            "matches": matches,
            "start_index": start_index,
            "end_index": end_index,
            "source_label": source_label,
        }

    def _current_editor_completion_context(self) -> Optional[Dict[str, object]]:
        if self._editor_language() not in {"yaml", "plc"} or self.current_edit_path is None:
            return None

        insert_index = self.editor_text.index("insert")
        line_no = int(insert_index.split(".")[0])
        line_to_cursor = self.editor_text.get("{}.0".format(line_no), insert_index)
        macro_map = self._editor_known_macro_map()
        token_pattern = r"(?:\d+|\$\{[A-Za-z0-9_]+(?:=[^}]*)?\}|\$\([A-Za-z0-9_]+(?:=[^\)]*)?\))"
        match = re.search(
            r"ec(?P<master>{token})\.s(?P<slave>{token})\.(?P<entry>[A-Za-z_][A-Za-z0-9_]*)?$".format(token=token_pattern),
            line_to_cursor,
        )
        if match is not None:
            master_id = self._resolve_editor_numeric_token(match.group("master"), macro_map)
            slave_id = self._resolve_editor_numeric_token(match.group("slave"), macro_map)
            if master_id is None or slave_id is None:
                return None
            entry_prefix = match.group("entry") or ""
            start_col = len(line_to_cursor) - len(entry_prefix)
            start_index = "{}.{}".format(line_no, start_col)
            return self._build_editor_completion_context(master_id, slave_id, entry_prefix, start_index, insert_index)

        if self._editor_language() == "plc":
            alias_match = re.search(r"(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\.(?P<entry>[A-Za-z_][A-Za-z0-9_]*)?$", line_to_cursor)
            if alias_match is None:
                return None
            alias_name = alias_match.group("alias")
            alias_info = self._editor_plc_ec_aliases(macro_map).get(alias_name)
            if alias_info is None:
                return None
            master_id, slave_id, resolved_path = alias_info
            entry_prefix = alias_match.group("entry") or ""
            start_col = len(line_to_cursor) - len(entry_prefix)
            start_index = "{}.{}".format(line_no, start_col)
            return self._build_editor_completion_context(
                master_id,
                slave_id,
                entry_prefix,
                start_index,
                insert_index,
                source_label="{} -> {}".format(alias_name, resolved_path),
            )
        return None

    def _ensure_editor_completion_popup(self) -> None:
        if self.editor_completion_popup is not None and self.editor_completion_popup.winfo_exists():
            return

        popup = self.tk.Toplevel(self.root)
        popup.withdraw()
        popup.overrideredirect(True)
        popup.transient(self.root)
        popup.configure(background="#d8c7a3")

        frame = self.ttk.Frame(popup, padding=1)
        frame.pack(fill=self.tk.BOTH, expand=True)

        label = self.ttk.Label(frame, textvariable=self.editor_completion_label_var, anchor=self.tk.W)
        label.pack(fill=self.tk.X, padx=4, pady=(4, 2))

        listbox = self.tk.Listbox(
            frame,
            width=44,
            height=8,
            exportselection=False,
            activestyle="none",
        )
        listbox.pack(fill=self.tk.BOTH, expand=True, padx=4, pady=(0, 4))
        listbox.bind("<ButtonRelease-1>", self._accept_editor_completion_from_listbox)

        self.editor_completion_popup = popup
        self.editor_completion_listbox = listbox

    def _show_editor_completion(self, context: Dict[str, object]) -> None:
        self._ensure_editor_completion_popup()
        if self.editor_completion_popup is None or self.editor_completion_listbox is None:
            return

        matches = list(context["matches"])
        previous_value = None
        if self.editor_completion_popup.winfo_viewable():
            selection = self.editor_completion_listbox.curselection()
            if selection:
                previous_value = self.editor_completion_listbox.get(selection[0])
        self.editor_completion_candidates = matches
        self.editor_completion_range = (str(context["start_index"]), str(context["end_index"]))
        label = "ec{}.s{} | {}".format(context["master_id"], context["slave_id"], context["hw_desc"])
        source_label = str(context.get("source_label", "") or "").strip()
        if source_label:
            label = "{} | {}".format(source_label, label)
        self.editor_completion_label_var.set(label)

        self.editor_completion_listbox.delete(0, self.tk.END)
        for item in matches:
            self.editor_completion_listbox.insert(self.tk.END, item)
        self.editor_completion_listbox.configure(height=min(8, max(1, len(matches))))
        if matches:
            selected_index = 0
            if previous_value in matches:
                selected_index = matches.index(previous_value)
            self.editor_completion_listbox.selection_clear(0, self.tk.END)
            self.editor_completion_listbox.selection_set(selected_index)
            self.editor_completion_listbox.activate(selected_index)
            self.editor_completion_listbox.see(selected_index)

        bbox = self.editor_text.bbox("insert")
        if bbox is None:
            self._hide_editor_completion()
            return

        x = self.editor_text.winfo_rootx() + bbox[0]
        y = self.editor_text.winfo_rooty() + bbox[1] + bbox[3] + 2
        self.editor_completion_popup.geometry("+{}+{}".format(x, y))
        self.editor_completion_popup.deiconify()
        self.editor_completion_popup.lift()

    def _hide_editor_completion(self) -> None:
        self.editor_completion_candidates = []
        self.editor_completion_range = None
        if self.editor_completion_popup is not None and self.editor_completion_popup.winfo_exists():
            self.editor_completion_popup.withdraw()

    def _update_editor_completion(self, force: bool = False) -> None:
        context = self._current_editor_completion_context()
        if context is None:
            self._hide_editor_completion()
            return
        if force or self.editor_completion_popup is not None and self.editor_completion_popup.winfo_viewable():
            self._show_editor_completion(context)
            return
        self._hide_editor_completion()

    def _move_editor_completion_selection(self, delta: int) -> None:
        if self.editor_completion_listbox is None or not self.editor_completion_candidates:
            return
        selection = self.editor_completion_listbox.curselection()
        current_index = selection[0] if selection else 0
        next_index = max(0, min(len(self.editor_completion_candidates) - 1, current_index + delta))
        self.editor_completion_listbox.selection_clear(0, self.tk.END)
        self.editor_completion_listbox.selection_set(next_index)
        self.editor_completion_listbox.activate(next_index)
        self.editor_completion_listbox.see(next_index)

    def _on_editor_completion_down(self, _event=None) -> Optional[str]:
        if self.editor_completion_popup is None or not self.editor_completion_popup.winfo_viewable():
            return None
        self._move_editor_completion_selection(1)
        return "break"

    def _on_editor_completion_up(self, _event=None) -> Optional[str]:
        if self.editor_completion_popup is None or not self.editor_completion_popup.winfo_viewable():
            return None
        self._move_editor_completion_selection(-1)
        return "break"

    def _insert_editor_completion(self, value: str) -> None:
        if self.editor_completion_range is None:
            return
        start_index, end_index = self.editor_completion_range
        self.editor_text.delete(start_index, end_index)
        self.editor_text.insert(start_index, value)
        self.editor_text.mark_set("insert", "{}+{}c".format(start_index, len(value)))
        if self.current_edit_path is not None:
            self.file_buffers[self.current_edit_path] = self.editor_text.get("1.0", "end-1c")
        self._hide_editor_completion()
        self._schedule_editor_update()

    def _on_editor_completion_accept(self, _event=None) -> Optional[str]:
        if self.editor_completion_popup is None or not self.editor_completion_popup.winfo_viewable():
            return None
        if self.editor_completion_listbox is None:
            return "break"
        selection = self.editor_completion_listbox.curselection()
        if not selection:
            if not self.editor_completion_candidates:
                self._hide_editor_completion()
                return "break"
            selection = (0,)
        self._insert_editor_completion(self.editor_completion_candidates[selection[0]])
        return "break"

    def _accept_editor_completion_from_listbox(self, _event=None) -> str:
        self._on_editor_completion_accept()
        self.editor_text.focus_set()
        return "break"

    def _on_editor_completion_escape(self, _event=None) -> Optional[str]:
        if self.editor_completion_popup is None or not self.editor_completion_popup.winfo_viewable():
            return None
        self._hide_editor_completion()
        return "break"

    def _set_editor_tree_highlight(self, item_id: str) -> None:
        if self._editor_tree_highlight_item == item_id:
            try:
                self.startup_tree.see(item_id)
            except Exception:
                self._editor_tree_highlight_item = None
            return
        if self._editor_tree_highlight_item:
            try:
                previous_tags = list(self.startup_tree.item(self._editor_tree_highlight_item, "tags") or ())
                if "editor-current" in previous_tags:
                    previous_tags.remove("editor-current")
                    self.startup_tree.item(self._editor_tree_highlight_item, tags=tuple(previous_tags))
            except Exception:
                pass
        current_tags = list(self.startup_tree.item(item_id, "tags") or ())
        if "editor-current" not in current_tags:
            current_tags.append("editor-current")
            self.startup_tree.item(item_id, tags=tuple(current_tags))
        self._editor_tree_highlight_item = item_id
        self.startup_tree.see(item_id)

    def _populate_param_tree_for_entry(self, entry_type: str, payload) -> None:
        self._update_selection_badges((entry_type, payload))
        self.selection_header_var.set(self._selection_header_text((entry_type, payload)))
        if entry_type == "file":
            file_node = payload
            rows = [
                ("PATH", self._relative_display(file_node.path)),
                ("SOURCE_LINE", str(file_node.parent_line)),
                ("OBJECTS", str(len(file_node.objects))),
            ]
            if file_node.parent_path is not None:
                rows.insert(1, ("PARENT_FILE", self._relative_display(file_node.parent_path)))
            self._populate_param_tree(rows)
            self._refresh_quick_edit_panel((entry_type, payload))
            return
        if entry_type == "category-group":
            file_path, category, count = payload
            self._populate_param_tree(
                [
                    ("TYPE", "group"),
                    ("CATEGORY", category),
                    ("FILE", self._relative_display(file_path)),
                    ("OBJECTS", str(count)),
                ]
            )
            self._refresh_quick_edit_panel(None)
            return

        obj = payload
        rows = [
            ("TYPE", obj.kind),
            ("TITLE", obj.title),
            ("SOURCE", "{}:{}".format(self._relative_display(obj.source), obj.line)),
        ]
        seen_keys = set(key for key, _value in rows)
        if obj.linked_file is not None:
            rows.append(("FILE", self._relative_display(obj.linked_file)))
            seen_keys.add("FILE")
        for key, value in obj.details:
            if key in seen_keys:
                continue
            rows.append((key, value))
            seen_keys.add(key)
        self._populate_param_tree(rows)
        self._refresh_quick_edit_panel((entry_type, payload))

    def _sync_tree_selection_from_editor(self) -> None:
        if self.current_edit_path is None:
            return
        current_path = self.current_edit_path.resolve()
        line = int(self.editor_text.index("insert").split(".")[0])
        location = (current_path, line)
        if self._last_editor_sync_location == location:
            return
        self._last_editor_sync_location = location

        object_key = None
        for candidate_key in self.object_tree_items:
            source_path, source_line, _kind, _title = candidate_key
            if source_path == current_path and source_line == line:
                object_key = candidate_key
                break
        if object_key is not None:
            item_id = self.object_tree_items[object_key]
            self._set_editor_tree_highlight(item_id)
            entry = self.startup_item_map.get(item_id)
            if entry is not None:
                self._populate_param_tree_for_entry(*entry)
                self._refresh_context_panel(entry)
            return

        if self.current_edit_linked_object_key is not None:
            linked_item = self.linked_file_item_by_object_key.get(self.current_edit_linked_object_key)
            if linked_item is not None:
                self._set_editor_tree_highlight(linked_item)
                entry = self.startup_item_map.get(linked_item)
                if entry is not None:
                    self._populate_param_tree_for_entry(*entry)
                    self._refresh_context_panel(entry)
                return

        linked_items = self.linked_file_tree_items.get(current_path, [])
        if len(linked_items) == 1:
            item_id = linked_items[0]
            self._set_editor_tree_highlight(item_id)
            entry = self.startup_item_map.get(item_id)
            if entry is not None:
                self._populate_param_tree_for_entry(*entry)
                self._refresh_context_panel(entry)
            return

        file_item = self.file_tree_items.get(current_path)
        if file_item is not None:
            self._set_editor_tree_highlight(file_item)
            entry = self.startup_item_map.get(file_item)
            if entry is not None:
                self._populate_param_tree_for_entry(*entry)
                self._refresh_context_panel(entry)

    def _browse_startup(self) -> None:
        from tkinter import filedialog

        current = self.startup_var.get().strip() or "."
        picked = filedialog.askopenfilename(
            title="Select ecmc startup file",
            initialdir=str(Path(current).expanduser().parent if current else Path.cwd()),
            filetypes=[
                ("Startup files", "*.cmd *.script *.iocsh"),
                ("All files", "*"),
            ],
        )
        if picked:
            self.startup_var.set(picked)
            self._load_startup_from_entry()

    def _load_startup_from_entry(self) -> None:
        from tkinter import messagebox

        value = self.startup_var.get().strip()
        if not value:
            messagebox.showerror("Missing startup file", "Select a startup file first.")
            return

        path = Path(value).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        self._open_startup(path, validate_now=False)

    def _open_startup(self, path: Path, validate_now: bool) -> None:
        from tkinter import messagebox

        if not path.exists():
            messagebox.showerror("Startup file missing", f"File not found:\n{path}")
            return

        self.startup_var.set(str(path))
        self._open_file_in_editor(path)
        self._refresh_startup_tree()
        if validate_now:
            self._validate_current_project()

    def _remember_current_buffer(self) -> None:
        if self.current_edit_path is None:
            return
        self.file_buffers[self.current_edit_path] = self.editor_text.get("1.0", "end-1c")

    def _open_file_in_editor(
        self,
        path: Path,
        line: Optional[int] = None,
        linked_object_key: Optional[Tuple[Path, int, str, str]] = None,
    ) -> None:
        from tkinter import messagebox

        resolved = path.resolve()
        self._remember_current_buffer()

        if resolved in self.file_buffers:
            content = self.file_buffers[resolved]
        elif resolved.exists():
            content = _read_text(resolved)
            self.file_buffers[resolved] = content
        else:
            messagebox.showerror("File missing", f"Cannot open missing file:\n{resolved}")
            return

        self.current_edit_path = resolved
        self._last_editor_sync_location = None
        self.current_edit_linked_object_key = linked_object_key
        self.editor_file_var.set(str(resolved))
        self._set_editor_content(content, line=line)
        self._highlight_tree_target_line(line)
        self._sync_file_browser_selection(resolved)
        self._refresh_context_panel()
        self._refresh_resolved_preview()
        self.status_var.set(f"Opened {resolved.name}")

    def _highlight_editor_line(self, line: Optional[int]) -> None:
        if line is None or line <= 0:
            return
        start = f"{line}.0"
        self.editor_text.mark_set("insert", start)
        self.editor_text.see(start)
        self._highlight_current_editor_line()

    def _highlight_tree_target_line(self, line: Optional[int]) -> None:
        self.editor_text.tag_remove("tree_target_line", "1.0", "end")
        self.editor_gutter.tag_remove("tree_target_line", "1.0", "end")
        if self._editor_target_line_clear_job is not None:
            try:
                self.root.after_cancel(self._editor_target_line_clear_job)
            except Exception:
                pass
        self._editor_target_line_clear_job = None
        if line is None or line <= 0:
            return
        start = f"{line}.0"
        end = f"{line}.0 lineend +1c"
        self.editor_text.tag_add("tree_target_line", start, end)
        self.editor_gutter.tag_add("tree_target_line", start, end)
        self.editor_text.tag_raise("tree_target_line")
        self.editor_gutter.tag_raise("tree_target_line")
        self._editor_target_line_clear_job = self.root.after(
            2400,
            lambda: (
                self.editor_text.tag_remove("tree_target_line", "1.0", "end"),
                self.editor_gutter.tag_remove("tree_target_line", "1.0", "end"),
            ),
        )

    def _save_current_file(self) -> None:
        from tkinter import messagebox

        if self.current_edit_path is None:
            messagebox.showerror("No file open", "There is no file loaded in the editor.")
            return

        self._remember_current_buffer()
        saved_count = self._save_buffered_paths([self.current_edit_path])
        if saved_count:
            self.status_var.set(f"Saved {self.current_edit_path.name}")
            self._refresh_startup_tree()
        else:
            self.status_var.set(f"No changes in {self.current_edit_path.name}")

    def _on_ctrl_s(self, _event=None) -> str:
        self._save_current_file()
        return "break"

    def _save_buffered_paths(self, paths: List[Path]) -> int:
        saved_count = 0
        for path in paths:
            resolved = path.resolve()
            if resolved not in self.file_buffers:
                continue
            content = self.file_buffers[resolved]
            current_disk = _read_text(resolved) if resolved.exists() else None
            if current_disk == content:
                continue
            resolved.write_text(content, encoding="utf-8")
            saved_count += 1
        return saved_count

    def _save_all_files(self) -> None:
        self._remember_current_buffer()
        paths = sorted(self.file_buffers.keys(), key=lambda path: str(path))
        saved_count = self._save_buffered_paths(paths)
        if saved_count == 0:
            self.status_var.set("No buffered file changes to save")
        elif saved_count == 1:
            self.status_var.set("Saved 1 buffered file")
        else:
            self.status_var.set("Saved {} buffered files".format(saved_count))
        if saved_count:
            self._refresh_startup_tree()

    def _reload_current_from_disk(self) -> None:
        from tkinter import messagebox

        if self.current_edit_path is None:
            messagebox.showerror("No file open", "There is no file loaded in the editor.")
            return
        if not self.current_edit_path.exists():
            messagebox.showerror("File missing", f"File not found:\n{self.current_edit_path}")
            return

        content = _read_text(self.current_edit_path)
        self.file_buffers[self.current_edit_path] = content
        self._set_editor_content(content)
        self._refresh_resolved_preview()
        self.status_var.set(f"Reloaded {self.current_edit_path.name}")

    def _refresh_startup_tree(self) -> None:
        from tkinter import messagebox

        startup_value = self.startup_var.get().strip()
        if not startup_value:
            return

        startup_path = Path(startup_value).expanduser()
        if not startup_path.is_absolute():
            startup_path = startup_path.resolve()
        if not startup_path.exists():
            messagebox.showerror("Startup file missing", f"File not found:\n{startup_path}")
            return

        self._remember_current_buffer()
        if startup_path not in self.file_buffers:
            self.file_buffers[startup_path] = _read_text(startup_path)
        startup_text = self.file_buffers[startup_path]
        tree_state = self._capture_startup_tree_state()

        self.latest_startup_tree = build_startup_tree(
            startup_path=startup_path,
            startup_text=startup_text,
            inventory=self.inventory,
            buffer_lookup=self.file_buffers,
        )
        self._populate_startup_tree(startup_path, self.latest_startup_tree, tree_state=tree_state)
        self._populate_file_browser(startup_path)
        self._refresh_context_panel()
        self._refresh_resolved_preview()
        self.status_var.set(f"Loaded {startup_path.name}")

    def _validate_current_project(self) -> None:
        from tkinter import messagebox

        startup_value = self.startup_var.get().strip()
        if not startup_value:
            messagebox.showerror("Missing startup file", "Select a startup file first.")
            return

        startup_path = Path(startup_value).expanduser()
        if not startup_path.is_absolute():
            startup_path = startup_path.resolve()
        if not startup_path.exists():
            messagebox.showerror("Startup file missing", f"File not found:\n{startup_path}")
            return

        self._remember_current_buffer()
        if startup_path not in self.file_buffers:
            self.file_buffers[startup_path] = _read_text(startup_path)
        startup_text = self.file_buffers[startup_path]

        result = validate_project(
            startup_path=startup_path,
            startup_text=startup_text,
            inventory=self.inventory,
            buffer_lookup=self.file_buffers,
        )
        self.latest_result = result
        self._refresh_startup_tree()

        error_count = sum(1 for issue in result.issues if issue.severity == "error")
        warning_count = sum(1 for issue in result.issues if issue.severity == "warning")
        self.status_var.set(
            f"Validated {startup_path.name}: {error_count} error(s), {warning_count} warning(s), "
            f"{len(result.references)} reference(s)"
        )
        self._show_validation_results(startup_path, result)

    def _tree_entry_restore_key(self, item_id: str) -> Optional[Tuple[object, ...]]:
        entry = self.startup_item_map.get(item_id)
        if entry is None:
            return None
        entry_type, payload = entry
        if entry_type == "file":
            return ("file", payload.path.resolve())
        if entry_type == "category-group":
            file_path, category, _count = payload
            return ("category-group", file_path.resolve(), category)
        if not isinstance(payload, StartupObject):
            return None
        object_key = self._object_tree_key(payload)
        if entry_type == "object":
            return ("object", object_key)
        if entry_type in {"detail", "linked-detail"}:
            return (entry_type, object_key, self._tree_text_logical_name(self.startup_tree.item(item_id, "text")))
        if entry_type in {"detail-group", "linked-detail-group", "linked-file"}:
            return (entry_type, object_key)
        return None

    def _capture_startup_tree_state(self) -> Dict[str, object]:
        expanded: Set[Tuple[object, ...]] = set()
        for item_id in self.startup_tree.get_children(""):
            stack = [item_id]
            while stack:
                current = stack.pop()
                if self.startup_tree.item(current, "open"):
                    restore_key = self._tree_entry_restore_key(current)
                    if restore_key is not None:
                        expanded.add(restore_key)
                stack.extend(self.startup_tree.get_children(current))
        selected_key = None
        selected = self.startup_tree.selection()
        if selected:
            selected_key = self._tree_entry_restore_key(selected[0])
        return {"expanded": expanded, "selected": selected_key}

    def _find_tree_item_by_restore_key(self, restore_key: Optional[Tuple[object, ...]]) -> Optional[str]:
        if restore_key is None:
            return None
        for item_id in self.startup_item_map:
            if self._tree_entry_restore_key(item_id) == restore_key:
                return item_id
        return None

    def _populate_startup_tree(
        self,
        startup_path: Path,
        startup_tree: StartupTreeModel,
        tree_state: Optional[Dict[str, object]] = None,
    ) -> None:
        self.startup_item_map.clear()
        self.object_tree_items.clear()
        self.linked_file_tree_items.clear()
        self.linked_file_item_by_object_key.clear()
        self.file_tree_items.clear()
        self._editor_tree_highlight_item = None
        self.startup_tree.delete(*self.startup_tree.get_children(""))
        self.param_tree.delete(*self.param_tree.get_children(""))
        expanded_keys = set(tree_state.get("expanded", set())) if tree_state else set()
        selected_key = tree_state.get("selected") if tree_state else None
        issue_severity_by_path: Dict[Path, str] = {}
        issue_severity_by_object_key: Dict[Tuple[Path, int, str, str], str] = {}
        dirty_state_by_path: Dict[Path, bool] = {}
        filter_kind = self._tree_filter_kind()
        tree_view_mode = self._tree_view_mode()
        tree_sort_mode = self._tree_sort_mode()
        search_query = self._tree_search_query()
        compact_view = bool(self.tree_compact_var.get())
        filtered_view = compact_view or filter_kind != "all" or bool(search_query)
        total_objects = 0
        total_slaves = 0
        total_axes = 0
        total_plcs = 0
        visible_objects = 0
        total_errors = 0
        total_warnings = 0
        total_dirty = 0
        total_missing = 0

        if self.latest_result is not None:
            for issue in self.latest_result.issues:
                if issue.severity not in {"error", "warning"}:
                    continue
                source_path = issue.source.resolve()
                issue_severity_by_path[source_path] = self._worst_status(
                    issue_severity_by_path.get(source_path, ""),
                    issue.severity,
                )
                for file_node in startup_tree.files:
                    for obj in file_node.objects:
                        object_key = self._object_tree_key(obj)
                        if source_path == obj.source.resolve() and issue.line == obj.line:
                            issue_severity_by_object_key[object_key] = self._worst_status(
                                issue_severity_by_object_key.get(object_key, ""),
                                issue.severity,
                            )
                        elif obj.linked_file is not None and source_path == obj.linked_file.resolve():
                            issue_severity_by_object_key[object_key] = self._worst_status(
                                issue_severity_by_object_key.get(object_key, ""),
                                issue.severity,
                            )

        def path_is_dirty(path: Optional[Path]) -> bool:
            if path is None:
                return False
            resolved = path.resolve()
            if resolved in dirty_state_by_path:
                return dirty_state_by_path[resolved]
            if resolved not in self.file_buffers:
                dirty_state_by_path[resolved] = False
                return False
            buffered = self.file_buffers[resolved]
            if not resolved.exists():
                dirty_state_by_path[resolved] = bool(buffered)
                return dirty_state_by_path[resolved]
            dirty_state_by_path[resolved] = _read_text(resolved) != buffered
            return dirty_state_by_path[resolved]

        file_items: Dict[Path, str] = {}
        for file_node in startup_tree.files:
            total_objects += len(file_node.objects)
            total_slaves += sum(1 for obj in file_node.objects if obj.kind == "slave")
            total_axes += sum(1 for obj in file_node.objects if obj.kind == "axis")
            total_plcs += sum(1 for obj in file_node.objects if obj.kind == "plc")
            parent_item = ""
            if file_node.parent_path is not None:
                parent_item = file_items.get(file_node.parent_path, "")
            file_path = file_node.path.resolve()
            file_severity = issue_severity_by_path.get(file_path, "")
            file_dirty = path_is_dirty(file_path)
            if file_severity == "error":
                total_errors += 1
            elif file_severity == "warning":
                total_warnings += 1
            if file_dirty:
                total_dirty += 1
            parent_key_by_key: Dict[Tuple[Path, int, str, str], Optional[Tuple[Path, int, str, str]]] = {}
            visible_object_keys: Set[Tuple[Path, int, str, str]] = set()
            slave_keys: Dict[int, Tuple[Path, int, str, str]] = {}
            axis_keys: Dict[int, Tuple[Path, int, str, str]] = {}
            plc_keys: Dict[int, Tuple[Path, int, str, str]] = {}
            last_plc_key: Optional[Tuple[Path, int, str, str]] = None

            for obj in file_node.objects:
                object_key = self._object_tree_key(obj)
                parent_key = None
                if obj.kind in {"component", "ecsdo", "ecdataitem", "slave_config"} and obj.parent_slave_id is not None:
                    parent_key = slave_keys.get(obj.parent_slave_id)
                elif obj.kind == "encoder" and obj.parent_axis_line is not None:
                    parent_key = axis_keys.get(obj.parent_axis_line)
                elif obj.kind in {"plcvar_analog", "plcvar_binary"}:
                    if obj.parent_plc_id is not None:
                        parent_key = plc_keys.get(obj.parent_plc_id, last_plc_key)
                    else:
                        parent_key = last_plc_key
                parent_key_by_key[object_key] = parent_key

                object_severity = issue_severity_by_object_key.get(object_key, "")
                source_dirty = path_is_dirty(obj.source)
                linked_dirty = path_is_dirty(obj.linked_file) if obj.linked_file is not None else False
                linked_missing = bool(obj.linked_file is not None and not obj.linked_file.exists())
                if object_severity == "error":
                    total_errors += 1
                elif object_severity == "warning":
                    total_warnings += 1
                if source_dirty or linked_dirty:
                    total_dirty += 1
                if linked_missing:
                    total_missing += 1
                if self._object_matches_tree_view(
                    obj,
                    filter_kind,
                    search_query,
                    object_severity,
                    source_dirty,
                    linked_dirty,
                    linked_missing,
                ):
                    visible_object_keys.add(object_key)

                if obj.kind == "slave" and obj.slave_id is not None:
                    slave_keys[obj.slave_id] = object_key
                if obj.kind == "axis":
                    axis_keys[obj.line] = object_key
                if obj.kind == "plc":
                    last_plc_key = object_key
                    if obj.parent_plc_id is not None:
                        plc_keys[obj.parent_plc_id] = object_key
                    else:
                        for key, value in obj.details:
                            if key == "PLC_ID":
                                parsed_plc_id = _parse_int_value(value)
                                if parsed_plc_id is not None:
                                    plc_keys[parsed_plc_id] = object_key
                                break

            for object_key in list(visible_object_keys):
                parent_key = parent_key_by_key.get(object_key)
                while parent_key is not None and parent_key not in visible_object_keys:
                    visible_object_keys.add(parent_key)
                    parent_key = parent_key_by_key.get(parent_key)

            file_badge = self._tree_status_badge(severity=file_severity, dirty=file_dirty)
            file_tags = ("node-file",) + self._tree_status_tags(severity=file_severity, dirty=file_dirty)
            visible_file_objects = sum(1 for obj in file_node.objects if self._object_tree_key(obj) in visible_object_keys)
            visible_objects += visible_file_objects
            file_summary = "{} object(s)".format(len(file_node.objects))
            if filtered_view:
                file_summary = "{} / {} shown".format(visible_file_objects, len(file_node.objects))
            file_item = self.startup_tree.insert(
                parent_item,
                "end",
                text=file_badge + self._prefixed_tree_text("▣", self._relative_display(file_node.path)),
                values=(file_summary,),
                open=filtered_view
                or ("file", file_node.path.resolve()) in expanded_keys
                or file_node.path.resolve() == startup_path.resolve(),
                tags=file_tags,
            )
            file_items[file_node.path] = file_item
            self.file_tree_items[file_node.path.resolve()] = file_item
            self.startup_item_map[file_item] = ("file", file_node)
            category_items: Dict[str, str] = {}
            if tree_view_mode == "objects":
                category_counts: Dict[str, int] = {}
                category_errors: Dict[str, int] = {}
                category_warnings: Dict[str, int] = {}
                category_dirty: Dict[str, int] = {}
                category_missing: Dict[str, int] = {}
                for obj in file_node.objects:
                    object_key = self._object_tree_key(obj)
                    if object_key not in visible_object_keys or parent_key_by_key.get(object_key) is not None:
                        continue
                    category = self._tree_category_for_object(obj)
                    category_counts[category] = category_counts.get(category, 0) + 1
                    object_severity = issue_severity_by_object_key.get(object_key, "")
                    if object_severity == "error":
                        category_errors[category] = category_errors.get(category, 0) + 1
                    elif object_severity == "warning":
                        category_warnings[category] = category_warnings.get(category, 0) + 1
                    object_dirty = path_is_dirty(obj.source) or (
                        obj.linked_file is not None and path_is_dirty(obj.linked_file)
                    )
                    if object_dirty:
                        category_dirty[category] = category_dirty.get(category, 0) + 1
                    if obj.linked_file is not None and not obj.linked_file.exists():
                        category_missing[category] = category_missing.get(category, 0) + 1
                for category in ("Slaves", "Axes", "PLCs", "Macros", "ECMC", "Other"):
                    count = category_counts.get(category, 0)
                    if count <= 0:
                        continue
                    category_error_count = category_errors.get(category, 0)
                    category_warning_count = category_warnings.get(category, 0)
                    category_dirty_count = category_dirty.get(category, 0)
                    category_missing_count = category_missing.get(category, 0)
                    category_badge = self._tree_status_badge(
                        severity="error" if category_error_count else ("warning" if category_warning_count else ""),
                        dirty=category_dirty_count > 0,
                        missing=category_missing_count > 0,
                    )
                    category_tags = ("node-category",) + self._tree_status_tags(
                        severity="error" if category_error_count else ("warning" if category_warning_count else ""),
                        dirty=category_dirty_count > 0,
                        missing=category_missing_count > 0,
                    )
                    category_item = self.startup_tree.insert(
                        file_item,
                        "end",
                        text=category_badge + self._prefixed_tree_text("▸", category),
                        values=(
                            self._tree_category_summary_text(
                                count,
                                category_error_count,
                                category_warning_count,
                                category_dirty_count,
                                category_missing_count,
                            ),
                        ),
                        open=filtered_view
                        or ("category-group", file_node.path.resolve(), category) in expanded_keys,
                        tags=category_tags,
                    )
                    category_items[category] = category_item
                    self.startup_item_map[category_item] = ("category-group", (file_node.path.resolve(), category, count))
            slave_items: Dict[int, str] = {}
            axis_items: Dict[int, str] = {}
            plc_items: Dict[int, str] = {}
            last_plc_item = ""
            top_level_flow_index = 0
            ordered_objects = list(file_node.objects)
            if tree_view_mode == "objects":
                object_by_key = {self._object_tree_key(obj): obj for obj in file_node.objects}

                def top_level_key_for(object_key: Tuple[Path, int, str, str]) -> Tuple[Path, int, str, str]:
                    current_key = object_key
                    parent_key = parent_key_by_key.get(current_key)
                    while parent_key is not None:
                        current_key = parent_key
                        parent_key = parent_key_by_key.get(current_key)
                    return current_key

                top_level_keys = [
                    self._object_tree_key(obj)
                    for obj in file_node.objects
                    if self._object_tree_key(obj) in visible_object_keys and parent_key_by_key.get(self._object_tree_key(obj)) is None
                ]
                sorted_top_level_keys = sorted(
                    top_level_keys,
                    key=lambda key: self._tree_sort_value_for_object(object_by_key[key], tree_sort_mode),
                )
                top_level_rank = {key: index for index, key in enumerate(sorted_top_level_keys)}
                ordered_objects = sorted(
                    file_node.objects,
                    key=lambda obj: (
                        top_level_rank.get(top_level_key_for(self._object_tree_key(obj)), len(sorted_top_level_keys)),
                        self._object_tree_key(obj)[1],
                    ),
                )

            for obj in ordered_objects:
                object_key = self._object_tree_key(obj)
                if object_key not in visible_object_keys:
                    continue
                obj_parent_item = file_item
                if obj.kind in {"component", "ecsdo", "ecdataitem", "slave_config"} and obj.parent_slave_id is not None:
                    obj_parent_item = slave_items.get(obj.parent_slave_id, file_item)
                elif obj.kind == "encoder" and obj.parent_axis_line is not None:
                    obj_parent_item = axis_items.get(obj.parent_axis_line, file_item)
                elif obj.kind in {"plcvar_analog", "plcvar_binary"}:
                    if obj.parent_plc_id is not None:
                        obj_parent_item = plc_items.get(obj.parent_plc_id, last_plc_item or file_item)
                    elif last_plc_item:
                        obj_parent_item = last_plc_item
                if obj_parent_item == file_item and tree_view_mode == "objects":
                    obj_parent_item = category_items.get(self._tree_category_for_object(obj), file_item)
                flow_index = None
                if obj_parent_item == file_item:
                    top_level_flow_index += 1
                    flow_index = top_level_flow_index
                object_severity = issue_severity_by_object_key.get(object_key, "")
                linked_missing = bool(obj.linked_file is not None and not obj.linked_file.exists())
                object_badge = self._tree_status_badge(severity=object_severity, missing=linked_missing)
                object_tags = self._tree_tags_for_object(obj) + self._tree_status_tags(
                    severity=object_severity,
                    missing=linked_missing,
                )
                obj_item = self.startup_tree.insert(
                    obj_parent_item,
                    "end",
                    text=object_badge + self._tree_label_for_object(obj, flow_index),
                    values=(self._tree_summary_for_object(obj),),
                    open=("object", object_key) in expanded_keys,
                    tags=object_tags,
                )
                self.startup_item_map[obj_item] = ("object", obj)
                self.object_tree_items[object_key] = obj_item
                if obj.kind == "slave" and obj.slave_id is not None:
                    slave_items[obj.slave_id] = obj_item
                if obj.kind == "axis":
                    axis_items[obj.line] = obj_item
                if obj.kind == "plc":
                    last_plc_item = obj_item
                    if obj.parent_plc_id is not None:
                        plc_items[obj.parent_plc_id] = obj_item
                    else:
                        for key, value in obj.details:
                            if key == "PLC_ID":
                                parsed_plc_id = _parse_int_value(value)
                                if parsed_plc_id is not None:
                                    plc_items[parsed_plc_id] = obj_item
                                break

                link_item = None
                if obj.linked_file is not None and not compact_view:
                    linked_path = obj.linked_file.resolve()
                    linked_severity = issue_severity_by_path.get(linked_path, "")
                    linked_dirty = path_is_dirty(linked_path)
                    linked_missing = not obj.linked_file.exists()
                    linked_badge = self._tree_status_badge(
                        severity=linked_severity,
                        dirty=linked_dirty,
                        missing=linked_missing,
                    )
                    linked_tags = ("node-link",) + self._tree_status_tags(
                        severity=linked_severity,
                        dirty=linked_dirty,
                        missing=linked_missing,
                    )
                    link_item = self.startup_tree.insert(
                        obj_item,
                        "end",
                        text=linked_badge + self._prefixed_tree_text("↗", "File"),
                        values=(self._relative_display(obj.linked_file),),
                        open=("linked-file", object_key) in expanded_keys,
                        tags=linked_tags,
                    )
                    self.startup_item_map[link_item] = ("linked-file", obj)
                    self.linked_file_tree_items.setdefault(obj.linked_file.resolve(), []).append(link_item)
                    self.linked_file_item_by_object_key[object_key] = link_item

                detail_skip = set()
                if obj.linked_file is not None:
                    detail_skip.add("FILE")
                if obj.command_details and not compact_view:
                    command_group_item = self.startup_tree.insert(
                        obj_item,
                        "end",
                        text=self._prefixed_tree_text("≡", "Macros"),
                        values=("{} macro(s)".format(len(obj.command_details)),),
                        open=("detail-group", object_key) in expanded_keys,
                        tags=("group-command",),
                    )
                    self.startup_item_map[command_group_item] = ("detail-group", obj)
                    for key, value in obj.command_details:
                        detail_skip.add(key)
                        detail_item = self.startup_tree.insert(
                            command_group_item,
                            "end",
                            text=self._prefixed_tree_text("·", key),
                            values=(value,),
                            tags=("node-detail",),
                        )
                        self.startup_item_map[detail_item] = ("detail", obj)

                if obj.linked_file_details and not compact_view:
                    linked_parent_item = link_item if link_item is not None else obj_item
                    linked_group_item = self.startup_tree.insert(
                        linked_parent_item,
                        "end",
                        text=self._prefixed_tree_text("≡", "Macros"),
                        values=("{} macro(s)".format(len(obj.linked_file_details)),),
                        open=("linked-detail-group", object_key) in expanded_keys,
                        tags=("group-linked",),
                    )
                    self.startup_item_map[linked_group_item] = ("linked-detail-group", obj)
                    for key, value in obj.linked_file_details:
                        detail_skip.add(key)
                        detail_item = self.startup_tree.insert(
                            linked_group_item,
                            "end",
                            text=self._prefixed_tree_text("·", key),
                            values=(value,),
                            tags=("node-link",),
                        )
                        self.startup_item_map[detail_item] = ("linked-detail", obj)

                for key, value in obj.details:
                    if compact_view:
                        continue
                    if key in detail_skip:
                        continue
                    detail_item = self.startup_tree.insert(
                        obj_item,
                        "end",
                        text=self._prefixed_tree_text("·", key),
                        values=(value,),
                        tags=("node-detail",),
                    )
                    self.startup_item_map[detail_item] = ("detail", obj)

        target_item = self._find_tree_item_by_restore_key(selected_key)
        if target_item is None:
            target_item = file_items.get(startup_path.resolve())
        if target_item is not None:
            self.startup_tree.selection_set(target_item)
            self.startup_tree.focus(target_item)
            self.startup_tree.see(target_item)
        overview_prefix = "Grouped" if tree_view_mode == "objects" else "Objects"
        if filtered_view:
            self.tree_overview_var.set(
                "{}  •  {} shown / {} objects  •  {} slaves  •  {} axes  •  {} PLC".format(
                    overview_prefix,
                    visible_objects,
                    total_objects,
                    total_slaves,
                    total_axes,
                    total_plcs,
                )
            )
        else:
            self.tree_overview_var.set(
                "{}  •  {} objects  •  {} slaves  •  {} axes  •  {} PLC".format(
                    overview_prefix,
                    total_objects,
                    total_slaves,
                    total_axes,
                    total_plcs,
                )
            )
        self.tree_error_var.set("{} error{}".format(total_errors, "" if total_errors == 1 else "s"))
        self.tree_warning_var.set("{} warning{}".format(total_warnings, "" if total_warnings == 1 else "s"))
        self.tree_unsaved_var.set("{} unsaved".format(total_dirty))
        self.tree_missing_var.set("{} missing".format(total_missing))
        if self.tree_error_button is not None:
            self.tree_error_button.configure(state="normal" if total_errors else "disabled")
        if self.tree_warning_button is not None:
            self.tree_warning_button.configure(state="normal" if total_warnings else "disabled")
        if self.tree_unsaved_button is not None:
            self.tree_unsaved_button.configure(state="normal" if total_dirty else "disabled")
        if self.tree_missing_button is not None:
            self.tree_missing_button.configure(state="normal" if total_missing else "disabled")

    def _relative_display(self, path: Path) -> str:
        if self.ecmccfg_root is not None:
            try:
                return str(path.resolve().relative_to(self.ecmccfg_root.resolve()))
            except ValueError:
                pass
        startup_value = self.startup_var.get().strip()
        if startup_value:
            try:
                return str(path.resolve().relative_to(Path(startup_value).expanduser().resolve().parent))
            except ValueError:
                pass
        return str(path)

    def _ordered_param_rows(self, rows: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        priority = {
            "TYPE": 0,
            "TITLE": 1,
            "PATH": 2,
            "FILE": 3,
            "SOURCE": 4,
            "HW_DESC": 5,
            "SLAVE_ID": 6,
            "AXIS_ID": 7,
            "AX_NAME": 8,
            "ENC_SID": 9,
            "PLC_ID": 10,
            "COMP": 11,
            "COMP_S_ID": 12,
            "CH_ID": 13,
            "PLUGIN_ID": 14,
            "DS_ID": 15,
            "LUT_ID": 16,
            "CONFIG": 17,
            "LOCAL_CONFIG": 18,
            "COMMAND": 19,
            "MODULE": 20,
            "VERSION": 21,
            "RATE_MS": 22,
            "OBJECTS": 90,
            "PARENT_FILE": 91,
            "SOURCE_LINE": 92,
        }
        return sorted(rows, key=lambda item: (priority.get(item[0], 50), item[0]))

    def _param_row_tags(self, key: str) -> Tuple[str, ...]:
        primary_keys = {
            "FILE",
            "HW_DESC",
            "SLAVE_ID",
            "AXIS_ID",
            "AX_NAME",
            "ENC_SID",
            "PLC_ID",
            "COMP",
            "COMP_S_ID",
            "CH_ID",
            "PLUGIN_ID",
            "DS_ID",
            "LUT_ID",
            "CONFIG",
            "LOCAL_CONFIG",
            "COMMAND",
            "MODULE",
            "VERSION",
            "RATE_MS",
        }
        meta_keys = {"TYPE", "TITLE", "SOURCE", "PATH", "PARENT_FILE", "SOURCE_LINE", "OBJECTS"}
        if key in primary_keys:
            return ("param-primary",)
        if key in meta_keys:
            return ("param-meta",)
        return ()

    def _ordered_context_rows(self, rows: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        def priority_for(key: str) -> Tuple[int, str]:
            if key.startswith("Macro "):
                return (20, key)
            priority = {
                "Selection": 0,
                "Title": 1,
                "Category": 2,
                "Linked File": 3,
                "Path": 4,
                "File": 5,
                "Source": 6,
                "HW_DESC": 7,
                "Slave ID": 8,
                "Parent Slave": 9,
                "Command": 10,
                "Allowed Macros": 11,
                "Missing Required": 12,
                "Known Groups": 13,
                "Supported Components": 14,
                "EC_COMP_TYPE": 15,
                "PVT Axes": 16,
                "Objects": 90,
                "Editor File": 91,
                "Editor Type": 92,
            }
            return (priority.get(key, 50), key)

        return sorted(rows, key=lambda item: priority_for(item[0]))

    def _context_row_tags(self, key: str) -> Tuple[str, ...]:
        primary_keys = {
            "Selection",
            "Title",
            "Category",
            "Linked File",
            "Path",
            "File",
            "Source",
            "HW_DESC",
            "Slave ID",
            "Parent Slave",
            "Command",
        }
        support_keys = {
            "Allowed Macros",
            "Missing Required",
            "Known Groups",
            "Supported Components",
            "EC_COMP_TYPE",
            "PVT Axes",
        }
        meta_keys = {"Objects", "Editor File", "Editor Type"}
        if key.startswith("Macro "):
            return ("context-support",)
        if key in primary_keys:
            return ("context-primary",)
        if key in support_keys:
            return ("context-support",)
        if key in meta_keys:
            return ("context-meta",)
        return ()

    def _populate_param_tree(self, rows: List[Tuple[str, str]]) -> None:
        self.param_tree.delete(*self.param_tree.get_children(""))
        if not rows:
            rows = [("Selection", "No object selected")]
        for key, value in self._ordered_param_rows(rows):
            self.param_tree.insert("", "end", text=key, values=(value,), tags=self._param_row_tags(key))

    def _populate_context_tree(self, rows: List[Tuple[str, str]]) -> None:
        if self.context_tree is None:
            return
        self.context_tree.delete(*self.context_tree.get_children(""))
        if not rows:
            rows = [("Context", "Open a startup file or select an object")]
        for key, value in self._ordered_context_rows(rows):
            self.context_tree.insert("", "end", text=key, values=(value,), tags=self._context_row_tags(key))

    def _populate_resolved_text(self, text: str) -> None:
        if self.resolved_text is None:
            return
        self.resolved_text.configure(state=self.tk.NORMAL)
        self.resolved_text.delete("1.0", "end")
        self.resolved_text.insert("1.0", text)
        self.resolved_text.configure(state=self.tk.DISABLED)

    def _issues_for_entry(self, entry: Optional[Tuple[str, object]]) -> List[ValidationIssue]:
        if entry is None or self.latest_result is None:
            return []
        entry_type, payload = entry
        issues: List[ValidationIssue] = []
        if entry_type == "file":
            file_path = payload.path.resolve()
            for issue in self.latest_result.issues:
                if issue.source.resolve() == file_path:
                    issues.append(issue)
            return issues
        if not isinstance(payload, StartupObject):
            return issues

        obj = payload
        source_path = obj.source.resolve()
        linked_path = obj.linked_file.resolve() if obj.linked_file is not None else None
        for issue in self.latest_result.issues:
            issue_source = issue.source.resolve()
            if issue_source == source_path and issue.line == obj.line:
                issues.append(issue)
                continue
            if linked_path is not None and issue_source == linked_path:
                issues.append(issue)
                continue
            if issue_source == source_path and obj.kind in issue.message.lower():
                issues.append(issue)
        return issues

    def _missing_required_command_macros(self, obj: StartupObject) -> List[str]:
        script_name = self._module_script_name_for_object(obj)
        if not script_name:
            return []
        macro_spec = self.inventory.module_macro_specs.get(script_name)
        if macro_spec is None or not macro_spec.required:
            return []
        current_keys = set(self._current_object_macro_map(obj))
        return sorted(key for key in macro_spec.required if key not in current_keys)

    def _unresolved_linked_file_macros(self, obj: StartupObject) -> List[str]:
        return sorted(key for key, value in obj.linked_file_details if str(value).strip() == "<unresolved>")

    def _axis_group_names_for_source(self, source: Path) -> List[str]:
        groups: Set[str] = set()
        if self.latest_startup_tree is None:
            return []
        global_macros = self._startup_known_macro_map()
        resolved_source = source.resolve()
        for file_node in self.latest_startup_tree.files:
            if file_node.path.resolve() != resolved_source:
                continue
            for obj in file_node.objects:
                if obj.kind != "axis" or obj.linked_file is None:
                    continue
                yaml_text = _read_text_from_buffers(obj.linked_file, self.file_buffers)
                if yaml_text is None:
                    continue
                macro_scope = dict(global_macros)
                for key, value in obj.command_details:
                    macro_scope[key] = value
                for key, value in obj.linked_file_details:
                    macro_scope[key] = value
                expanded_yaml = _expand_text_macros(yaml_text, macro_scope)
                for entry in _parse_simple_yaml_paths(expanded_yaml):
                    if entry.path == "axis.group" and entry.value is not None:
                        group_name = _normalize_value(entry.value)
                        if group_name:
                            groups.add(group_name)
        return sorted(groups)

    def _pvt_axis_titles_for_source(self, source: Path) -> List[str]:
        titles: List[str] = []
        if self.latest_startup_tree is None:
            return titles
        global_macros = self._startup_known_macro_map()
        resolved_source = source.resolve()
        for file_node in self.latest_startup_tree.files:
            if file_node.path.resolve() != resolved_source:
                continue
            for obj in file_node.objects:
                if obj.kind != "axis" or obj.linked_file is None:
                    continue
                yaml_text = _read_text_from_buffers(obj.linked_file, self.file_buffers)
                if yaml_text is None:
                    continue
                macro_scope = dict(global_macros)
                for key, value in obj.command_details:
                    macro_scope[key] = value
                for key, value in obj.linked_file_details:
                    macro_scope[key] = value
                expanded_yaml = _expand_text_macros(yaml_text, macro_scope)
                parsed_entries = _parse_simple_yaml_paths(expanded_yaml)
                has_pvt = any(
                    entry.path in {"axis.pvt", "epics.motorRecord.pvt"}
                    or entry.path.startswith("axis.pvt.")
                    or entry.path.startswith("epics.motorRecord.pvt.")
                    for entry in parsed_entries
                )
                if has_pvt:
                    titles.append(obj.title)
        return titles

    def _axis_objects_for_source(self, source: Path) -> List[StartupObject]:
        objects: List[StartupObject] = []
        if self.latest_startup_tree is None:
            return objects
        resolved_source = source.resolve()
        for file_node in self.latest_startup_tree.files:
            if file_node.path.resolve() != resolved_source:
                continue
            for obj in file_node.objects:
                if obj.kind == "axis":
                    objects.append(obj)
        return objects

    def _find_axis_object_by_title(self, source: Path, title: str) -> Optional[StartupObject]:
        normalized_title = str(title).strip()
        if not normalized_title:
            return None
        for obj in self._axis_objects_for_source(source):
            if obj.title == normalized_title:
                return obj
        return None

    def _help_issue_action_for_entry(
        self,
        issue: ValidationIssue,
        entry: Optional[Tuple[str, object]],
    ) -> Optional[Tuple[str, str, object]]:
        if entry is None or entry[0] == "file" or not isinstance(entry[1], StartupObject):
            return None
        obj = entry[1]
        message = issue.message
        same_object_issue = issue.source.resolve() == obj.source.resolve() and issue.line == obj.line

        missing_macro_match = re.search(r"Missing required macro '([^']+)'", message)
        if missing_macro_match and same_object_issue and self._selected_inline_macro_context() is not None:
            return ("Add Macro", "add_macro", missing_macro_match.group(1))

        if (
            "Missing " in message and "reference:" in message
        ) or "Missing hardware config" in message or "Missing linked file" in message:
            if self._is_tree_entry_editable(entry):
                preferred_field = ""
                if "Missing file reference:" in message or "Missing linked file" in message:
                    preferred_field = "FILE"
                elif "Missing config reference:" in message or "Missing hardware config" in message:
                    preferred_field = "CONFIG"
                elif "Missing local_config reference:" in message:
                    preferred_field = "LOCAL_CONFIG"
                return ("Edit", "edit_field", (obj, preferred_field))

        if ("MST_GRP_NAME" in message or "SLV_GRP_NAME" in message) and obj.kind == "master_slave_sm":
            axis_objects = self._axis_objects_for_source(obj.source)
            if axis_objects:
                return ("Jump Axis", "jump_object", axis_objects[0])
            if self._is_tree_entry_editable(entry):
                preferred_field = "MST_GRP_NAME" if "MST_GRP_NAME" in message else "SLV_GRP_NAME"
                return ("Edit", "edit_field", (obj, preferred_field))

        if "Duplicate axis.id" in message or "Duplicate epics.name" in message:
            if obj.kind == "axis":
                match = re.search(r"for '([^']+)' and '([^']+)'", message)
                if match:
                    first_title, second_title = match.groups()
                    other_title = first_title if second_title == obj.title else second_title
                    other_axis = self._find_axis_object_by_title(obj.source, other_title)
                    if other_axis is not None:
                        return ("Jump Axis", "jump_object", other_axis)
                if obj.linked_file is not None:
                    return ("Open File", "open_linked", obj)

        if obj.linked_file is not None and issue.source.resolve() == obj.linked_file.resolve():
            return ("Open File", "open_linked", obj)

        return None

    def _help_suggestions_for_entry(self, entry: Optional[Tuple[str, object]]) -> List[Tuple[str, Optional[str]]]:
        if entry is None:
            return [("Select an object to see targeted suggestions.", None)]

        entry_type, payload = entry
        suggestions: List[Tuple[str, Optional[str]]] = []
        if entry_type == "file":
            suggestions.append(("Action: Validate the startup file", "validate"))
            suggestions.append(("Use the tree flow order to review top-level execution from top to bottom.", None))
            return suggestions
        if not isinstance(payload, StartupObject):
            suggestions.append(("Action: Open startup file", "open_linked"))
            suggestions.append(("Use grouped view to scan by category, or switch back to Flow to inspect execution order.", None))
            return suggestions

        obj = payload
        detail_map = self._object_detail_map(obj)
        if obj.linked_file is not None:
            suggestions.append(("Action: Open linked file", "open_linked"))
            if not obj.linked_file.exists():
                suggestions.append(("The linked file is missing. Fix the FILE path or create the file.", None))

        if self._is_tree_entry_editable(entry):
            suggestions.append(("Action: Open full edit", "edit"))

        missing_required_macros = self._missing_required_command_macros(obj)
        if missing_required_macros and self._selected_inline_macro_context() is not None:
            suggestions.append(("Action: Add missing command macro", "add_macro"))
        if self._selected_inline_macro_context() is not None and not missing_required_macros:
            suggestions.append(("Action: Add macro", "add_macro"))

        if self.startup_var.get().strip():
            suggestions.append(("Action: Validate startup file", "validate"))

        if missing_required_macros:
            suggestions.append(("Required command macros still missing: {}".format(", ".join(missing_required_macros)), None))
        unresolved_linked_macros = self._unresolved_linked_file_macros(obj)
        if unresolved_linked_macros:
            suggestions.append(("Linked-file macros still unresolved: {}".format(", ".join(unresolved_linked_macros)), None))
        allowed_macros = self._available_command_macro_names(obj)
        if allowed_macros:
            preview = ", ".join(allowed_macros[:6])
            if len(allowed_macros) > 6:
                preview += ", ..."
            suggestions.append(("Allowed command macros: {}".format(preview), None))

        if obj.kind == "slave":
            hw_desc = detail_map.get("HW_DESC", "").strip()
            if hw_desc:
                comp_hw_type = self.inventory.hardware_component_types.get(hw_desc, hw_desc)
                support_map = self.inventory.component_support.get(comp_hw_type, {})
                if support_map:
                    suggestions.append(
                        (
                            "Supported components for {}: {}".format(
                                hw_desc,
                                ", ".join(sorted(support_map.keys())[:6]),
                            ),
                            None,
                        )
                    )
        elif obj.kind == "axis":
            if detail_map.get("DRV_SID"):
                suggestions.append(("Use Resolved view to verify DRV_SID/ENC_SID macros before saving.", None))
            if obj.linked_file is not None:
                suggestions.append(("Axis YAML can be reused. Check AX_NAME and AXIS_ID to avoid duplicates.", None))
        elif obj.kind == "slave_config":
            suggestions.append(("Slave config normally belongs directly under the slave it configures.", None))
            if detail_map.get("HW_DESC"):
                config_values = self._available_hardware_config_values(detail_map.get("HW_DESC", ""))
                if config_values:
                    preview = ", ".join(config_values[:6])
                    if len(config_values) > 6:
                        preview += ", ..."
                    suggestions.append(("Known config suffixes for {}: {}".format(detail_map.get("HW_DESC", ""), preview), None))
        elif obj.kind == "configured_axis":
            suggestions.append(("Open the linked axis config file to inspect the parameters applied before addAxis.cmd.", None))
            if detail_map.get("CFG_MACROS"):
                suggestions.append(("Check CFG_MACROS carefully. Invalid payloads are validated but not auto-corrected.", None))
        elif obj.kind == "apply_config":
            suggestions.append(("Run validation after applyConfig placement to confirm bus objects are in the intended order.", None))
            suggestions.append(("applyConfig should normally come after slave and axis setup, before app mode.", None))
        elif obj.kind == "master_slave_sm":
            known_groups = self._axis_group_names_for_source(obj.source)
            if known_groups:
                suggestions.append(("Known axis groups in this startup file: {}".format(", ".join(known_groups[:8])), None))
            suggestions.append(("Check master and slave group names against the configured axis groups before saving.", None))
        elif obj.kind == "subst_config":
            suggestions.append(("Use the linked substitution file to review generated axes and macros before execution.", None))
            if detail_map.get("MACROS"):
                suggestions.append(("Validate the substitution macro payload after edits to catch malformed segments.", None))
        elif obj.kind == "pvt_controller":
            pvt_axes = self._pvt_axis_titles_for_source(obj.source)
            if pvt_axes:
                preview = ", ".join(pvt_axes[:4])
                if len(pvt_axes) > 4:
                    preview += ", ..."
                suggestions.append(("Axes with PVT support in this startup file: {}".format(preview), None))
            suggestions.append(("Use a valid EtherCAT trigger entry if hardware triggering is enabled.", None))
            if not detail_map.get("TRG_EC_ENTRY"):
                suggestions.append(("Without TRG_EC_ENTRY the controller relies on software triggering only.", None))
        elif obj.kind == "lut":
            suggestions.append(("Open the linked LUT file to verify the curve data and LUT_ID.", None))
            suggestions.append(("Keep LUT_ID stable if PLCs or other objects refer to this lookup table later.", None))
        elif obj.kind == "encoder":
            suggestions.append(("Verify encoder slave and channel macros against the linked YAML.", None))
        elif obj.kind == "component":
            suggestions.append(("Validate after edits to confirm component support for the selected slave.", None))
        elif obj.kind == "ecmc_command":
            suggestions.append(("Open full edit to use parsed ECMC syntax help and parameter guidance.", None))
        elif obj.kind in {"plc", "plcvar_analog", "plcvar_binary"}:
            suggestions.append(("Use EC Help in the editor to pick valid EtherCAT entries in PLC content.", None))

        issues = self._issues_for_entry(entry)
        if issues:
            suggestions.insert(0, ("This selection has {} validation issue(s). Double-click a problem above to jump.".format(len(issues)), None))
        if not suggestions:
            suggestions.append(("No specific suggestions for this selection.", None))
        return suggestions

    def _suggestions_for_entry(self, entry: Optional[Tuple[str, object]]) -> List[str]:
        return [text for text, _action in self._help_suggestions_for_entry(entry)]

    def _populate_help_panel(self, entry: Optional[Tuple[str, object]] = None) -> None:
        if entry is None:
            entry = self._selected_startup_entry()
        self._update_help_header(entry)
        if self.help_issue_tree is not None:
            self.help_issue_item_map.clear()
            self.help_issue_action_map.clear()
            self.help_issue_tree.delete(*self.help_issue_tree.get_children(""))
            issues = self._issues_for_entry(entry)
            if not issues:
                item_id = self.help_issue_tree.insert("", "end", values=("INFO", "No problems for current selection", ""))
                self.help_issue_item_map[item_id] = ValidationIssue(
                    severity="info",
                    source=self.current_edit_path or Path("."),
                    line=1,
                    message="No problems for current selection",
                )
            else:
                for issue in issues:
                    action_info = self._help_issue_action_for_entry(issue, entry)
                    action_label = action_info[0] if action_info is not None else ""
                    item_id = self.help_issue_tree.insert(
                        "",
                        "end",
                        values=(issue.severity.upper(), issue.message, action_label),
                    )
                    self.help_issue_item_map[item_id] = issue
                    if action_info is not None:
                        self.help_issue_action_map[item_id] = (action_info[1], action_info[2])
        if self.help_suggestion_list is not None:
            self.help_suggestion_list.delete(0, self.tk.END)
            self.help_suggestion_actions = []
            for item, action in self._help_suggestions_for_entry(entry):
                self.help_suggestion_list.insert(self.tk.END, item)
                self.help_suggestion_actions.append(action)

    def _open_selected_help_issue(self, _event=None) -> str:
        if self.help_issue_tree is None:
            return "break"
        selected = self.help_issue_tree.selection()
        if not selected:
            return "break"
        item_id = selected[0]
        action_info = self.help_issue_action_map.get(item_id)
        if action_info is not None:
            action_name, payload = action_info
            if action_name == "add_macro":
                self._add_inline_macro(preferred_name=str(payload))
                return "break"
            if action_name == "edit":
                self._edit_selected_object()
                return "break"
            if action_name == "edit_field" and isinstance(payload, tuple) and len(payload) == 2:
                obj, preferred_field = payload
                if isinstance(obj, StartupObject):
                    target_item = self.object_tree_items.get(self._object_tree_key(obj))
                    if target_item is not None:
                        self.startup_tree.selection_set(target_item)
                        self.startup_tree.focus(target_item)
                        self.startup_tree.see(target_item)
                    self._edit_selected_object(preferred_field=preferred_field)
                return "break"
            if action_name == "open_linked":
                self._open_selected_object_file()
                return "break"
            if action_name == "jump_object" and isinstance(payload, StartupObject):
                target_item = self.object_tree_items.get(self._object_tree_key(payload))
                if target_item is not None:
                    self.startup_tree.selection_set(target_item)
                    self.startup_tree.focus(target_item)
                    self.startup_tree.see(target_item)
                    self._on_startup_tree_selected()
                return "break"
        issue = self.help_issue_item_map.get(item_id)
        if issue is None or issue.severity == "info":
            return "break"
        self._open_file_in_editor(issue.source, line=issue.line)
        return "break"

    def _run_selected_help_suggestion(self, _event=None) -> str:
        if self.help_suggestion_list is None:
            return "break"
        selection = self.help_suggestion_list.curselection()
        if not selection:
            return "break"
        index = int(selection[0])
        if index < 0 or index >= len(self.help_suggestion_actions):
            return "break"
        action = self.help_suggestion_actions[index]
        if action == "open_linked":
            self._open_selected_object_file()
        elif action == "edit":
            self._edit_selected_object()
        elif action == "add_macro":
            self._add_inline_macro()
        elif action == "validate":
            self._validate_current_project()
        return "break"

    def _quick_edit_fields_for_entry(self, entry: Optional[Tuple[str, object]]) -> Tuple[Optional[StartupObject], List[str]]:
        if entry is None or entry[0] == "file" or not isinstance(entry[1], StartupObject):
            return None, []
        obj = entry[1]
        editable_map = self._object_detail_map(obj)
        protected = {"TYPE", "TITLE", "SOURCE"}
        ordered_fields: List[str] = []
        if obj.linked_file is not None and "FILE" in editable_map:
            ordered_fields.append("FILE")
        for key, _value in obj.details:
            if key in editable_map and key not in protected and key not in ordered_fields:
                ordered_fields.append(key)
        for key, _value in obj.command_details:
            if key in editable_map and key not in protected and key not in ordered_fields:
                ordered_fields.append(key)
        for key, _value in obj.linked_file_details:
            if key in editable_map and key not in protected and key not in ordered_fields:
                ordered_fields.append(key)
        for key in sorted(editable_map):
            if key not in protected and key not in ordered_fields:
                ordered_fields.append(key)
        return obj, ordered_fields

    def _refresh_quick_edit_panel(self, entry: Optional[Tuple[str, object]] = None, selected_key: str = "") -> None:
        if entry is None:
            entry = self._selected_startup_entry()
        obj, fields = self._quick_edit_fields_for_entry(entry)
        self.quick_edit_fields = fields
        if self.quick_edit_frame is not None:
            if obj is not None and fields:
                if not self.quick_edit_frame.winfo_manager():
                    self.quick_edit_frame.pack(side=self.tk.BOTTOM, fill=self.tk.X, pady=(6, 0))
            elif self.quick_edit_frame.winfo_manager():
                self.quick_edit_frame.pack_forget()
        if self.quick_edit_key_combo is not None:
            self.quick_edit_key_combo.configure(values=fields)
        state = "normal" if obj is not None and bool(fields) else "disabled"
        if self.quick_edit_key_combo is not None:
            self.quick_edit_key_combo.configure(state="readonly" if state == "normal" else "disabled")
        if self.quick_edit_value_entry is not None:
            self.quick_edit_value_entry.configure(state=state)
        if self.quick_edit_apply_button is not None:
            self.quick_edit_apply_button.configure(state=state)
        if self.quick_edit_open_button is not None:
            self.quick_edit_open_button.configure(state=state)

        if obj is None or not fields:
            self.quick_edit_key_var.set("")
            self.quick_edit_value_var.set("")
            self.quick_edit_hint_var.set("")
            return

        current_key = selected_key if selected_key in fields else self.quick_edit_key_var.get().strip()
        if current_key not in fields:
            current_key = fields[0]
        editable_map = self._object_detail_map(obj)
        self.quick_edit_key_var.set(current_key)
        self.quick_edit_value_var.set(str(editable_map.get(current_key, "")))
        self.quick_edit_hint_var.set("Updates the selected object directly.")

    def _refresh_resolved_preview(self) -> None:
        if self.current_edit_path is not None:
            current_path = self.current_edit_path.resolve()
            if current_path in self.file_buffers:
                source_text = self.file_buffers[current_path]
            elif current_path.exists():
                source_text = _read_text(current_path)
            else:
                source_text = ""
            macro_map = self._editor_known_macro_map()
            expanded = _expand_text_macros(source_text, macro_map) if source_text else ""
            header = ["# Resolved view", "# File: {}".format(self._relative_display(current_path))]
            if macro_map:
                header.append(
                    "# Macros: {}".format(
                        ", ".join("{}={}".format(key, macro_map[key]) for key in sorted(macro_map))
                    )
                )
            self._populate_resolved_text("\n".join(header + ["", expanded]))
            return

        entry = self._selected_startup_entry()
        if entry is None or entry[0] == "file" or not isinstance(entry[1], StartupObject):
            self._populate_resolved_text("Open a file or select an object to see resolved values.")
            return
        obj = entry[1]
        macro_map = self._current_object_macro_map(obj)
        source_line = ""
        try:
            content = self.file_buffers.get(obj.source.resolve(), _read_text(obj.source))
            lines = content.splitlines()
            if 0 < obj.line <= len(lines):
                source_line = lines[obj.line - 1]
        except Exception:
            source_line = ""
        resolved_line = _expand_text_macros(source_line, macro_map) if source_line else ""
        self._populate_resolved_text(
            "# Resolved object\n# {}\n\n{}\n".format(obj.title, resolved_line or "No resolved content available.")
        )

    def _on_param_tree_selected(self, _event=None) -> None:
        selected = self.param_tree.selection()
        if not selected:
            self._refresh_quick_edit_panel()
            return
        item_id = selected[0]
        key = self.param_tree.item(item_id, "text")
        self._refresh_quick_edit_panel(selected_key=key)

    def _on_quick_edit_field_changed(self, _event=None) -> None:
        entry = self._selected_startup_entry()
        obj, fields = self._quick_edit_fields_for_entry(entry)
        if obj is None or not fields:
            return
        key = self.quick_edit_key_var.get().strip()
        if key not in fields:
            return
        editable_map = self._object_detail_map(obj)
        self.quick_edit_value_var.set(str(editable_map.get(key, "")))

    def _apply_quick_edit(self, _event=None) -> str:
        from tkinter import messagebox

        entry = self._selected_startup_entry()
        obj, fields = self._quick_edit_fields_for_entry(entry)
        if obj is None or not fields:
            return "break"
        key = self.quick_edit_key_var.get().strip()
        if key not in fields:
            messagebox.showinfo("Select field", "Choose a field to update first.")
            return "break"
        editable_map = self._object_detail_map(obj)
        updated_values = dict(editable_map)
        updated_values[key] = self.quick_edit_value_var.get()
        if self._apply_direct_object_update(obj, updated_values):
            self._refresh_quick_edit_panel(selected_key=key)
        return "break"

    def _update_object_action_buttons(self, entry: Optional[Tuple[str, object]] = None) -> None:
        if entry is None:
            entry = self._selected_startup_entry()
        editable_state = "disabled"
        macro_state = "disabled"
        open_state = "disabled"
        copy_state = "disabled"
        paste_state = "disabled"
        delete_state = "disabled"
        move_up_state = "disabled"
        move_down_state = "disabled"
        hint = "Select an object to edit, open a linked file, or add a macro."
        if entry is not None:
            entry_type, payload = entry
            if self._is_tree_entry_editable(entry):
                editable_state = "normal"
            if self._can_open_tree_entry_file(entry):
                open_state = "normal"
            if entry_type == "object":
                copy_state = "normal"
                delete_state = "normal"
                move_up_state = "normal" if self._can_move_selected_object(-1) else "disabled"
                move_down_state = "normal" if self._can_move_selected_object(1) else "disabled"
            if entry_type in {"object", "file"} and self.copied_object_text:
                paste_state = "normal"
            if self._selected_inline_macro_context() is not None:
                macro_state = "normal"
            if entry_type == "file":
                hint = "Open the file, filter the tree, or paste copied objects into this startup file."
            elif entry_type == "object":
                actions: List[str] = []
                if editable_state == "normal":
                    actions.append("Edit")
                if open_state == "normal":
                    actions.append("Open linked file")
                if macro_state == "normal":
                    actions.append("Add macro")
                if copy_state == "normal":
                    actions.append("Copy")
                if delete_state == "normal":
                    actions.append("Remove")
                if move_up_state == "normal" or move_down_state == "normal":
                    actions.append("Move")
                if actions:
                    hint = "Actions: {}.".format(" • ".join(actions))
            elif entry_type in {"linked-file", "linked-detail", "linked-detail-group"}:
                hint = "Open the linked file or inspect macros applied to it."
            elif entry_type == "category-group":
                hint = "Use this group to scan related objects quickly."
        if self.edit_object_button is not None:
            self.edit_object_button.configure(state=editable_state)
        if self.add_macro_button is not None:
            self.add_macro_button.configure(state=macro_state)
        if self.open_object_file_button is not None:
            self.open_object_file_button.configure(state=open_state)
        if self.context_open_button is not None:
            self.context_open_button.configure(state=open_state)
        if self.help_open_button is not None:
            self.help_open_button.configure(state=open_state)
        if self.copy_object_button is not None:
            self.copy_object_button.configure(state=copy_state)
        if self.paste_object_button is not None:
            self.paste_object_button.configure(state=paste_state)
        if self.delete_object_button is not None:
            self.delete_object_button.configure(state=delete_state)
        if self.move_up_button is not None:
            self.move_up_button.configure(state=move_up_state)
        if self.move_down_button is not None:
            self.move_down_button.configure(state=move_down_state)
        if self.context_edit_button is not None:
            self.context_edit_button.configure(state=editable_state)
        if self.help_edit_button is not None:
            self.help_edit_button.configure(state=editable_state)
        if self.context_macro_button is not None:
            self.context_macro_button.configure(state=macro_state)
        if self.help_macro_button is not None:
            self.help_macro_button.configure(state=macro_state)
        if self.help_validate_button is not None:
            self.help_validate_button.configure(state="normal" if self.latest_startup_tree is not None else "disabled")
        self.object_action_hint_var.set(hint)

    def _can_open_tree_entry_file(self, entry: Optional[Tuple[str, object]]) -> bool:
        return self._tree_entry_open_target(entry) is not None

    def _tree_entry_open_target(
        self, entry: Optional[Tuple[str, object]]
    ) -> Optional[Tuple[Path, Optional[int], Optional[str]]]:
        if entry is None:
            return None
        entry_type, payload = entry
        if entry_type == "file":
            return (payload.path, payload.parent_line, None)
        if entry_type == "category-group":
            file_path, _category, _count = payload
            return (file_path, None, None)
        if not isinstance(payload, StartupObject):
            return None
        obj = payload
        if entry_type in {"linked-file", "linked-detail", "linked-detail-group"}:
            if obj.linked_file is None:
                return None
            return (obj.linked_file, None, self._object_tree_key(obj))
        if entry_type in {"object", "detail", "detail-group"} and obj.linked_file is not None:
            return (obj.linked_file, None, self._object_tree_key(obj))
        return None

    def _is_tree_entry_editable(self, entry: Optional[Tuple[str, object]]) -> bool:
        if entry is None:
            return False
        entry_type, payload = entry
        if entry_type == "object":
            return payload.kind in self._editable_object_kinds()
        if entry_type in {"detail", "linked-detail", "linked-file"}:
            editable_map = self._object_detail_map(payload)
            if entry_type == "linked-file":
                return "FILE" in editable_map
            selected = self.startup_tree.selection()
            if not selected:
                return False
            key = self._tree_text_logical_name(self.startup_tree.item(selected[0], "text"))
            return key in editable_map
        return False

    def _context_rows_for_entry(self, entry_type: str, payload) -> List[Tuple[str, str]]:
        rows: List[Tuple[str, str]] = []
        if self.current_edit_path is not None:
            rows.append(("Editor File", self._relative_display(self.current_edit_path)))
            language = self._editor_language()
            if language:
                rows.append(("Editor Type", language))

        if entry_type == "file":
            file_node = payload
            rows.append(("Selection", "file"))
            rows.append(("Path", self._relative_display(file_node.path)))
            rows.append(("Objects", str(len(file_node.objects))))
            return rows
        if entry_type == "category-group":
            file_path, category, count = payload
            rows.append(("Selection", "group"))
            rows.append(("Category", category))
            rows.append(("File", self._relative_display(file_path)))
            rows.append(("Objects", str(count)))
            return rows

        obj = payload
        rows.append(("Selection", obj.kind))
        rows.append(("Title", obj.title))
        rows.append(("Source", "{}:{}".format(self._relative_display(obj.source), obj.line)))
        if obj.linked_file is not None:
            rows.append(("Linked File", self._relative_display(obj.linked_file)))
        if obj.slave_id is not None:
            rows.append(("Slave ID", str(obj.slave_id)))
        if obj.parent_slave_id is not None and obj.kind != "slave":
            rows.append(("Parent Slave", str(obj.parent_slave_id)))

        detail_map = self._object_detail_map(obj)
        hw_desc = detail_map.get("HW_DESC", "").strip()
        if hw_desc:
            rows.append(("HW_DESC", hw_desc))

        macro_map = {}
        if entry_type in {"linked-file", "linked-detail", "linked-detail-group"}:
            macro_map.update(self._startup_known_macro_map())
            for key, value in obj.linked_file_details:
                macro_map[key] = value
            for key, value in obj.command_details:
                macro_map.setdefault(key, value)
        else:
            macro_map = self._current_object_macro_map(obj)
        if macro_map:
            for key in sorted(macro_map):
                rows.append(("Macro {}".format(key), str(macro_map[key])))

        allowed_macros = self._available_command_macro_names(obj)
        missing_required_macros = self._missing_required_command_macros(obj)
        script_name = self._module_script_name_for_object(obj)
        if script_name:
            rows.append(("Command", script_name))
        if allowed_macros:
            rows.append(("Allowed Macros", ", ".join(allowed_macros)))
        if missing_required_macros:
            rows.append(("Missing Required", ", ".join(missing_required_macros)))

        if obj.kind in {"slave", "component"}:
            comp_hw_type = self.inventory.hardware_component_types.get(hw_desc, hw_desc) if hw_desc else ""
            if comp_hw_type:
                rows.append(("EC_COMP_TYPE", comp_hw_type))
                support_map = self.inventory.component_support.get(comp_hw_type, {})
                if support_map:
                    rows.append(("Supported Components", ", ".join(sorted(support_map.keys()))))
        elif obj.kind == "master_slave_sm":
            known_groups = self._axis_group_names_for_source(obj.source)
            if known_groups:
                rows.append(("Known Groups", ", ".join(known_groups)))
        elif obj.kind == "pvt_controller":
            pvt_axes = self._pvt_axis_titles_for_source(obj.source)
            if pvt_axes:
                rows.append(("PVT Axes", ", ".join(pvt_axes)))

        return rows

    def _refresh_context_panel(self, entry: Optional[Tuple[str, object]] = None) -> None:
        if entry is None:
            entry = self._selected_startup_entry()
        if entry is None:
            self.selection_header_var.set("No object selected")
            self._update_selection_badges(None)
            rows: List[Tuple[str, str]] = []
            if self.current_edit_path is not None:
                rows.append(("Editor File", self._relative_display(self.current_edit_path)))
                language = self._editor_language()
                if language:
                    rows.append(("Editor Type", language))
                linked_obj = self._linked_object_for_current_editor()
                if linked_obj is not None:
                    rows.append(("Linked Object", linked_obj.title))
                    for key, value in sorted(self._editor_known_macro_map().items()):
                        rows.append(("Macro {}".format(key), str(value)))
            self._populate_context_tree(rows)
            self._update_object_action_buttons(None)
            self._refresh_quick_edit_panel(None)
            self._refresh_resolved_preview()
            self._populate_help_panel(None)
            return
        self._populate_context_tree(self._context_rows_for_entry(*entry))
        self._update_object_action_buttons(entry)
        self._refresh_quick_edit_panel(entry)
        self._refresh_resolved_preview()
        self._populate_help_panel(entry)

    def _open_selected_object_file(self) -> None:
        entry = self._selected_startup_entry()
        target = self._tree_entry_open_target(entry)
        if target is None:
            return
        path, line, linked_object_key = target
        self._open_file_in_editor(path, line=line, linked_object_key=linked_object_key)

    def _show_existing_editor_location(
        self,
        path: Path,
        line: Optional[int] = None,
        linked_object_key: Optional[Tuple[Path, int, str, str]] = None,
    ) -> bool:
        resolved = path.resolve()
        if self.current_edit_path != resolved:
            return False
        self.current_edit_linked_object_key = linked_object_key
        self.editor_file_var.set(str(resolved))
        if line is not None and line > 0:
            self._highlight_editor_line(line)
            self._highlight_tree_target_line(line)
        return True

    def _show_tree_entry_in_editor(self, entry: Optional[Tuple[str, object]]) -> None:
        if entry is None:
            return
        entry_type, payload = entry
        if entry_type == "file":
            if self._show_existing_editor_location(payload.path, line=payload.parent_line):
                return
            self._open_file_in_editor(payload.path, line=payload.parent_line)
            return
        if entry_type == "category-group":
            file_path, _category, _count = payload
            if self._show_existing_editor_location(file_path):
                return
            self._open_file_in_editor(file_path)
            return
        obj = payload
        if entry_type in {"linked-file", "linked-detail", "linked-detail-group"}:
            if obj.linked_file is not None and obj.linked_file.exists():
                if self._show_existing_editor_location(
                    obj.linked_file,
                    linked_object_key=self._object_tree_key(obj),
                ):
                    return
                self._open_file_in_editor(obj.linked_file, linked_object_key=self._object_tree_key(obj))
            return
        if self._show_existing_editor_location(obj.source, line=obj.line):
            return
        self._open_file_in_editor(obj.source, line=obj.line)

    def _file_browser_root_for_startup(self, startup_path: Path) -> Path:
        return startup_path.resolve().parent

    def _iter_file_browser_children(self, path: Path) -> List[Path]:
        try:
            children = sorted(
                path.iterdir(),
                key=lambda child: (not child.is_dir(), child.name.lower()),
            )
        except Exception:
            return []
        return [child for child in children if not child.name.startswith(".")]

    def _insert_file_browser_node(self, parent_item: str, path: Path) -> str:
        resolved = path.resolve()
        item_type = "DIR" if resolved.is_dir() else resolved.suffix.lower().lstrip(".").upper() or "FILE"
        item_id = self.file_browser_tree.insert(
            parent_item,
            "end",
            text=resolved.name,
            values=(item_type,),
            open=False,
        )
        self.file_browser_item_paths[item_id] = resolved
        self.file_browser_path_items[resolved] = item_id
        if resolved.is_dir() and self._iter_file_browser_children(resolved):
            self.file_browser_tree.insert(item_id, "end", text="")
        return item_id

    def _remove_file_browser_subtree_mappings(self, item_id: str) -> None:
        for child_item in self.file_browser_tree.get_children(item_id):
            self._remove_file_browser_subtree_mappings(child_item)
        child_path = self.file_browser_item_paths.pop(item_id, None)
        if child_path is not None:
            self.file_browser_path_items.pop(child_path, None)
            self.file_browser_loaded_dirs.discard(child_path)

    def _populate_file_browser_directory(self, item_id: str, path: Path) -> None:
        if self.file_browser_tree is None:
            return
        resolved = path.resolve()
        if resolved in self.file_browser_loaded_dirs:
            return
        for child_item in self.file_browser_tree.get_children(item_id):
            self._remove_file_browser_subtree_mappings(child_item)
        self.file_browser_tree.delete(*self.file_browser_tree.get_children(item_id))
        for child in self._iter_file_browser_children(resolved):
            self._insert_file_browser_node(item_id, child)
        self.file_browser_loaded_dirs.add(resolved)

    def _ensure_file_browser_path(self, path: Path) -> Optional[str]:
        if self.file_browser_tree is None or self.file_browser_root is None or self.file_browser_root_item is None:
            return None
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self.file_browser_root)
        except ValueError:
            return None
        if relative.parts == ():
            return self.file_browser_root_item

        current_path = self.file_browser_root
        current_item = self.file_browser_root_item
        for part in relative.parts:
            self._populate_file_browser_directory(current_item, current_path)
            next_path = (current_path / part).resolve()
            next_item = self.file_browser_path_items.get(next_path)
            if next_item is None:
                return None
            if next_path.is_dir():
                self.file_browser_tree.item(next_item, open=True)
            current_path = next_path
            current_item = next_item
        return current_item

    def _populate_file_browser(self, startup_path: Path) -> None:
        if self.file_browser_tree is None:
            return
        root_path = self._file_browser_root_for_startup(startup_path)
        self.file_browser_root = root_path
        self.file_browser_item_paths.clear()
        self.file_browser_path_items.clear()
        self.file_browser_loaded_dirs.clear()
        self.file_browser_tree.delete(*self.file_browser_tree.get_children(""))

        root_label = root_path.name or str(root_path)
        root_item = self.file_browser_tree.insert(
            "",
            "end",
            text=root_label,
            values=("DIR",),
            open=True,
        )
        self.file_browser_item_paths[root_item] = root_path
        self.file_browser_path_items[root_path] = root_item
        self.file_browser_root_item = root_item
        self._populate_file_browser_directory(root_item, root_path)
        self.file_browser_tree.selection_set(root_item)
        self.file_browser_tree.focus(root_item)
        self.file_browser_tree.see(root_item)

    def _sync_file_browser_selection(self, path: Optional[Path]) -> None:
        if self.file_browser_tree is None or path is None:
            return
        item_id = self._ensure_file_browser_path(path)
        if item_id is None:
            return
        current = self.file_browser_tree.selection()
        if current and current[0] == item_id:
            return
        self.file_browser_tree.selection_set(item_id)
        self.file_browser_tree.focus(item_id)
        self.file_browser_tree.see(item_id)

    def _on_file_browser_selected(self, _event=None) -> None:
        if self.file_browser_tree is None:
            return
        selected = self.file_browser_tree.selection()
        if not selected:
            return
        path = self.file_browser_item_paths.get(selected[0])
        if path is None or path.is_dir():
            return
        if self.current_edit_path == path.resolve():
            return
        self._open_file_in_editor(path)

    def _on_file_browser_open(self, _event=None) -> None:
        if self.file_browser_tree is None:
            return
        item_id = self.file_browser_tree.focus()
        if not item_id:
            return
        path = self.file_browser_item_paths.get(item_id)
        if path is None or not path.is_dir():
            return
        self._populate_file_browser_directory(item_id, path)

    def _selected_startup_entry(self) -> Optional[Tuple[str, object]]:
        selected = self.startup_tree.selection()
        if not selected:
            return None
        return self.startup_item_map.get(selected[0])

    def _close_startup_menu(self) -> None:
        try:
            self.startup_menu.unpost()
        except Exception:
            pass

    def _on_escape(self, _event=None) -> None:
        self._close_startup_menu()

    def _on_global_left_click(self, event=None) -> None:
        widget = getattr(event, "widget", None)
        if widget is self.startup_tree:
            self._close_startup_menu()
            return
        widget_name = ""
        try:
            widget_name = str(widget)
        except Exception:
            widget_name = ""
        if "menu" in widget_name.lower():
            return
        self._close_startup_menu()

    def _on_status_message_changed(self, *_args) -> None:
        message = self.status_var.get().strip()
        if not message or message == self._last_logged_status:
            return
        self._last_logged_status = message
        self._append_log_message(message)

    def _append_log_message(self, message: str) -> None:
        if self.log_text is None or not message.strip():
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state=self.tk.NORMAL)
        if self.log_text.index("end-1c") != "1.0":
            self.log_text.insert("end", "\n")
        self.log_text.insert("end", "[{}] {}".format(timestamp, message.strip()))
        self.log_text.see("end")
        self.log_text.configure(state=self.tk.DISABLED)

    def _clear_activity_log(self) -> None:
        if self.log_text is None:
            return
        self.log_text.configure(state=self.tk.NORMAL)
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state=self.tk.DISABLED)
        self._last_logged_status = ""
        self._append_log_message("Activity log cleared")

    def _set_log_panel_visible(self, visible: bool) -> None:
        if self.log_pane is None or self.log_frame is None:
            return
        if visible and not self.log_visible:
            self.log_pane.add(self.log_frame, weight=2)
            self.log_visible = True
        elif not visible and self.log_visible:
            try:
                self.log_pane.forget(self.log_frame)
            except Exception:
                pass
            self.log_visible = False
        if self.log_toggle_button is not None:
            self.log_toggle_button.configure(text="Hide Results" if self.log_visible else "Show Results")

    def _hide_log_panel(self) -> None:
        self._set_log_panel_visible(False)

    def _selected_object_kind(self) -> str:
        entry = self._selected_startup_entry()
        if entry is None:
            return ""
        entry_type, payload = entry
        if entry_type == "file":
            return "file"
        if not isinstance(payload, StartupObject):
            return entry_type
        return payload.kind

    def _editable_object_kinds(self) -> Set[str]:
        return {
            "require",
            "master",
            "macro",
            "slave",
            "slave_config",
            "apply_config",
            "axis",
            "configured_axis",
            "encoder",
            "plc",
            "component",
            "plugin",
            "datastorage",
            "ecsdo",
            "ecdataitem",
            "master_slave_sm",
            "subst_config",
            "pvt_controller",
            "lut",
            "plcvar_analog",
            "plcvar_binary",
            "ecmc_command",
            "record_update_rate",
            "restore_record_update_rate",
        }

    def _module_script_name_for_object(self, obj: StartupObject) -> str:
        mapping = {
            "master": "addMaster.cmd",
            "slave": "addSlave.cmd",
            "slave_config": "applySlaveConfig.cmd",
            "apply_config": "applyConfig.cmd",
            "axis": "loadYamlAxis.cmd",
            "configured_axis": "configureAxis.cmd",
            "encoder": "loadYamlEnc.cmd",
            "plc": "loadPLCFile.cmd",
            "plugin": "loadPlugin.cmd",
            "datastorage": "addDataStorage.cmd",
            "component": "applyComponent.cmd",
            "ecsdo": "addEcSdoRT.cmd",
            "ecdataitem": "addEcDataItem.cmd",
            "master_slave_sm": "addMasterSlaveSM.cmd",
            "subst_config": "loadSubstConfig.cmd",
            "pvt_controller": "pvtControllerConfig.cmd",
            "lut": "loadLUTFile.cmd",
            "record_update_rate": "setRecordUpdateRate.cmd",
            "restore_record_update_rate": "restoreRecordUpdateRate.cmd",
        }
        if obj.kind == "require":
            module_name = dict(obj.details).get("MODULE", "")
            if module_name == "ecmccfg":
                return "startup.cmd"
        return mapping.get(obj.kind, "")

    def _selected_inline_macro_context(self) -> Optional[Tuple[str, StartupObject]]:
        entry = self._selected_startup_entry()
        if entry is None:
            return None
        entry_type, payload = entry
        if not isinstance(payload, StartupObject):
            return None
        if not self._module_script_name_for_object(payload):
            return None
        if payload.kind == "require":
            return ("command", payload)
        if entry_type in {"linked-file", "linked-detail", "linked-detail-group"}:
            return ("linked", payload)
        if entry_type in {"object", "detail", "detail-group"}:
            return ("command", payload)
        return None

    def _current_object_macro_map(self, obj: StartupObject) -> Dict[str, str]:
        resolved_target = obj.source.resolve()
        if resolved_target in self.file_buffers:
            content = self.file_buffers[resolved_target]
        elif resolved_target.exists():
            content = _read_text(resolved_target)
        else:
            return {}

        lines = content.splitlines()
        index = obj.line - 1
        if index < 0 or index >= len(lines):
            return {}
        if obj.kind == "require":
            _module_name, version, require_macro_pairs = _parse_require_invocation(lines[index])
            values = {key: value for key, value in require_macro_pairs}
            if version:
                values.setdefault("ECMC_VER", version)
            return values
        payload = _extract_script_call_macro_text(lines[index])
        pairs, _malformed = _parse_macro_payload(payload)
        return {key: value for key, value in pairs}

    def _available_command_macro_names(self, obj: StartupObject) -> List[str]:
        script_name = self._module_script_name_for_object(obj)
        if not script_name:
            return []
        macro_spec = self.inventory.module_macro_specs.get(script_name, MacroSpec(set(), set()))
        current_keys = set(self._current_object_macro_map(obj).keys())
        return sorted(key for key in macro_spec.allowed if key not in current_keys)

    def _prompt_inline_macro(
        self,
        obj: StartupObject,
        scope: str,
        preferred_name: str = "",
    ) -> Optional[Tuple[str, str]]:
        from tkinter import messagebox

        tk = self.tk
        ttk = self.ttk

        current_macros = self._current_object_macro_map(obj)
        available_names: List[str] = []
        if scope == "command":
            available_names = self._available_command_macro_names(obj)
            if not available_names:
                messagebox.showinfo(
                    "No macros available",
                    "All documented command macros for {} are already present.".format(obj.title),
                )
                return None

        dialog = tk.Toplevel(self.root)
        dialog.title("Add Macro")
        dialog.transient(self.root)
        dialog.withdraw()

        initial_name = preferred_name.strip().upper()
        if available_names:
            if initial_name not in available_names:
                initial_name = available_names[0]
        name_var = tk.StringVar(value=initial_name if initial_name else (available_names[0] if available_names else ""))
        value_var = tk.StringVar(value="")
        result: Dict[str, Optional[Tuple[str, str]]] = {"macro": None}
        next_row = 0

        if available_names:
            ttk.Label(dialog, text="Allowed macros").grid(row=next_row, column=0, sticky=tk.NW, padx=8, pady=4)
            allowed_text = tk.Text(
                dialog,
                width=52,
                height=min(8, max(2, (len(available_names) + 2) // 3)),
                wrap=tk.WORD,
                state=tk.NORMAL,
                relief=tk.FLAT,
                background=dialog.cget("background"),
            )
            allowed_text.grid(row=next_row, column=1, sticky=tk.EW, padx=8, pady=4)
            allowed_text.insert("1.0", ", ".join(available_names))
            allowed_text.configure(state=tk.DISABLED)
            next_row += 1

        ttk.Label(dialog, text="Macro name").grid(row=next_row, column=0, sticky=tk.W, padx=8, pady=4)
        if available_names:
            name_widget = ttk.Combobox(dialog, textvariable=name_var, values=available_names, width=48, state="readonly")
        else:
            name_widget = ttk.Entry(dialog, textvariable=name_var, width=52)
        name_widget.grid(row=next_row, column=1, sticky=tk.EW, padx=8, pady=4)
        next_row += 1

        ttk.Label(dialog, text="Macro value").grid(row=next_row, column=0, sticky=tk.W, padx=8, pady=4)
        value_widget = ttk.Entry(dialog, textvariable=value_var, width=52)
        value_widget.grid(row=next_row, column=1, sticky=tk.EW, padx=8, pady=4)
        next_row += 1

        def submit() -> None:
            name = name_var.get().strip().upper()
            value = value_var.get().strip()
            if not name:
                messagebox.showerror("Missing name", "Enter a macro name.", parent=dialog)
                return
            if not re.fullmatch(r"[A-Z0-9_]+", name):
                messagebox.showerror("Invalid name", "Macro names must match [A-Z0-9_]+.", parent=dialog)
                return
            if scope == "command" and name not in available_names:
                messagebox.showerror("Not allowed", "Macro '{}' is not allowed for this command.".format(name), parent=dialog)
                return
            if name in current_macros:
                messagebox.showerror("Already exists", "Macro '{}' is already present.".format(name), parent=dialog)
                return
            result["macro"] = (name, value)
            dialog.destroy()

        def cancel() -> None:
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=next_row, column=0, columnspan=2, sticky=tk.E, padx=8, pady=8)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Add", command=submit).pack(side=tk.RIGHT, padx=(0, 6))

        dialog.columnconfigure(1, weight=1)
        dialog.bind("<Escape>", lambda _event: cancel())
        dialog.bind("<Return>", lambda _event: submit())
        dialog.deiconify()
        dialog.wait_visibility()
        dialog.grab_set()
        name_widget.focus_set()
        self.root.wait_window(dialog)
        return result["macro"]

    def _render_updated_object_line(self, obj: StartupObject, macro_name: str, macro_value: str) -> str:
        resolved_target = obj.source.resolve()
        if resolved_target in self.file_buffers:
            content = self.file_buffers[resolved_target]
        elif resolved_target.exists():
            content = _read_text(resolved_target)
        else:
            raise IOError("Missing file: {}".format(resolved_target))

        lines = content.splitlines(True)
        index = obj.line - 1
        if index < 0 or index >= len(lines):
            raise IndexError("Line {} out of range in {}".format(obj.line, resolved_target))

        raw_line = lines[index]
        line = raw_line.rstrip("\r\n")
        leading_ws = re.match(r"\s*", line).group(0)
        command_name = _extract_command_name(line)
        command_target = _extract_script_target(line)
        if obj.kind == "require":
            module_name, version, require_macro_pairs = _parse_require_invocation(line)
            if not module_name:
                raise ValueError("Selected require line cannot accept inline macros.")
            if any(key == macro_name for key, _value in require_macro_pairs):
                raise ValueError("Macro '{}' already exists.".format(macro_name))
            if macro_name == "ECMC_VER" and version:
                raise ValueError("ECMC_VER is already defined by the require version token.")
            require_macro_pairs.append((macro_name, macro_value))
            payload = ", ".join(
                "{}={}".format(key, _quote_startup_value(value))
                for key, value in require_macro_pairs
            )
            version_part = " {}".format(version) if version else ""
            payload_part = ' "{}"'.format(payload) if payload else ""
            newline = "\n" if raw_line.endswith("\n") else ""
            return "{}require {}{}{}{}".format(leading_ws, module_name, version_part, payload_part, newline)
        if not command_name or not command_target:
            raise ValueError("Selected object line cannot accept inline macros.")

        payload_pairs, malformed = _parse_macro_payload(_extract_script_call_macro_text(line))
        if malformed:
            raise ValueError("Selected object line contains malformed macros.")
        if any(key == macro_name for key, _value in payload_pairs):
            raise ValueError("Macro '{}' already exists.".format(macro_name))
        payload_pairs.append((macro_name, macro_value))
        payload = ", ".join(
            "{}={}".format(key, _quote_startup_value(value))
            for key, value in payload_pairs
        )
        newline = "\n" if raw_line.endswith("\n") else ""
        return '{}{} {}, "{}"{}'.format(leading_ws, command_name, command_target, payload, newline)

    def _add_inline_macro(self, preferred_name: str = "") -> None:
        from tkinter import messagebox

        context = self._selected_inline_macro_context()
        if context is None:
            messagebox.showinfo("Invalid selection", "Select an object or macro section first.")
            return

        scope, obj = context
        macro = self._prompt_inline_macro(obj, scope, preferred_name=preferred_name)
        if macro is None:
            return

        try:
            new_line = self._render_updated_object_line(obj, macro[0], macro[1])
            self._rewrite_object_line(obj, new_line)
        except Exception as exc:
            messagebox.showerror("Add macro failed", str(exc))
            return

        self.status_var.set("Added macro {} to {}".format(macro[0], obj.title))
        self._refresh_startup_tree()

    def _add_startup_menu_item(self, menu, label: str, kind: str, before_selected: Optional[bool] = None) -> None:
        if before_selected is None:
            menu.add_command(label=label, command=lambda k=kind: self._insert_object_template(k))
            return
        menu.add_command(
            label=label,
            command=lambda k=kind, before=before_selected: self._insert_object_template(k, before_selected=before),
        )

    def _add_startup_menu_insert_group(self, menu, title: str, item_specs: List[Tuple[str, str, Optional[bool]]]) -> None:
        submenu = self.tk.Menu(menu, tearoff=0)
        for label, kind, before_selected in item_specs:
            self._add_startup_menu_item(submenu, label, kind, before_selected)
        menu.add_cascade(label=title, menu=submenu)

    def _add_startup_menu_position_group(
        self,
        menu,
        title: str,
        item_specs: List[Tuple[str, str, Optional[bool]]],
    ) -> None:
        before_menu = self.tk.Menu(menu, tearoff=0)
        after_menu = self.tk.Menu(menu, tearoff=0)
        for label, kind, _before_selected in item_specs:
            self._add_startup_menu_item(before_menu, label, kind, True)
            self._add_startup_menu_item(after_menu, label, kind, False)
        position_menu = self.tk.Menu(menu, tearoff=0)
        position_menu.add_cascade(label="Before", menu=before_menu)
        position_menu.add_cascade(label="After", menu=after_menu)
        menu.add_cascade(label=title, menu=position_menu)

    def _rebuild_startup_menu(self) -> None:
        self.startup_menu.delete(0, self.tk.END)
        entry = self._selected_startup_entry()
        selected_kind = self._selected_object_kind()
        entry_type = entry[0] if entry is not None else ""

        root_insert_items = [
            ("Slave", "slave", None),
            ("Slave Config", "slave_config", None),
            ("Apply Config", "apply_config", None),
            ("Axis", "axis", None),
            ("Configure Axis", "configured_axis", None),
            ("Macro", "macro", None),
            ("Master/Slave SM", "master_slave_sm", None),
            ("Subst Config", "subst_config", None),
            ("PVT Controller", "pvt_controller", None),
            ("LUT", "lut", None),
            ("Set Record Rate", "record_update_rate", None),
            ("Restore Record Rate", "restore_record_update_rate", None),
            ("ECMC Command", "ecmc_command", None),
            ("PLC", "plc", None),
            ("Plugin", "plugin", None),
            ("DataStorage", "datastorage", None),
            ("EcSdoRT", "ecsdo", None),
            ("EcDataItem", "ecdataitem", None),
        ]
        file_insert_items = [
            ("Add Slave", "slave", False),
            ("Add Slave Config", "slave_config", False),
            ("Apply Config", "apply_config", False),
            ("Add Axis", "axis", False),
            ("Configure Axis", "configured_axis", False),
            ("Add Macro", "macro", False),
            ("Add Master/Slave SM", "master_slave_sm", False),
            ("Add Subst Config", "subst_config", False),
            ("Add PVT Controller", "pvt_controller", False),
            ("Add LUT", "lut", False),
            ("Set Record Rate", "record_update_rate", False),
            ("Restore Record Rate", "restore_record_update_rate", False),
            ("Add ECMC Command", "ecmc_command", False),
            ("Add PLC", "plc", False),
            ("Add Plugin", "plugin", False),
            ("Add DataStorage", "datastorage", False),
            ("Add EcSdoRT", "ecsdo", False),
            ("Add EcDataItem", "ecdataitem", False),
        ]

        if entry is None:
            self._add_startup_menu_insert_group(self.startup_menu, "Add", file_insert_items)
            return

        if entry[0] == "file":
            self._add_startup_menu_insert_group(self.startup_menu, "Add", file_insert_items)
            return

        if self._selected_inline_macro_context() is not None:
            self.startup_menu.add_command(label="Add Macro", command=self._add_inline_macro)
            self.startup_menu.add_separator()

        if entry_type == "object":
            self.startup_menu.add_command(label="Copy Object", command=self._copy_selected_object)
            if self.copied_object_text:
                self.startup_menu.add_command(label="Paste Object", command=self._paste_copied_object)
            self.startup_menu.add_separator()
        elif entry_type == "file" and self.copied_object_text:
            self.startup_menu.add_command(label="Paste Object", command=self._paste_copied_object)
            self.startup_menu.add_separator()

        if selected_kind in {"slave", "axis", "plc"}:
            child_menu = self.tk.Menu(self.startup_menu, tearoff=0)
            if selected_kind == "slave":
                self._add_startup_menu_item(child_menu, "Add Slave Config", "slave_config")
                self._add_startup_menu_item(child_menu, "Add Component", "component")
            elif selected_kind == "axis":
                self._add_startup_menu_item(child_menu, "Add Encoder", "encoder")
            elif selected_kind == "plc":
                self._add_startup_menu_item(child_menu, "Add PLC Analog", "plcvar_analog")
                self._add_startup_menu_item(child_menu, "Add PLC Binary", "plcvar_binary")
            self.startup_menu.add_cascade(label="Add Child", menu=child_menu)

        self._add_startup_menu_position_group(self.startup_menu, "Insert", root_insert_items)

        if selected_kind in self._editable_object_kinds():
            self.startup_menu.add_separator()
            self.startup_menu.add_command(label="Edit Object", command=self._edit_selected_object)
            self.startup_menu.add_command(label="Remove Object", command=self._remove_selected_object)

    def _on_startup_tree_right_click(self, event) -> str:
        row_id = self.startup_tree.identify_row(event.y)
        if row_id:
            self.startup_tree.selection_set(row_id)
            self.startup_tree.focus(row_id)
        self._rebuild_startup_menu()
        try:
            self.startup_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.startup_menu.grab_release()
        return "break"

    def _next_tree_id(self, kind: str, attr_name: str) -> str:
        max_value = -1
        if self.latest_startup_tree is None:
            return "0"
        for file_node in self.latest_startup_tree.files:
            for obj in file_node.objects:
                if obj.kind != kind:
                    continue
                for key, value in obj.details:
                    if key == attr_name:
                        parsed = _parse_int_value(value)
                        if parsed is not None:
                            max_value = max(max_value, parsed)
        return str(max_value + 1 if max_value >= 0 else 0)

    def _object_detail_map(self, obj: StartupObject) -> Dict[str, str]:
        values = dict(obj.details)
        for key, value in obj.command_details:
            values[key] = value
        for key, value in obj.linked_file_details:
            values[key] = value
        if obj.linked_file is not None and "FILE" not in values:
            values["FILE"] = self._relative_display(obj.linked_file)
        return values

    def _available_slave_hw_descs(self) -> List[str]:
        excluded_dirs = {"Encoders", "Motors", "Sensors"}
        allowed: List[str] = []
        for file_name, paths in self.inventory.hardware_configs.items():
            stem = Path(file_name).stem
            if not stem.lower().startswith("ecmc") or len(stem) <= 4:
                continue
            hw_desc = stem[4:]
            if hw_desc not in self.inventory.hardware_descs:
                continue
            exclude = False
            for path in paths:
                if self.inventory.ecmccfg_root is None:
                    continue
                try:
                    relative = path.resolve().relative_to((self.inventory.ecmccfg_root / "hardware").resolve())
                except Exception:
                    continue
                if relative.parts and relative.parts[0] in excluded_dirs:
                    exclude = True
                    break
            if not exclude:
                allowed.append(hw_desc)
        return sorted(set(allowed))

    def _available_hardware_config_values(self, hw_desc: str) -> List[str]:
        hw_desc = str(hw_desc or "").strip()
        if not hw_desc:
            return []
        prefix = "ecmc{}".format(hw_desc)
        values: List[str] = []
        for file_name in sorted(self.inventory.hardware_configs):
            stem = Path(file_name).stem
            if not stem.startswith(prefix):
                continue
            suffix = stem[len(prefix) :]
            if suffix:
                values.append(suffix)
        return sorted(set(values))

    def _available_ecmc_command_syntaxes(self) -> List[str]:
        syntaxes: List[str] = []
        for command_name in sorted(self.inventory.ecmc_command_definitions):
            for definition in self.inventory.ecmc_command_definitions.get(command_name, []):
                if definition.syntax and definition.syntax not in syntaxes:
                    syntaxes.append(definition.syntax)
        return syntaxes

    def _ecmc_command_hint(self, command_text: str) -> str:
        definitions = _lookup_ecmc_command_definitions(command_text, self.inventory)
        if not definitions:
            return "Unknown ECMC command"
        definition = definitions[0]
        parts: List[str] = []
        if definition.function_name:
            parts.append(definition.function_name)
        if definition.brief:
            parts.append(definition.brief)
        return " | ".join(parts) if parts else "Known ECMC command"

    def _startup_root_dir(self) -> Path:
        startup_value = self.startup_var.get().strip()
        if startup_value:
            startup_path = Path(startup_value).expanduser()
            if not startup_path.is_absolute():
                startup_path = startup_path.resolve()
            return startup_path.parent.resolve()
        return Path.cwd().resolve()

    def _candidate_project_files(self, suffixes: Set[str], limit: int = 200) -> List[str]:
        root_dir = self._startup_root_dir()
        candidates: List[str] = []
        seen: Set[str] = set()
        for path in root_dir.rglob("*"):
            if len(candidates) >= limit:
                break
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            try:
                relative = path.resolve().relative_to(root_dir)
                display = "./{}".format(relative.as_posix())
            except Exception:
                display = str(path.resolve())
            if display not in seen:
                seen.add(display)
                candidates.append(display)
        return sorted(candidates)

    def _known_object_values(self, kind: str, key_name: str) -> List[str]:
        values: Set[str] = set()
        if self.latest_startup_tree is None:
            return []
        for file_node in self.latest_startup_tree.files:
            for obj in file_node.objects:
                if obj.kind != kind:
                    continue
                for key, value in obj.details:
                    if key == key_name and str(value).strip():
                        values.add(str(value).strip())
        return sorted(values, key=lambda value: (_parse_int_value(value) is None, _parse_int_value(value) or 0, value))

    def _known_slave_ids(self) -> List[str]:
        return self._known_object_values("slave", "SLAVE_ID")

    def _known_plc_ids(self) -> List[str]:
        return self._known_object_values("plc", "PLC_ID")

    def _known_axis_ids(self) -> List[str]:
        return self._known_object_values("axis", "AXIS_ID")

    def _known_master_ids(self) -> List[str]:
        values = self._known_object_values("master", "MASTER_ID")
        return values or ["0"]

    def _known_plugin_ids(self) -> List[str]:
        return self._known_object_values("plugin", "PLUGIN_ID")

    def _known_datastorage_ids(self) -> List[str]:
        return self._known_object_values("datastorage", "DS_ID")

    def _filter_picker_values(self, values: List[str], current: str) -> List[str]:
        if not current:
            return list(values)
        lowered = current.lower()
        startswith_matches = [value for value in values if value.lower().startswith(lowered)]
        contains_matches = [value for value in values if lowered in value.lower() and value not in startswith_matches]
        return startswith_matches + contains_matches

    def _resolve_ecmc_numeric_token(self, token: str) -> Optional[int]:
        cleaned = token.strip()
        if not cleaned:
            return None
        macro_map = self._startup_known_macro_map()
        expanded = _expand_text_macros(cleaned, macro_map).strip()
        return _parse_int_value(expanded)

    def _ecmc_dynamic_param_kind(self, param_name: str) -> str:
        lowered = param_name.lower()
        if "entryidstring" in lowered or lowered in {"entryid", "entry_id"}:
            return "entry_id"
        if lowered in {"ecpath", "ecentrypathstring"} or "ecpath" in lowered:
            return "ec_path"
        return ""

    def _ecmc_slave_token_from_values(
        self,
        params: List[Tuple[str, str]],
        current_values: List[str],
    ) -> str:
        for index, (_param_type, param_name) in enumerate(params):
            lowered = param_name.lower()
            if any(token in lowered for token in {"slave", "slavepos", "slaveposition", "busposition"}):
                if index < len(current_values):
                    return current_values[index].strip()
        return ""

    def _ecmc_master_token_from_values(
        self,
        params: List[Tuple[str, str]],
        current_values: List[str],
    ) -> str:
        for index, (_param_type, param_name) in enumerate(params):
            if "master" in param_name.lower():
                if index < len(current_values):
                    return current_values[index].strip()
        known_master_ids = self._known_master_ids()
        return known_master_ids[0] if known_master_ids else "0"

    def _ecmc_entry_choices_for_slave_token(self, slave_token: str) -> List[str]:
        slave_id = self._resolve_ecmc_numeric_token(slave_token)
        if slave_id is None:
            return []
        hw_desc = self._startup_slave_hw_desc_map().get(slave_id, "")
        if not hw_desc:
            return []
        return sorted(self.inventory.hardware_entries.get(hw_desc, set()))

    def _ecmc_all_ec_path_choices(self, master_token: str = "", slave_filter: str = "") -> List[str]:
        master_id = self._resolve_ecmc_numeric_token(master_token)
        if master_id is None:
            master_id = 0
        paths: List[str] = []
        slave_map = self._startup_slave_hw_desc_map()
        for slave_id in sorted(slave_map):
            if slave_filter and str(slave_id) != slave_filter.strip():
                continue
            hw_desc = slave_map.get(slave_id, "")
            for entry_name in sorted(self.inventory.hardware_entries.get(hw_desc, set())):
                paths.append("ec{}.s{}.{}".format(master_id, slave_id, entry_name))
        return paths

    def _ecmc_ec_path_choices(self, current_value: str) -> List[str]:
        cleaned = current_value.strip()
        if not cleaned:
            return self._ecmc_all_ec_path_choices()
        match = re.match(r"(?P<prefix>ec(?P<master>[^.]+)\.s(?P<slave>[^.]+)\.)(?P<entry>[A-Za-z0-9_]*)$", cleaned)
        if not match:
            return self._filter_picker_values(self._ecmc_all_ec_path_choices(), cleaned)
        prefix = match.group("prefix")
        slave_token = match.group("slave")
        entry_prefix = match.group("entry")
        entries = self._ecmc_entry_choices_for_slave_token(slave_token)
        return [prefix + entry_name for entry_name in self._filter_picker_values(entries, entry_prefix)]

    def _parse_ecmc_command_signature(self, syntax: str) -> Tuple[str, List[Tuple[str, str]]]:
        cleaned = syntax.strip()
        if cleaned.startswith("Cfg."):
            cleaned = cleaned[4:]
        match = re.match(r"(?P<name>[A-Za-z_][A-Za-z0-9_?]*)\((?P<args>.*)\)$", cleaned)
        if not match:
            return cleaned.split("=", 1)[0].strip(), []
        params: List[Tuple[str, str]] = []
        for token in _split_top_level(match.group("args")):
            item = token.strip().lstrip(",").strip()
            if not item:
                continue
            parts = item.rsplit(" ", 1)
            if len(parts) == 2:
                param_type, param_name = parts[0].strip(), parts[1].strip()
            else:
                param_type, param_name = "", parts[0].strip()
            params.append((param_type, param_name))
        return match.group("name"), params

    def _split_ecmc_command_arguments(self, command_text: str) -> List[str]:
        cleaned = _strip_wrapper_pairs(_normalize_value(command_text)).strip()
        if cleaned.startswith("Cfg."):
            cleaned = cleaned[4:]
        start = cleaned.find("(")
        end = cleaned.rfind(")")
        if start < 0 or end < start:
            return []
        return [token.strip() for token in _split_top_level(cleaned[start + 1 : end])]

    def _ecmc_parameter_choices(self, command_name: str, param_name: str, param_type: str) -> List[str]:
        lowered_name = param_name.lower()
        lowered_type = param_type.lower()
        if "axis" in lowered_name:
            return self._known_axis_ids()
        if any(token in lowered_name for token in {"slave", "slavepos", "slaveposition", "busposition"}):
            return self._known_slave_ids()
        if "master" in lowered_name:
            return self._known_master_ids()
        if "plc" in lowered_name:
            return self._known_plc_ids()
        if "plugin" in lowered_name:
            return self._known_plugin_ids()
        if "storage" in lowered_name or lowered_name in {"ds_id", "dsid"}:
            return self._known_datastorage_ids()
        if "datatype" in lowered_name or lowered_name == "dt":
            return ["BIT", "U8", "S8", "U16", "S16", "U32", "S32", "U64", "S64", "F32", "F64"]
        if "bytesize" in lowered_name or lowered_name == "bits":
            return ["1", "2", "4", "8", "16", "32", "64"]
        if any(token in lowered_name for token in {"enable", "allow", "block", "use", "done"}):
            return ["0", "1"]
        if lowered_type in {"int", "uint16_t", "uint32_t", "uint8_t"} and lowered_name in {"value", "mode"}:
            return ["0", "1"]
        return []

    def _build_ecmc_command_text(self, syntax: str, arguments: List[str]) -> str:
        cleaned = syntax.strip()
        if cleaned.startswith("Cfg."):
            cleaned = cleaned[4:]
        match = re.match(r"(?P<name>[A-Za-z_][A-Za-z0-9_?]*)\((?P<args>.*)\)$", cleaned)
        if not match:
            return syntax.strip()
        return "Cfg.{}({})".format(match.group("name"), ",".join(arg.strip() for arg in arguments))

    def _default_insert_slave_id(self, target_path: Path, content: str, insert_line: int) -> str:
        objects, _nested = _extract_startup_objects_from_file(
            target_path.resolve(),
            content,
            self.inventory,
            self.file_buffers,
        )
        previous_slave_id: Optional[int] = None
        previous_slave_line = -1
        for obj in objects:
            if obj.kind != "slave" or obj.slave_id is None:
                continue
            if obj.line < insert_line and obj.line > previous_slave_line:
                previous_slave_line = obj.line
                previous_slave_id = obj.slave_id
        if previous_slave_id is None:
            return "0"
        return str(previous_slave_id + 1)

    def _selected_editable_object(self) -> Optional[StartupObject]:
        entry = self._selected_startup_entry()
        if entry is None:
            return None
        entry_type, payload = entry
        if entry_type == "file" or not isinstance(payload, StartupObject):
            return None
        return payload

    def _find_slave_object(self, source: Path, slave_id: Optional[int]) -> Optional[StartupObject]:
        if slave_id is None or self.latest_startup_tree is None:
            return None
        resolved_source = source.resolve()
        for file_node in self.latest_startup_tree.files:
            for obj in file_node.objects:
                if obj.kind != "slave" or obj.slave_id != slave_id:
                    continue
                if obj.source.resolve() == resolved_source:
                    return obj
        return None

    def _context_slave_for_selected_entry(self) -> Optional[StartupObject]:
        entry = self._selected_startup_entry()
        if entry is None:
            return None
        entry_type, payload = entry
        if entry_type == "file" or not isinstance(payload, StartupObject):
            return None
        if payload.kind == "slave":
            return payload
        if payload.parent_slave_id is not None:
            return self._find_slave_object(payload.source, payload.parent_slave_id)
        return None

    def _context_slave_for_component_dialog(self) -> Optional[StartupObject]:
        entry = self._selected_startup_entry()
        if entry is None:
            return None
        entry_type, payload = entry
        if entry_type == "file" or not isinstance(payload, StartupObject):
            return None
        if payload.kind == "component":
            return self._find_slave_object(payload.source, payload.parent_slave_id)
        return self._context_slave_for_selected_entry()

    def _component_dialog_context(self, initial_values: Dict[str, str]) -> Dict[str, object]:
        slave_obj = self._context_slave_for_component_dialog()
        hw_desc = ""
        default_slave_id = initial_values.get("COMP_S_ID", "")
        if slave_obj is not None:
            hw_desc = dict(slave_obj.details).get("HW_DESC", "").strip()
            if not default_slave_id and slave_obj.slave_id is not None:
                default_slave_id = str(slave_obj.slave_id)
        comp_hw_type = initial_values.get("EC_COMP_TYPE", "").strip()
        if not comp_hw_type:
            comp_hw_type = self.inventory.hardware_component_types.get(hw_desc, hw_desc)
        support_map = self.inventory.component_support.get(comp_hw_type, {})
        supported_components = sorted(
            definition.name
            for definition in self.inventory.component_definitions.values()
            if definition.comp_type in support_map
        )
        initial_component = initial_values.get("COMP", "").strip()
        if initial_component and initial_component not in supported_components:
            supported_components.append(initial_component)
            supported_components.sort()
        return {
            "slave_obj": slave_obj,
            "hw_desc": hw_desc,
            "comp_hw_type": comp_hw_type,
            "default_slave_id": default_slave_id,
            "support_map": support_map,
            "supported_components": supported_components,
        }

    def _prompt_for_component_values(self, initial_values: Dict[str, str], mode: str = "add") -> Optional[str]:
        from tkinter import messagebox

        tk = self.tk
        ttk = self.ttk
        context = self._component_dialog_context(initial_values)
        hw_desc = str(context["hw_desc"])
        comp_hw_type = str(context["comp_hw_type"])
        support_map = dict(context["support_map"])
        supported_components = list(context["supported_components"])
        is_add_from_slave = context["slave_obj"] is not None and not initial_values.get("COMP", "").strip()

        if is_add_from_slave and self.inventory.component_support and not support_map:
            messagebox.showinfo(
                "No supported components",
                "No component support was found in ecmccomp for HW_DESC '{}' (EC_COMP_TYPE '{}').".format(
                    hw_desc or "<unknown>",
                    comp_hw_type or "<unset>",
                ),
            )
            return None

        action_label = "Edit" if mode == "edit" else "Add"
        dialog = tk.Toplevel(self.root)
        dialog.title("{} Component".format(action_label))
        dialog.transient(self.root)
        dialog.withdraw()

        comp_var = tk.StringVar(value=initial_values.get("COMP", supported_components[0] if supported_components else ""))
        ec_comp_type_var = tk.StringVar(value=comp_hw_type)
        comp_s_id_var = tk.StringVar(value=initial_values.get("COMP_S_ID", str(context["default_slave_id"])))
        ch_id_var = tk.StringVar(value=initial_values.get("CH_ID", "1"))
        macros_var = tk.StringVar(value=initial_values.get("MACROS", ""))
        slave_var = tk.StringVar(
            value="{} -> {}".format(hw_desc or "<unknown HW_DESC>", comp_hw_type or "<unset>")
            if hw_desc or comp_hw_type
            else "No slave selected"
        )
        supported_types_var = tk.StringVar(value="")
        component_type_var = tk.StringVar(value="")
        supported_macros_var = tk.StringVar(value="")
        channels_var = tk.StringVar(value="")
        warning_var = tk.StringVar(value="")
        result: Dict[str, Optional[str]] = {"command": None}

        ttk.Label(dialog, text="Slave").grid(row=0, column=0, sticky=tk.NW, padx=8, pady=4)
        ttk.Label(dialog, textvariable=slave_var, justify=tk.LEFT, wraplength=460).grid(
            row=0, column=1, sticky=tk.W, padx=8, pady=4
        )

        ttk.Label(dialog, text="EC_COMP_TYPE").grid(row=1, column=0, sticky=tk.W, padx=8, pady=4)
        ec_comp_type_widget = ttk.Combobox(
            dialog,
            textvariable=ec_comp_type_var,
            values=sorted(self.inventory.component_support.keys()),
            width=48,
        )
        ec_comp_type_widget.grid(row=1, column=1, sticky=tk.EW, padx=8, pady=4)

        ttk.Label(dialog, text="Supported types").grid(row=2, column=0, sticky=tk.NW, padx=8, pady=4)
        ttk.Label(dialog, textvariable=supported_types_var, justify=tk.LEFT, wraplength=460).grid(
            row=2, column=1, sticky=tk.W, padx=8, pady=4
        )

        ttk.Label(dialog, text="COMP").grid(row=3, column=0, sticky=tk.W, padx=8, pady=4)
        comp_widget = ttk.Combobox(dialog, textvariable=comp_var, values=supported_components, width=48)
        if supported_components:
            comp_widget.configure(state="readonly")
        comp_widget.grid(row=3, column=1, sticky=tk.EW, padx=8, pady=4)

        ttk.Label(dialog, text="Resolved type").grid(row=4, column=0, sticky=tk.NW, padx=8, pady=4)
        ttk.Label(dialog, textvariable=component_type_var, justify=tk.LEFT, wraplength=460).grid(
            row=4, column=1, sticky=tk.W, padx=8, pady=4
        )

        ttk.Label(dialog, text="COMP_S_ID").grid(row=5, column=0, sticky=tk.W, padx=8, pady=4)
        comp_s_id_widget = ttk.Entry(dialog, textvariable=comp_s_id_var, width=52)
        comp_s_id_widget.grid(row=5, column=1, sticky=tk.EW, padx=8, pady=4)

        ttk.Label(dialog, text="CH_ID").grid(row=6, column=0, sticky=tk.W, padx=8, pady=4)
        ch_id_widget = ttk.Entry(dialog, textvariable=ch_id_var, width=52)
        ch_id_widget.grid(row=6, column=1, sticky=tk.EW, padx=8, pady=4)

        ttk.Label(dialog, text="Allowed macros").grid(row=7, column=0, sticky=tk.NW, padx=8, pady=4)
        ttk.Label(dialog, textvariable=supported_macros_var, justify=tk.LEFT, wraplength=460).grid(
            row=7, column=1, sticky=tk.W, padx=8, pady=4
        )

        ttk.Label(dialog, text="Channels").grid(row=8, column=0, sticky=tk.NW, padx=8, pady=4)
        ttk.Label(dialog, textvariable=channels_var, justify=tk.LEFT, wraplength=460).grid(
            row=8, column=1, sticky=tk.W, padx=8, pady=4
        )

        ttk.Label(dialog, text="MACROS").grid(row=9, column=0, sticky=tk.W, padx=8, pady=4)
        macros_widget = ttk.Entry(dialog, textvariable=macros_var, width=52)
        macros_widget.grid(row=9, column=1, sticky=tk.EW, padx=8, pady=4)

        warning_label = ttk.Label(dialog, textvariable=warning_var, justify=tk.LEFT, foreground="#8b5e00", wraplength=460)
        warning_label.grid(row=10, column=0, columnspan=2, sticky=tk.W, padx=8, pady=(0, 4))

        def refresh_component_context(*_args) -> None:
            current_ec_comp_type = ec_comp_type_var.get().strip()
            current_support_map = self.inventory.component_support.get(current_ec_comp_type, {})
            type_names = sorted(current_support_map.keys())
            supported_types_var.set(", ".join(type_names) if type_names else "No component types found in ecmccomp")

            current_components = sorted(
                definition.name
                for definition in self.inventory.component_definitions.values()
                if definition.comp_type in current_support_map
            )
            current_component = comp_var.get().strip()
            if current_components:
                if current_component and current_component not in current_components:
                    comp_widget.configure(values=current_components + [current_component], state="normal")
                else:
                    comp_widget.configure(values=current_components, state="readonly")
                    if current_component not in current_components:
                        comp_var.set(current_components[0])
            else:
                comp_widget.configure(values=(), state="normal")

            component_name = comp_var.get().strip()
            definition = self.inventory.component_definitions.get(component_name)
            component_type = definition.comp_type if definition is not None else ""
            component_type_var.set(component_type or "Unknown component type")

            support = current_support_map.get(component_type) if component_type else None
            if support is not None and support.supported_macros:
                supported_macros_var.set(", ".join(sorted(support.supported_macros)))
            elif support is not None:
                supported_macros_var.set("(no special macros)")
            else:
                supported_macros_var.set("No macro information available")

            if support is not None and support.channel_count is not None:
                channels_var.set(str(support.channel_count))
            elif support is not None:
                channels_var.set("Unknown")
            else:
                channels_var.set("-")

            if definition is None and component_name:
                warning_var.set("Component '{}' was not found in ecmccomp.".format(component_name))
            elif component_type and component_type not in current_support_map:
                warning_var.set(
                    "Component type '{}' is not supported by EC_COMP_TYPE '{}'.".format(
                        component_type,
                        current_ec_comp_type or "<unset>",
                    )
                )
            else:
                warning_var.set("")

        def submit() -> None:
            component_name = comp_var.get().strip()
            current_ec_comp_type = ec_comp_type_var.get().strip()
            current_support_map = self.inventory.component_support.get(current_ec_comp_type, {})
            definition = self.inventory.component_definitions.get(component_name)

            if not component_name:
                messagebox.showerror("Missing component", "Select a component.", parent=dialog)
                return
            if self.inventory.component_definitions and definition is None:
                messagebox.showerror(
                    "Unknown component",
                    "Component '{}' was not found in ecmccomp.".format(component_name),
                    parent=dialog,
                )
                return
            if definition is not None and current_support_map and definition.comp_type not in current_support_map:
                messagebox.showerror(
                    "Unsupported component",
                    "Component '{}' of type '{}' is not supported by EC_COMP_TYPE '{}'.".format(
                        component_name,
                        definition.comp_type,
                        current_ec_comp_type or "<unset>",
                    ),
                    parent=dialog,
                )
                return

            macro_pairs, malformed = _parse_macro_payload(macros_var.get().strip())
            if malformed:
                messagebox.showerror(
                    "Invalid macros",
                    "Could not parse: {}".format(", ".join(malformed)),
                    parent=dialog,
                )
                return
            if definition is not None:
                support = current_support_map.get(definition.comp_type)
                allowed_macros = support.supported_macros if support is not None else set()
                invalid_macros = sorted(key for key, _value in macro_pairs if allowed_macros and key not in allowed_macros)
                if invalid_macros:
                    messagebox.showerror(
                        "Unsupported macros",
                        "These macros are not supported for {} on {}: {}".format(
                            component_name,
                            current_ec_comp_type or "<unset>",
                            ", ".join(invalid_macros),
                        ),
                        parent=dialog,
                    )
                    return

            items = [
                ("COMP", component_name),
                ("EC_COMP_TYPE", current_ec_comp_type),
                ("COMP_S_ID", comp_s_id_var.get().strip()),
                ("CH_ID", ch_id_var.get().strip()),
                ("MACROS", macros_var.get().strip()),
            ]
            result["command"] = _render_startup_command("applyComponent.cmd", items)
            dialog.destroy()

        def cancel() -> None:
            dialog.destroy()

        ec_comp_type_var.trace_add("write", refresh_component_context)
        comp_var.trace_add("write", refresh_component_context)
        refresh_component_context()

        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=11, column=0, columnspan=2, sticky=tk.E, padx=8, pady=8)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text=action_label, command=submit).pack(side=tk.RIGHT, padx=(0, 6))

        dialog.columnconfigure(1, weight=1)
        dialog.bind("<Escape>", lambda _event: cancel())
        dialog.bind("<Return>", lambda _event: submit())
        dialog.deiconify()
        dialog.wait_visibility()
        dialog.grab_set()
        comp_widget.focus_set()
        self.root.wait_window(dialog)
        return result["command"]

    def _prompt_for_ecmc_command_values(self, initial_values: Dict[str, str], mode: str = "add") -> Optional[str]:
        from tkinter import messagebox

        tk = self.tk
        ttk = self.ttk
        action_label = "Edit" if mode == "edit" else "Add"

        dialog = tk.Toplevel(self.root)
        dialog.title("{} ECMC Command".format(action_label))
        dialog.transient(self.root)
        dialog.withdraw()

        syntax_values = self._available_ecmc_command_syntaxes()
        wrapper_var = tk.StringVar(value=initial_values.get("WRAPPER", "ecmcConfigOrDie").strip() or "ecmcConfigOrDie")
        command_var = tk.StringVar(value=initial_values.get("COMMAND", "").strip())
        syntax_var = tk.StringVar(value="")
        hint_var = tk.StringVar(value="")
        preview_var = tk.StringVar(value="")
        result: Dict[str, Optional[str]] = {"command": None}

        current_command = command_var.get().strip()
        matching_syntax = ""
        current_name = _extract_ecmc_inner_command_name(current_command)
        if current_name:
            definitions = self.inventory.ecmc_command_definitions.get(current_name, [])
            if definitions:
                matching_syntax = definitions[0].syntax
        if not matching_syntax and syntax_values:
            matching_syntax = syntax_values[0]
        syntax_var.set(matching_syntax)

        ttk.Label(dialog, text="Wrapper").grid(row=0, column=0, sticky=tk.W, padx=8, pady=4)
        wrapper_widget = ttk.Combobox(
            dialog,
            textvariable=wrapper_var,
            values=["ecmcConfigOrDie", "ecmcConfig"],
            width=48,
            state="readonly",
        )
        wrapper_widget.grid(row=0, column=1, sticky=tk.EW, padx=8, pady=4)

        ttk.Label(dialog, text="Syntax").grid(row=1, column=0, sticky=tk.NW, padx=8, pady=4)
        syntax_container, syntax_focus_widget = self._create_filtered_value_picker(
            dialog,
            syntax_var,
            syntax_values,
            width=48,
            height=10,
        )
        syntax_container.grid(row=1, column=1, sticky=tk.EW, padx=8, pady=4)

        ttk.Label(dialog, text="Hint").grid(row=2, column=0, sticky=tk.NW, padx=8, pady=4)
        ttk.Label(dialog, textvariable=hint_var, justify=tk.LEFT, wraplength=520).grid(
            row=2, column=1, sticky=tk.W, padx=8, pady=4
        )

        params_frame = ttk.Frame(dialog)
        params_frame.grid(row=3, column=0, columnspan=2, sticky=tk.EW, padx=8, pady=(4, 0))
        params_frame.columnconfigure(1, weight=1)
        dialog.columnconfigure(1, weight=1)

        ttk.Label(dialog, text="Preview").grid(row=4, column=0, sticky=tk.NW, padx=8, pady=4)
        ttk.Label(dialog, textvariable=preview_var, justify=tk.LEFT, wraplength=520).grid(
            row=4, column=1, sticky=tk.W, padx=8, pady=4
        )

        param_vars: List[tk.StringVar] = []
        param_refreshers: List[object] = []
        raw_arg_values = self._split_ecmc_command_arguments(current_command)

        def rebuild_preview(*_args) -> None:
            syntax = syntax_var.get().strip()
            if not syntax:
                preview_var.set(command_var.get().strip())
                return
            values = [var.get().strip() for var in param_vars]
            current_preview = self._build_ecmc_command_text(syntax, values)
            command_var.set(current_preview)
            preview_var.set(current_preview)

        def rebuild_params(*_args) -> None:
            for child in params_frame.winfo_children():
                child.destroy()
            param_vars.clear()
            param_refreshers.clear()

            syntax = syntax_var.get().strip()
            hint_var.set(self._ecmc_command_hint(syntax))
            command_name, params = self._parse_ecmc_command_signature(syntax)
            existing_args = self._split_ecmc_command_arguments(command_var.get().strip())
            if not existing_args:
                existing_args = raw_arg_values

            if not params:
                ttk.Label(
                    params_frame,
                    text="No parsed parameters for this syntax. The preview will use the selected syntax as-is.",
                    justify=tk.LEFT,
                    wraplength=520,
                ).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=4)
                rebuild_preview()
                return

            for row, (param_type, param_name) in enumerate(params):
                ttk.Label(params_frame, text=param_name).grid(row=row, column=0, sticky=tk.W, pady=4)
                default_value = existing_args[row] if row < len(existing_args) else ""
                value_var = tk.StringVar(value=default_value)
                param_vars.append(value_var)
                dynamic_kind = self._ecmc_dynamic_param_kind(param_name)

                if dynamic_kind:
                    picker_frame = ttk.Frame(params_frame)
                    picker_frame.grid(row=row, column=1, sticky=tk.EW, pady=4)
                    picker_frame.columnconfigure(0, weight=1)
                    picker_frame.rowconfigure(1, weight=1)

                    entry_widget = ttk.Entry(picker_frame, textvariable=value_var, width=52)
                    entry_widget.grid(row=0, column=0, sticky=tk.EW)

                    list_container = ttk.Frame(picker_frame)
                    list_container.grid(row=1, column=0, sticky=tk.EW, pady=(4, 0))
                    list_container.columnconfigure(0, weight=1)
                    list_container.rowconfigure(0, weight=1)

                    listbox = tk.Listbox(
                        list_container,
                        exportselection=False,
                        height=6,
                        activestyle="none",
                    )
                    listbox.grid(row=0, column=0, sticky=tk.EW)
                    scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=listbox.yview)
                    scrollbar.grid(row=0, column=1, sticky=tk.NS)
                    listbox.configure(yscrollcommand=scrollbar.set)

                    def refresh_dynamic_list(
                        _event=None,
                        target_var=value_var,
                        target_listbox=listbox,
                        target_kind=dynamic_kind,
                        target_index=row,
                    ) -> None:
                        current_values = [var.get().strip() for var in param_vars]
                        if target_kind == "entry_id":
                            slave_token = self._ecmc_slave_token_from_values(params, current_values)
                            base_choices = self._ecmc_entry_choices_for_slave_token(slave_token)
                            choices = self._filter_picker_values(base_choices, target_var.get().strip())
                        else:
                            choices = self._ecmc_ec_path_choices(target_var.get().strip())
                        target_listbox.delete(0, tk.END)
                        for item in choices[:200]:
                            target_listbox.insert(tk.END, item)
                        if choices:
                            target_listbox.selection_clear(0, tk.END)
                            target_listbox.selection_set(0)
                            target_listbox.activate(0)

                    def accept_dynamic_choice(
                        _event=None,
                        target_var=value_var,
                        target_listbox=listbox,
                    ) -> str:
                        selection = target_listbox.curselection()
                        if not selection:
                            return "break"
                        target_var.set(str(target_listbox.get(selection[0])))
                        return "break"

                    entry_widget.bind("<KeyRelease>", refresh_dynamic_list, add="+")
                    listbox.bind("<ButtonRelease-1>", accept_dynamic_choice, add="+")
                    listbox.bind("<Double-Button-1>", accept_dynamic_choice, add="+")
                    listbox.bind("<Return>", accept_dynamic_choice, add="+")
                    param_refreshers.append(refresh_dynamic_list)
                    refresh_dynamic_list()
                    value_var.trace_add("write", rebuild_preview)
                else:
                    choices = self._ecmc_parameter_choices(command_name, param_name, param_type)
                    if choices:
                        widget = ttk.Combobox(params_frame, textvariable=value_var, values=choices, width=48)
                        self._enable_combobox_filter(widget, choices)
                    else:
                        widget = ttk.Entry(params_frame, textvariable=value_var, width=52)
                    widget.grid(row=row, column=1, sticky=tk.EW, pady=4)
                    value_var.trace_add("write", rebuild_preview)

            for value_var in param_vars:
                value_var.trace_add("write", lambda *_args: [refresher() for refresher in param_refreshers])

            rebuild_preview()

        def submit() -> None:
            syntax = syntax_var.get().strip()
            if not syntax:
                messagebox.showerror("Missing syntax", "Select an ECMC command syntax.", parent=dialog)
                return

            _command_name, params = self._parse_ecmc_command_signature(syntax)
            values = [var.get().strip() for var in param_vars]
            if params:
                missing = [params[index][1] for index, value in enumerate(values) if not value]
                if missing:
                    messagebox.showerror(
                        "Missing parameters",
                        "Enter values for: {}".format(", ".join(missing)),
                        parent=dialog,
                    )
                    return
                command_text = self._build_ecmc_command_text(syntax, values)
            else:
                command_text = syntax
            command_text = _normalize_ecmc_command_text(command_text)

            wrapper_name = wrapper_var.get().strip() or "ecmcConfigOrDie"
            result["command"] = '{} "{}"\n'.format(wrapper_name, command_text.replace('"', '\\"'))
            dialog.destroy()

        def cancel() -> None:
            dialog.destroy()

        syntax_var.trace_add("write", rebuild_params)
        wrapper_var.trace_add("write", rebuild_preview)
        rebuild_params()

        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=5, column=0, columnspan=2, sticky=tk.E, padx=8, pady=8)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text=action_label, command=submit).pack(side=tk.RIGHT, padx=(0, 6))

        dialog.bind("<Escape>", lambda _event: cancel())
        dialog.deiconify()
        dialog.wait_visibility()
        dialog.grab_set()
        syntax_focus_widget.focus_set()
        self.root.wait_window(dialog)
        return result["command"]

    def _ordered_value_items_for_object(self, obj: StartupObject, values: Dict[str, str]) -> List[Tuple[str, str]]:
        ordered_keys: List[str] = []
        if obj.linked_file is not None and "FILE" in values:
            ordered_keys.append("FILE")
        for key, _value in obj.details:
            if key in values and key not in ordered_keys:
                ordered_keys.append(key)
        for key, _value in obj.command_details:
            if key in values and key not in ordered_keys:
                ordered_keys.append(key)
        for key, _value in obj.linked_file_details:
            if key in values and key not in ordered_keys:
                ordered_keys.append(key)
        for key in values:
            if key not in ordered_keys:
                ordered_keys.append(key)
        return [(key, str(values.get(key, ""))) for key in ordered_keys]

    def _build_object_command_from_values(self, obj: StartupObject, values: Dict[str, str]) -> Optional[str]:
        kind = obj.kind
        ordered_items = self._ordered_value_items_for_object(obj, values)
        item_map = {key: value for key, value in ordered_items}

        if kind == "macro":
            return "epicsEnvSet({}, {})\n".format(item_map.get("NAME", "").strip(), item_map.get("VALUE", "").strip())
        if kind == "require":
            module_name = item_map.get("MODULE", "").strip()
            version = item_map.get("VERSION", "").strip()
            macro_pairs: List[Tuple[str, str]] = []
            for key, value in ordered_items:
                if key in {"MODULE", "VERSION"}:
                    continue
                if key == "ECMC_VER" and version:
                    continue
                macro_pairs.append((key, value))
            payload = ", ".join(
                "{}={}".format(key, _quote_startup_value(value))
                for key, value in macro_pairs
                if str(key).strip() and str(value).strip()
            )
            version_part = " {}".format(version) if version else ""
            payload_part = ' "{}"'.format(payload) if payload else ""
            return "require {}{}{}\n".format(module_name, version_part, payload_part)
        if kind == "ecmc_command":
            wrapper_name = item_map.get("WRAPPER", "ecmcConfigOrDie").strip() or "ecmcConfigOrDie"
            command_text = _normalize_ecmc_command_text(item_map.get("COMMAND", "").strip())
            return '{} "{}"\n'.format(wrapper_name, command_text.replace('"', '\\"'))
        if kind == "restore_record_update_rate":
            return "${SCRIPTEXEC} ${ecmccfg_DIR}restoreRecordUpdateRate.cmd\n"
        if kind in {"plcvar_analog", "plcvar_binary"}:
            script_name = "ecmcPlcAnalog.db" if kind == "plcvar_analog" else "ecmcPlcBinary.db"
            payload = ", ".join(
                "{}={}".format(key, _quote_startup_value(value))
                for key, value in ordered_items
                if str(value).strip()
            )
            return 'dbLoadRecords("{}", "{}")\n'.format(script_name, payload)

        script_name = self._module_script_name_for_object(obj)
        if not script_name:
            return None
        render_items = [(key, value) for key, value in ordered_items if key != "FILE" or str(value).strip()]
        return _render_startup_command(script_name, render_items)

    def _apply_direct_object_update(self, obj: StartupObject, values: Dict[str, str]) -> bool:
        from tkinter import messagebox

        new_line = self._build_object_command_from_values(obj, values)
        if not new_line:
            messagebox.showinfo("Unsupported update", "Direct editing is not available for this object.")
            return False
        try:
            self._rewrite_object_line(obj, new_line)
        except Exception as exc:
            messagebox.showerror("Update failed", str(exc))
            return False
        self.status_var.set("Updated {} in {}".format(obj.kind, obj.source.name))
        self._refresh_startup_tree()
        return True

    def _apply_object_update(self, obj: StartupObject, values: Dict[str, str], preferred_field: str = "") -> bool:
        from tkinter import messagebox

        new_line = self._prompt_for_object_values(
            obj.kind,
            initial_values=values,
            mode="edit",
            preferred_field=preferred_field,
        )
        if not new_line:
            return False
        try:
            self._rewrite_object_line(obj, new_line)
        except Exception as exc:
            messagebox.showerror("Update failed", str(exc))
            return False
        self.status_var.set("Edited {} in {}".format(obj.kind, obj.source.name))
        self._refresh_startup_tree()
        return True

    def _prompt_for_object_values(
        self,
        kind: str,
        initial_values: Optional[Dict[str, str]] = None,
        mode: str = "add",
        preferred_field: str = "",
    ) -> Optional[str]:
        tk = self.tk
        ttk = self.ttk
        initial_values = initial_values or {}
        action_label = "Edit" if mode == "edit" else "Add"

        if kind == "slave":
            script_file_values = self._candidate_project_files({".cmd", ".script", ".iocsh"})
            subst_values = self._candidate_project_files({".subs", ".substitutions", ".template"})
            hw_desc_values = self._available_slave_hw_descs()
            fields = [
                ("HW_DESC", "HW_DESC", initial_values.get("HW_DESC", hw_desc_values[0] if hw_desc_values else ""), "combo", hw_desc_values),
                ("SLAVE_ID", "SLAVE_ID", initial_values.get("SLAVE_ID", self._next_tree_id("slave", "SLAVE_ID")), "entry", None),
                ("SUBST_FILE", "SUBST_FILE", initial_values.get("SUBST_FILE", ""), "combo", subst_values),
                ("P_SCRIPT", "P_SCRIPT", initial_values.get("P_SCRIPT", ""), "combo", script_file_values),
                ("NELM", "NELM", initial_values.get("NELM", ""), "entry", None),
                ("DEFAULT_SUBS", "DEFAULT_SUBS", initial_values.get("DEFAULT_SUBS", ""), "entry", None),
                ("DEFAULT_SLAVE_PVS", "DEFAULT_SLAVE_PVS", initial_values.get("DEFAULT_SLAVE_PVS", ""), "entry", None),
                ("MACROS", "MACROS", initial_values.get("MACROS", ""), "entry", None),
            ]
            script_name = "addSlave.cmd"
        elif kind == "slave_config":
            slave_obj = self._context_slave_for_selected_entry()
            default_hw_desc = initial_values.get("HW_DESC", "")
            default_slave_id = initial_values.get("SLAVE_ID", "")
            if slave_obj is not None:
                slave_detail_map = dict(slave_obj.details)
                if not default_hw_desc:
                    default_hw_desc = slave_detail_map.get("HW_DESC", "")
                if not default_slave_id and slave_obj.slave_id is not None:
                    default_slave_id = str(slave_obj.slave_id)
            hw_desc_values = self._available_slave_hw_descs()
            config_values = self._available_hardware_config_values(default_hw_desc)
            fields = [
                ("CONFIG", "CONFIG", initial_values.get("CONFIG", ""), "combo", config_values),
                (
                    "LOCAL_CONFIG",
                    "LOCAL_CONFIG",
                    initial_values.get("LOCAL_CONFIG", ""),
                    "combo",
                    self._candidate_project_files({".cmd", ".script", ".iocsh"}),
                ),
                ("SLAVE_ID", "SLAVE_ID", default_slave_id, "combo", self._known_slave_ids()),
                ("HW_DESC", "HW_DESC", default_hw_desc, "combo", hw_desc_values),
                ("EXTRA_MACROS", "Extra macros", "", "entry", None),
            ]
            script_name = "applySlaveConfig.cmd"
        elif kind == "apply_config":
            fields = []
            script_name = "applyConfig.cmd"
        elif kind == "master":
            fields = [
                ("MASTER_ID", "MASTER_ID", initial_values.get("MASTER_ID", "0"), "entry", None),
            ]
            script_name = "addMaster.cmd"
        elif kind == "require":
            macro_items = []
            for key, value in initial_values.items():
                if key in {"MODULE", "VERSION"}:
                    continue
                if key == "ECMC_VER" and initial_values.get("VERSION", "").strip():
                    continue
                if str(value).strip():
                    macro_items.append("{}={}".format(key, _quote_startup_value(str(value).strip())))
            fields = [
                ("MODULE", "MODULE", initial_values.get("MODULE", "ecmccfg"), "entry", None),
                ("VERSION", "VERSION", initial_values.get("VERSION", ""), "entry", None),
                ("MACROS", "MACROS", ", ".join(macro_items), "entry", None),
            ]
            script_name = "require"
        elif kind == "axis":
            yaml_values = self._candidate_project_files({".yaml", ".yml"})
            slave_id_values = self._known_slave_ids()
            axis_id_values = self._known_axis_ids()
            fields = [
                ("FILE", "YAML file", initial_values.get("FILE", "./cfg/axis.yaml"), "combo", yaml_values),
                ("AX_NAME", "AX_NAME", initial_values.get("AX_NAME", ""), "entry", None),
                ("AXIS_ID", "AXIS_ID", initial_values.get("AXIS_ID", ""), "combo", axis_id_values),
                ("DRV_SID", "DRV_SID", initial_values.get("DRV_SID", ""), "combo", slave_id_values),
                ("ENC_SID", "ENC_SID", initial_values.get("ENC_SID", ""), "combo", slave_id_values),
                ("DRV_CH", "DRV_CH", initial_values.get("DRV_CH", ""), "entry", None),
                ("ENC_CH", "ENC_CH", initial_values.get("ENC_CH", ""), "entry", None),
                ("DEV", "DEV", initial_values.get("DEV", ""), "entry", None),
                ("PREFIX", "PREFIX", initial_values.get("PREFIX", ""), "entry", None),
                ("EXTRA_MACROS", "Extra macros", "", "entry", None),
            ]
            script_name = "loadYamlAxis.cmd"
        elif kind == "configured_axis":
            axis_config_values = self._candidate_project_files({".ax", ".pax"})
            fields = [
                ("CONFIG", "CONFIG", initial_values.get("CONFIG", "./cfg/axis.ax"), "combo", axis_config_values),
                ("DEV", "DEV", initial_values.get("DEV", ""), "entry", None),
                ("CLEAR_VARS_CMD", "CLEAR_VARS_CMD", initial_values.get("CLEAR_VARS_CMD", ""), "combo", ["", "ecmc_axis_unset", "empty"]),
                ("CFG_MACROS", "CFG_MACROS", initial_values.get("CFG_MACROS", ""), "entry", None),
            ]
            script_name = "configureAxis.cmd"
        elif kind == "macro":
            fields = [
                ("NAME", "Macro name", initial_values.get("NAME", ""), "entry", None),
                ("VALUE", "Macro value", initial_values.get("VALUE", "${ECMC_EC_SLAVE_NUM}"), "entry", None),
                (
                    "VALUE_PRESET",
                    "Preset",
                    initial_values.get("VALUE_PRESET", ""),
                    "combo",
                    ["", "Slave ID", "Master ID", "PLC ID", "Axis ID", "DataStorage ID"],
                ),
            ]
            script_name = "epicsEnvSet"
        elif kind == "encoder":
            yaml_values = self._candidate_project_files({".yaml", ".yml"})
            fields = [
                ("FILE", "YAML file", initial_values.get("FILE", "./cfg/encoder.yaml"), "combo", yaml_values),
                ("DEV", "DEV", initial_values.get("DEV", ""), "entry", None),
                ("PREFIX", "PREFIX", initial_values.get("PREFIX", ""), "entry", None),
                ("EXTRA_MACROS", "Extra macros", "", "entry", None),
            ]
            script_name = "loadYamlEnc.cmd"
        elif kind == "plc":
            plc_values = self._candidate_project_files({".plc"})
            plc_id_values = self._known_plc_ids()
            fields = [
                ("FILE", "PLC file", initial_values.get("FILE", "./cfg/main.plc"), "combo", plc_values),
                ("PLC_ID", "PLC_ID", initial_values.get("PLC_ID", self._next_tree_id("plc", "PLC_ID")), "combo", plc_id_values),
                ("SAMPLE_RATE_MS", "SAMPLE_RATE_MS", initial_values.get("SAMPLE_RATE_MS", ""), "entry", None),
                ("PLC_MACROS", "PLC_MACROS", initial_values.get("PLC_MACROS", ""), "entry", None),
                ("INC", "INC", initial_values.get("INC", ""), "entry", None),
                ("DESC", "DESC", initial_values.get("DESC", ""), "entry", None),
            ]
            script_name = "loadPLCFile.cmd"
        elif kind == "master_slave_sm":
            fields = [
                ("NAME", "NAME", initial_values.get("NAME", ""), "entry", None),
                ("MST_GRP_NAME", "MST_GRP_NAME", initial_values.get("MST_GRP_NAME", ""), "entry", None),
                ("SLV_GRP_NAME", "SLV_GRP_NAME", initial_values.get("SLV_GRP_NAME", ""), "entry", None),
                ("MST_DISABLE", "MST_DISABLE", initial_values.get("MST_DISABLE", "0"), "combo", ["0", "1"]),
                ("SLV_DISABLE", "SLV_DISABLE", initial_values.get("SLV_DISABLE", "0"), "combo", ["0", "1"]),
                ("SM_ID", "SM_ID", initial_values.get("SM_ID", ""), "entry", None),
            ]
            script_name = "addMasterSlaveSM.cmd"
        elif kind == "subst_config":
            subst_values = self._candidate_project_files({".subs", ".subst", ".substitutions", ".template"})
            fields = [
                ("FILE", "Subst file", initial_values.get("FILE", "./cfg.subs"), "combo", subst_values),
                ("MACROS", "MACROS", initial_values.get("MACROS", ""), "entry", None),
            ]
            script_name = "loadSubstConfig.cmd"
        elif kind == "component":
            return self._prompt_for_component_values(initial_values, mode=mode)
        elif kind == "plugin":
            plugin_values = self._candidate_project_files({".so", ".dylib", ".dll"})
            fields = [
                ("FILE", "Plugin file", initial_values.get("FILE", "./ecmcPlugin.so"), "combo", plugin_values),
                ("PLUGIN_ID", "PLUGIN_ID", initial_values.get("PLUGIN_ID", self._next_tree_id("plugin", "PLUGIN_ID")), "entry", None),
                ("CONFIG", "CONFIG", initial_values.get("CONFIG", ""), "entry", None),
                ("REPORT", "REPORT", initial_values.get("REPORT", ""), "entry", None),
            ]
            script_name = "loadPlugin.cmd"
        elif kind == "datastorage":
            fields = [
                ("DS_SIZE", "DS_SIZE", initial_values.get("DS_SIZE", "1000"), "entry", None),
                ("DS_ID", "DS_ID", initial_values.get("DS_ID", self._next_tree_id("datastorage", "DS_ID")), "entry", None),
                ("DS_TYPE", "DS_TYPE", initial_values.get("DS_TYPE", ""), "entry", None),
                ("SAMPLE_RATE_MS", "SAMPLE_RATE_MS", initial_values.get("SAMPLE_RATE_MS", ""), "entry", None),
                ("DS_DEBUG", "DS_DEBUG", initial_values.get("DS_DEBUG", ""), "entry", None),
                ("DESC", "DESC", initial_values.get("DESC", ""), "entry", None),
            ]
            script_name = "addDataStorage.cmd"
        elif kind == "ecsdo":
            slave_id_values = self._known_slave_ids()
            default_slave_id = ""
            entry = self._selected_startup_entry()
            if entry is not None:
                entry_type, payload = entry
                if entry_type != "file":
                    default_slave_id = str(payload.slave_id or payload.parent_slave_id or "")
            fields = [
                ("SLAVE_ID", "SLAVE_ID", initial_values.get("SLAVE_ID", default_slave_id), "combo", slave_id_values),
                ("INDEX", "INDEX", initial_values.get("INDEX", "0x0000"), "entry", None),
                ("SUBINDEX", "SUBINDEX", initial_values.get("SUBINDEX", "0x00"), "entry", None),
                ("DT", "DT", initial_values.get("DT", "U16"), "combo", ["BIT", "U8", "S8", "U16", "S16", "U32", "S32", "U64", "S64", "F32", "F64"]),
                ("NAME", "NAME", initial_values.get("NAME", ""), "entry", None),
                ("P_SCRIPT", "P_SCRIPT", initial_values.get("P_SCRIPT", ""), "combo", self._candidate_project_files({".cmd", ".script", ".iocsh"})),
            ]
            script_name = "addEcSdoRT.cmd"
        elif kind == "ecdataitem":
            slave_id_values = self._known_slave_ids()
            default_slave_id = ""
            entry = self._selected_startup_entry()
            if entry is not None:
                entry_type, payload = entry
                if entry_type != "file":
                    default_slave_id = str(payload.slave_id or payload.parent_slave_id or "")
            fields = [
                ("STRT_ENTRY_S_ID", "STRT_ENTRY_S_ID", initial_values.get("STRT_ENTRY_S_ID", default_slave_id), "combo", slave_id_values),
                ("STRT_ENTRY_NAME", "STRT_ENTRY_NAME", initial_values.get("STRT_ENTRY_NAME", ""), "entry", None),
                ("OFFSET_BYTE", "OFFSET_BYTE", initial_values.get("OFFSET_BYTE", ""), "entry", None),
                ("OFFSET_BITS", "OFFSET_BITS", initial_values.get("OFFSET_BITS", ""), "entry", None),
                ("RW", "RW", initial_values.get("RW", "2"), "combo", ["0", "1", "2"]),
                ("DT", "DT", initial_values.get("DT", "U16"), "combo", ["BIT", "U8", "S8", "U16", "S16", "U32", "S32", "U64", "S64", "F32", "F64"]),
                ("NAME", "NAME", initial_values.get("NAME", ""), "entry", None),
                ("P_SCRIPT", "P_SCRIPT", initial_values.get("P_SCRIPT", ""), "combo", self._candidate_project_files({".cmd", ".script", ".iocsh"})),
                ("REC_FIELDS", "REC_FIELDS", initial_values.get("REC_FIELDS", ""), "entry", None),
                ("REC_TYPE", "REC_TYPE", initial_values.get("REC_TYPE", ""), "combo", ["ai", "ao", "bi", "bo", "longin", "longout", "mbbi", "mbbo", "stringin", "stringout", "waveform"]),
                ("INIT_VAL", "INIT_VAL", initial_values.get("INIT_VAL", ""), "entry", None),
                ("LOAD_RECS", "LOAD_RECS", initial_values.get("LOAD_RECS", ""), "entry", None),
            ]
            script_name = "addEcDataItem.cmd"
        elif kind == "pvt_controller":
            fields = [
                ("TRG_EC_ENTRY", "TRG_EC_ENTRY", initial_values.get("TRG_EC_ENTRY", ""), "combo", self._ecmc_ec_path_choices("")),
                ("TRG_DUR_S", "TRG_DUR_S", initial_values.get("TRG_DUR_S", "0.1"), "entry", None),
                ("NAXES", "NAXES", initial_values.get("NAXES", ""), "entry", None),
                ("NPOINTS", "NPOINTS", initial_values.get("NPOINTS", ""), "entry", None),
                ("NREADBACK", "NREADBACK", initial_values.get("NREADBACK", ""), "entry", None),
                ("NPULSES", "NPULSES", initial_values.get("NPULSES", ""), "entry", None),
                ("MAX_SIZE", "MAX_SIZE", initial_values.get("MAX_SIZE", ""), "entry", None),
                ("SOFT_TRG_FLNK", "SOFT_TRG_FLNK", initial_values.get("SOFT_TRG_FLNK", ""), "entry", None),
            ]
            script_name = "pvtControllerConfig.cmd"
        elif kind == "lut":
            lut_values = self._candidate_project_files({".lut", ".corr", ".txt"})
            fields = [
                ("FILE", "LUT file", initial_values.get("FILE", "./cfg/lut.txt"), "combo", lut_values),
                ("LUT_ID", "LUT_ID", initial_values.get("LUT_ID", self._next_tree_id("lut", "LUT_ID")), "entry", None),
            ]
            script_name = "loadLUTFile.cmd"
        elif kind in {"plcvar_analog", "plcvar_binary"}:
            plc_id_values = self._known_plc_ids()
            default_plc_id = ""
            entry = self._selected_startup_entry()
            if entry is not None:
                entry_type, payload = entry
                if entry_type != "file":
                    if payload.kind == "plc":
                        for key, value in payload.details:
                            if key == "PLC_ID":
                                default_plc_id = value
                                break
                    elif payload.parent_plc_id is not None:
                        default_plc_id = str(payload.parent_plc_id)
            fields = [
                ("P", "P", initial_values.get("P", "$(IOC):"), "entry", None),
                ("PORT", "PORT", initial_values.get("PORT", "MC_CPU1"), "entry", None),
                (
                    "ASYN_NAME",
                    "ASYN_NAME",
                    initial_values.get("ASYN_NAME", "plcs.plc{}.static.var".format(default_plc_id or "0")),
                    "combo",
                    ["plcs.plc{}.static.var".format(plc_id) for plc_id in plc_id_values] if plc_id_values else [],
                ),
                ("REC_NAME", "REC_NAME", initial_values.get("REC_NAME", "-Var"), "entry", None),
                ("TSE", "TSE", initial_values.get("TSE", "0"), "combo", ["0", "1", "2", "-2"]),
                ("T_SMP_MS", "T_SMP_MS", initial_values.get("T_SMP_MS", "1000"), "entry", None),
                ("EXTRA_MACROS", "Extra macros", "", "entry", None),
            ]
            script_name = "ecmcPlcAnalog.db" if kind == "plcvar_analog" else "ecmcPlcBinary.db"
        elif kind == "ecmc_command":
            return self._prompt_for_ecmc_command_values(initial_values, mode=mode)
        elif kind == "record_update_rate":
            fields = [
                ("RATE_MS", "RATE_MS", initial_values.get("RATE_MS", ""), "entry", None),
            ]
            script_name = "setRecordUpdateRate.cmd"
        elif kind == "restore_record_update_rate":
            fields = []
            script_name = "restoreRecordUpdateRate.cmd"
        else:
            return None

        dialog = tk.Toplevel(self.root)
        dialog.title("{} {}".format(action_label, kind.title()))
        dialog.transient(self.root)
        dialog.withdraw()

        vars_by_name = {}
        first_widget = [None]
        preferred_focus_widget = [None]
        result = {"command": None}
        ecmc_command_hint_var = tk.StringVar(value="")

        for row, (name, label, default, widget_kind, values) in enumerate(fields):
            ttk.Label(dialog, text=label).grid(row=row, column=0, sticky=tk.W, padx=8, pady=4)
            var = tk.StringVar(value=default)
            vars_by_name[name] = var
            if kind in {"slave", "slave_config"} and name == "HW_DESC" and values:
                container, focus_widget = self._create_filtered_value_picker(dialog, var, list(values or []), width=48)
                container.grid(row=row, column=1, sticky=tk.EW, padx=8, pady=4)
                widget = focus_widget
                dialog.rowconfigure(row, weight=1)
            elif widget_kind == "combo":
                if kind == "ecmc_command" and name == "COMMAND" and values:
                    container, focus_widget = self._create_filtered_value_picker(dialog, var, list(values or []), width=48, height=10)
                    container.grid(row=row, column=1, sticky=tk.EW, padx=8, pady=4)
                    widget = focus_widget
                    dialog.rowconfigure(row, weight=1)
                else:
                    widget = ttk.Combobox(dialog, textvariable=var, values=values or [], width=48)
                    if values:
                        self._enable_combobox_filter(
                            widget,
                            list(values or []),
                            auto_post=(kind in {"slave", "slave_config"} and name == "HW_DESC"),
                        )
            else:
                widget = ttk.Entry(dialog, textvariable=var, width=52)
            if not (kind == "ecmc_command" and name == "COMMAND" and values):
                widget.grid(row=row, column=1, sticky=tk.EW, padx=8, pady=4)
            if first_widget[0] is None:
                first_widget[0] = widget
            if preferred_field and name == preferred_field:
                preferred_focus_widget[0] = widget

        next_row = len(fields)
        if kind == "ecmc_command":
            ttk.Label(dialog, text="Hint").grid(row=next_row, column=0, sticky=tk.NW, padx=8, pady=4)
            ttk.Label(dialog, textvariable=ecmc_command_hint_var, justify=tk.LEFT, wraplength=460).grid(
                row=next_row, column=1, sticky=tk.W, padx=8, pady=4
            )
            next_row += 1

            def refresh_ecmc_command_hint(*_args) -> None:
                ecmc_command_hint_var.set(self._ecmc_command_hint(vars_by_name["COMMAND"].get().strip()))

            vars_by_name["COMMAND"].trace_add("write", refresh_ecmc_command_hint)
            refresh_ecmc_command_hint()
        elif kind == "restore_record_update_rate":
            ttk.Label(
                dialog,
                text="Restores record update rate to the original startup setting.",
                justify=tk.LEFT,
                wraplength=460,
            ).grid(row=next_row, column=0, columnspan=2, sticky=tk.W, padx=8, pady=4)
            next_row += 1
        elif kind == "apply_config":
            ttk.Label(
                dialog,
                text="Applies the EtherCAT bus configuration and calculates process image offsets.",
                justify=tk.LEFT,
                wraplength=460,
            ).grid(row=next_row, column=0, columnspan=2, sticky=tk.W, padx=8, pady=4)
            next_row += 1

        if kind == "macro":
            def apply_macro_preset(*_args):
                preset = vars_by_name["VALUE_PRESET"].get().strip()
                preset_map = {
                    "Slave ID": "${ECMC_EC_SLAVE_NUM}",
                    "Master ID": "${ECMC_EC_MASTER_ID}",
                    "PLC ID": "${ECMC_PLC_ID}",
                    "Axis ID": "${AX_ID}",
                    "DataStorage ID": "${DS_ID}",
                }
                if preset in preset_map:
                    vars_by_name["VALUE"].set(preset_map[preset])

            vars_by_name["VALUE_PRESET"].trace_add("write", apply_macro_preset)

        def submit():
            items = []
            extra_macros = ""
            for name, _label, _default, _widget_kind, _values in fields:
                value = vars_by_name[name].get().strip()
                if name == "EXTRA_MACROS":
                    extra_macros = value
                    continue
                if kind == "macro" and name == "VALUE_PRESET":
                    continue
                items.append((name, value))
            items.extend(_parse_extra_macro_items(extra_macros))
            if kind == "macro":
                name = ""
                value = ""
                for key, item_value in items:
                    if key == "NAME":
                        name = item_value
                    elif key == "VALUE":
                        value = item_value
                result["command"] = "epicsEnvSet({}, {})\n".format(name, value)
            elif kind == "require":
                item_map = {key: value for key, value in items}
                module_name = item_map.get("MODULE", "").strip()
                version = item_map.get("VERSION", "").strip()
                macro_pairs, malformed = _parse_macro_payload(item_map.get("MACROS", ""))
                if malformed:
                    from tkinter import messagebox

                    messagebox.showerror(
                        "Invalid macros",
                        "Could not parse: {}".format(", ".join(malformed)),
                        parent=dialog,
                    )
                    return
                if version:
                    macro_pairs = [(key, value) for key, value in macro_pairs if key != "ECMC_VER"]
                payload = ", ".join(
                    "{}={}".format(key, _quote_startup_value(value))
                    for key, value in macro_pairs
                    if str(key).strip()
                )
                version_part = " {}".format(version) if version else ""
                payload_part = ' "{}"'.format(payload) if payload else ""
                result["command"] = "require {}{}{}\n".format(module_name, version_part, payload_part)
            elif kind == "ecmc_command":
                item_map = {key: value for key, value in items}
                wrapper_name = item_map.get("WRAPPER", "ecmcConfigOrDie").strip() or "ecmcConfigOrDie"
                command_text = item_map.get("COMMAND", "").strip()
                if not command_text:
                    from tkinter import messagebox

                    messagebox.showerror("Missing command", "Enter an ECMC command string.", parent=dialog)
                    return
                result["command"] = '{} "{}"\n'.format(wrapper_name, command_text.replace('"', '\\"'))
            elif kind == "restore_record_update_rate":
                result["command"] = '${SCRIPTEXEC} ${ecmccfg_DIR}%s\n' % script_name
            elif kind in {"plcvar_analog", "plcvar_binary"}:
                payload = ", ".join(
                    "{}={}".format(key, _quote_startup_value(value))
                    for key, value in items
                    if str(value).strip()
                )
                result["command"] = 'dbLoadRecords("{}", "{}")\n'.format(script_name, payload)
            else:
                result["command"] = _render_startup_command(script_name, items)
            dialog.destroy()

        def cancel():
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=next_row, column=0, columnspan=2, sticky=tk.E, padx=8, pady=8)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text=action_label, command=submit).pack(side=tk.RIGHT, padx=(0, 6))

        dialog.columnconfigure(1, weight=1)
        dialog.bind("<Escape>", lambda _event: cancel())
        dialog.bind("<Return>", lambda _event: submit())
        dialog.deiconify()
        dialog.wait_visibility()
        dialog.grab_set()
        focus_widget = preferred_focus_widget[0] or first_widget[0]
        if focus_widget is not None:
            focus_widget.focus_set()
            if hasattr(focus_widget, "selection_range"):
                try:
                    focus_widget.selection_range(0, "end")
                except Exception:
                    pass
        self.root.wait_window(dialog)
        return result["command"]

    def _rewrite_object_line(self, obj: StartupObject, new_line: str) -> None:
        resolved_target = obj.source.resolve()
        self._remember_current_buffer()
        if resolved_target in self.file_buffers:
            content = self.file_buffers[resolved_target]
        elif resolved_target.exists():
            content = _read_text(resolved_target)
        else:
            raise IOError("Missing file: {}".format(resolved_target))

        lines = content.splitlines(True)
        index = obj.line - 1
        if index < 0 or index >= len(lines):
            raise IndexError("Line {} out of range in {}".format(obj.line, resolved_target))
        lines[index] = new_line
        new_content = "".join(lines)
        self.file_buffers[resolved_target] = new_content
        resolved_target.write_text(new_content, encoding="utf-8")

        if self.current_edit_path == resolved_target:
            self.editor_text.delete("1.0", "end")
            self.editor_text.insert("1.0", new_content)
            self._highlight_editor_line(obj.line)

    def _default_new_object_file(self, startup_file: Path, kind: str) -> Path:
        startup_dir = startup_file.resolve().parent
        cfg_dir = startup_dir / "cfg"
        target_dir = cfg_dir if cfg_dir.exists() else startup_dir
        filename_map = {
            "axis": "axis.yaml",
            "encoder": "encoder.yaml",
            "plc": "main.plc",
        }
        return target_dir / filename_map[kind]

    def _extract_object_file_target(self, kind: str, command_text: str, startup_file: Path) -> Optional[Path]:
        if kind not in {"axis", "encoder", "plc"}:
            return None
        payload = _extract_script_call_macro_text(command_text.strip())
        pairs, _malformed = _parse_macro_payload(payload)
        file_value = ""
        for key, value in pairs:
            if key.upper() == "FILE":
                file_value = _strip_quotes(value).strip()
                break
        if not file_value:
            return self._default_new_object_file(startup_file, kind)
        if file_value.startswith("./cfg/") and not (startup_file.resolve().parent / "cfg").exists():
            return self._default_new_object_file(startup_file, kind).with_name(Path(file_value).name)
        candidate = Path(file_value)
        if candidate.is_absolute():
            return candidate
        return (startup_file.resolve().parent / candidate).resolve()

    def _starter_file_content(self, kind: str) -> str:
        if kind == "axis":
            return (
                "axis:\n"
                "  id: ${AXIS_ID=1}\n"
                "  name: ${AX_NAME=M1}\n"
            )
        if kind == "encoder":
            return (
                "encoder:\n"
                "  type: incremental\n"
                "  description: Placeholder encoder\n"
            )
        if kind == "plc":
            return (
                "# Starter PLC file\n"
                "VAR\n"
                "END_VAR\n"
            )
        return ""

    def _ensure_object_file_exists(self, kind: str, command_text: str, startup_file: Path) -> Optional[Path]:
        file_target = self._extract_object_file_target(kind, command_text, startup_file)
        if file_target is None:
            return None
        file_target.parent.mkdir(parents=True, exist_ok=True)
        if not file_target.exists():
            file_target.write_text(self._starter_file_content(kind), encoding="utf-8")
        return file_target

    def _is_top_level_object_kind(self, kind: str) -> bool:
        return kind not in {"component", "encoder", "plcvar_analog", "plcvar_binary"}

    def _prepare_insert_block(
        self,
        lines: List[str],
        insert_at: int,
        block_text: str,
        top_level: bool,
    ) -> Tuple[str, int]:
        block = block_text.rstrip("\n") + "\n"
        prefix = ""
        suffix = ""

        if top_level:
            if insert_at > 0 and lines[insert_at - 1].strip():
                prefix = "\n"
            if insert_at < len(lines) and lines[insert_at].strip():
                suffix = "\n"

        return prefix + block + suffix, insert_at + 1 + (1 if prefix else 0)

    def _insert_object_template(self, kind: str, before_selected: bool = False) -> None:
        from tkinter import messagebox

        entry = self._selected_startup_entry()
        if entry is None:
            messagebox.showinfo("No selection", "Select a startup object first.")
            return

        selected_kind = self._selected_object_kind()
        if kind == "component" and selected_kind != "slave":
            messagebox.showinfo("Invalid selection", "Select a slave object before adding a component.")
            return
        if kind == "encoder" and selected_kind != "axis":
            messagebox.showinfo("Invalid selection", "Select an axis object before adding an encoder.")
            return
        if kind in {"plcvar_analog", "plcvar_binary"} and selected_kind != "plc":
            messagebox.showinfo("Invalid selection", "Select a PLC object before adding a PLC variable.")
            return

        entry_type, payload = entry
        if entry_type == "file":
            target_path = payload.path
            target_line = None if not before_selected else 1
        else:
            obj = payload
            target_path = obj.source
            if kind in {"component", "encoder", "plcvar_analog", "plcvar_binary"}:
                target_line = obj.line
                before_selected = False
            else:
                target_line = obj.line

        resolved_target = target_path.resolve()
        self._remember_current_buffer()
        if resolved_target in self.file_buffers:
            content = self.file_buffers[resolved_target]
        elif resolved_target.exists():
            content = _read_text(resolved_target)
        else:
            messagebox.showerror("File missing", "Cannot update missing file:\n{}".format(resolved_target))
            return

        initial_values: Dict[str, str] = {}
        if kind == "slave":
            lines = content.splitlines(True)
            if target_line is None:
                insert_at = len(lines)
            else:
                insert_at = max(0, min(len(lines), target_line - 1 + (0 if before_selected else 1)))
            initial_values["SLAVE_ID"] = self._default_insert_slave_id(resolved_target, content, insert_at + 1)

        insert_text = self._prompt_for_object_values(kind, initial_values=initial_values)
        if not insert_text:
            return

        created_file = self._ensure_object_file_exists(kind, insert_text, resolved_target)

        lines = content.splitlines(True)
        if target_line is None:
            insert_at = len(lines)
        else:
            insert_at = max(0, min(len(lines), target_line - 1 + (0 if before_selected else 1)))
        insert_block, highlight_line = self._prepare_insert_block(
            lines,
            insert_at,
            insert_text,
            self._is_top_level_object_kind(kind),
        )
        lines.insert(insert_at, insert_block)
        new_content = "".join(lines)
        self.file_buffers[resolved_target] = new_content
        resolved_target.write_text(new_content, encoding="utf-8")

        if self.current_edit_path == resolved_target:
            self.editor_text.delete("1.0", "end")
            self.editor_text.insert("1.0", new_content)
            self._highlight_editor_line(highlight_line)

        if created_file is not None:
            self.status_var.set("Inserted {} in {} and created {}".format(kind, resolved_target.name, created_file.name))
        else:
            self.status_var.set("Inserted {} in {}".format(kind, resolved_target.name))
        self._refresh_startup_tree()

    def _collect_subtree_objects(self, item_id: str) -> List[StartupObject]:
        objects = []
        entry = self.startup_item_map.get(item_id)
        if entry is not None and entry[0] == "object":
            objects.append(entry[1])
        for child_id in self.startup_tree.get_children(item_id):
            objects.extend(self._collect_subtree_objects(child_id))
        return objects

    def _selected_object_item(self) -> Optional[Tuple[str, StartupObject]]:
        selected = self.startup_tree.selection()
        if not selected:
            return None
        entry = self.startup_item_map.get(selected[0])
        if entry is None or not isinstance(entry[1], StartupObject):
            return None
        obj = entry[1]
        object_item = selected[0]
        if entry[0] != "object":
            object_item = self.object_tree_items.get(self._object_tree_key(obj), object_item)
        return object_item, obj

    def _selected_object_items(self) -> List[Tuple[str, StartupObject]]:
        selected_items: List[Tuple[str, StartupObject]] = []
        seen_items: Set[str] = set()
        for item_id in self.startup_tree.selection():
            entry = self.startup_item_map.get(item_id)
            if entry is None or not isinstance(entry[1], StartupObject):
                continue
            obj = entry[1]
            object_item = item_id
            if entry[0] != "object":
                object_item = self.object_tree_items.get(self._object_tree_key(obj), object_item)
            if object_item in seen_items:
                continue
            seen_items.add(object_item)
            selected_items.append((object_item, obj))
        return selected_items

    def _can_move_selected_object(self, direction: int) -> bool:
        selected_info = self._selected_object_item()
        if selected_info is None:
            return False
        object_item, _obj = selected_info
        parent_item = self.startup_tree.parent(object_item)
        sibling_items = [
            item_id
            for item_id in self.startup_tree.get_children(parent_item)
            if self.startup_item_map.get(item_id, ("", None))[0] == "object"
        ]
        if object_item not in sibling_items:
            return False
        index = sibling_items.index(object_item)
        neighbor_index = index + direction
        return 0 <= neighbor_index < len(sibling_items)

    def _move_selected_object(self, direction: int) -> None:
        from tkinter import messagebox

        selected_info = self._selected_object_item()
        if selected_info is None:
            messagebox.showinfo("No selection", "Select an object first.")
            return

        object_item, obj = selected_info
        parent_item = self.startup_tree.parent(object_item)
        sibling_items = [
            item_id
            for item_id in self.startup_tree.get_children(parent_item)
            if self.startup_item_map.get(item_id, ("", None))[0] == "object"
        ]
        if object_item not in sibling_items:
            return
        current_index = sibling_items.index(object_item)
        neighbor_index = current_index + direction
        if neighbor_index < 0 or neighbor_index >= len(sibling_items):
            return

        neighbor_item = sibling_items[neighbor_index]
        neighbor_entry = self.startup_item_map.get(neighbor_item)
        if neighbor_entry is None or neighbor_entry[0] != "object":
            return
        neighbor_obj = neighbor_entry[1]

        current_objects = self._collect_subtree_objects(object_item)
        neighbor_objects = self._collect_subtree_objects(neighbor_item)
        current_path = obj.source.resolve()
        neighbor_path = neighbor_obj.source.resolve()
        if current_path != neighbor_path:
            messagebox.showinfo("Move unsupported", "Objects can only be moved within the same file.")
            return

        current_lines = sorted(item.line for item in current_objects)
        neighbor_lines = sorted(item.line for item in neighbor_objects)
        if not current_lines or not neighbor_lines:
            return

        self._remember_current_buffer()
        if current_path in self.file_buffers:
            content = self.file_buffers[current_path]
        elif current_path.exists():
            content = _read_text(current_path)
        else:
            messagebox.showerror("File missing", "Cannot move object in missing file:\n{}".format(current_path))
            return

        lines = content.splitlines(True)
        current_start = min(current_lines) - 1
        current_end = max(current_lines)
        neighbor_start = min(neighbor_lines) - 1
        neighbor_end = max(neighbor_lines)

        if direction < 0:
            move_block = lines[current_start:current_end]
            neighbor_block = lines[neighbor_start:neighbor_end]
            middle_block = lines[neighbor_end:current_start]
            new_lines = lines[:neighbor_start] + move_block + middle_block + neighbor_block + lines[current_end:]
        else:
            move_block = lines[current_start:current_end]
            neighbor_block = lines[neighbor_start:neighbor_end]
            middle_block = lines[current_end:neighbor_start]
            new_lines = lines[:current_start] + neighbor_block + middle_block + move_block + lines[neighbor_end:]

        new_content = "".join(new_lines)
        self.file_buffers[current_path] = new_content
        current_path.write_text(new_content, encoding="utf-8")

        if self.current_edit_path == current_path:
            self.editor_text.delete("1.0", "end")
            self.editor_text.insert("1.0", new_content)

        self.status_var.set("Moved {} {}".format(obj.kind, "up" if direction < 0 else "down"))
        self._refresh_startup_tree()

    def _move_selected_object_up(self) -> None:
        self._move_selected_object(-1)

    def _move_selected_object_down(self) -> None:
        self._move_selected_object(1)

    def _selected_object_subtree_text(self) -> Optional[Tuple[List[StartupObject], str]]:
        from tkinter import messagebox

        selected_info = self._selected_object_items()
        if not selected_info:
            messagebox.showinfo("Invalid selection", "Select one or more object nodes to copy.")
            return None

        all_subtree_objects: List[StartupObject] = []
        seen_keys: Set[Tuple[Path, int, str, str]] = set()
        source_path: Optional[Path] = None
        root_objects: List[StartupObject] = []
        for root_item, root_obj in selected_info:
            root_objects.append(root_obj)
            subtree_objects = self._collect_subtree_objects(root_item)
            if not subtree_objects:
                continue
            current_source_path = root_obj.source.resolve()
            if source_path is None:
                source_path = current_source_path
            elif current_source_path != source_path:
                messagebox.showinfo("Copy unsupported", "Only objects from one startup file can be copied together.")
                return None
            for obj in subtree_objects:
                object_key = self._object_tree_key(obj)
                if object_key in seen_keys:
                    continue
                seen_keys.add(object_key)
                all_subtree_objects.append(obj)

        if source_path is None or not all_subtree_objects:
            return None

        if source_path in self.file_buffers:
            content = self.file_buffers[source_path]
        elif source_path.exists():
            content = _read_text(source_path)
        else:
            messagebox.showerror("File missing", "Cannot copy from missing file:\n{}".format(source_path))
            return None

        lines = content.splitlines(True)
        line_numbers = sorted(obj.line for obj in all_subtree_objects)
        block_lines = [lines[line_no - 1] for line_no in line_numbers if 0 < line_no <= len(lines)]
        root_objects.sort(key=lambda item: item.line)
        return root_objects, "".join(block_lines)

    def _copy_selected_object(self) -> None:
        copied = self._selected_object_subtree_text()
        if copied is None:
            return
        objects, text = copied
        self.copied_object_text = text
        self.copied_object_kind = objects[0].kind if len(objects) == 1 else "objects"
        self.copied_object_top_level = any(self._is_top_level_object_kind(obj.kind) for obj in objects)
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
        except Exception:
            pass
        self.status_var.set(
            "Copied {} {}".format(len(objects), "object" if len(objects) == 1 else "objects")
        )
        self._refresh_context_panel()

    def _on_tree_copy(self, _event=None) -> str:
        self._copy_selected_object()
        return "break"

    def _paste_copied_object(self) -> None:
        from tkinter import messagebox

        if not self.copied_object_text.strip():
            messagebox.showinfo("Nothing copied", "Copy an object first.")
            return

        entry = self._selected_startup_entry()
        if entry is None:
            messagebox.showinfo("No selection", "Select a file or object where the copy should be pasted.")
            return

        entry_type, payload = entry
        if entry_type == "file":
            target_path = payload.path.resolve()
            insert_line = None
        elif entry_type == "object":
            target_path = payload.source.resolve()
            subtree_objects = self._collect_subtree_objects(self.startup_tree.selection()[0])
            insert_line = max(obj.line for obj in subtree_objects) if subtree_objects else payload.line
        else:
            messagebox.showinfo("Invalid selection", "Select a file or object node to paste into.")
            return

        self._remember_current_buffer()
        if target_path in self.file_buffers:
            content = self.file_buffers[target_path]
        elif target_path.exists():
            content = _read_text(target_path)
        else:
            messagebox.showerror("File missing", "Cannot paste into missing file:\n{}".format(target_path))
            return

        lines = content.splitlines(True)
        insert_at = len(lines) if insert_line is None else max(0, min(len(lines), insert_line))
        block_text, highlight_line = self._prepare_insert_block(
            lines,
            insert_at,
            self.copied_object_text,
            self.copied_object_top_level,
        )
        lines.insert(insert_at, block_text)
        new_content = "".join(lines)
        self.file_buffers[target_path] = new_content
        target_path.write_text(new_content, encoding="utf-8")

        if self.current_edit_path == target_path:
            self.editor_text.delete("1.0", "end")
            self.editor_text.insert("1.0", new_content)
            self._highlight_editor_line(highlight_line)

        self.status_var.set("Pasted {} into {}".format(self.copied_object_kind or "object", target_path.name))
        self._refresh_startup_tree()

    def _on_tree_paste(self, _event=None) -> str:
        self._paste_copied_object()
        return "break"

    def _remove_selected_object(self) -> None:
        from tkinter import messagebox

        selected = self.startup_tree.selection()
        if not selected:
            messagebox.showinfo("No selection", "Select one or more objects first.")
            return

        selected_info = self._selected_object_items()
        if not selected_info:
            messagebox.showinfo("Invalid selection", "Select one or more object nodes to remove.")
            return

        subtree_objects: List[StartupObject] = []
        seen_keys: Set[Tuple[Path, int, str, str]] = set()
        for object_item, _obj in selected_info:
            for subtree_obj in self._collect_subtree_objects(object_item):
                object_key = self._object_tree_key(subtree_obj)
                if object_key in seen_keys:
                    continue
                seen_keys.add(object_key)
                subtree_objects.append(subtree_obj)

        lines_by_path = {}
        for obj in subtree_objects:
            lines_by_path.setdefault(obj.source.resolve(), set()).add(obj.line)

        summary = []
        for path, lines in sorted(lines_by_path.items(), key=lambda item: str(item[0])):
            summary.append("{}: line(s) {}".format(self._relative_display(path), ", ".join(str(line) for line in sorted(lines))))

        confirmed = messagebox.askyesno(
            "Remove object",
            "Remove the selected object and its subobjects?\n\n{}".format("\n".join(summary)),
        )
        if not confirmed:
            return

        self._remember_current_buffer()
        affected_current_file = False
        for path, lines_to_remove in lines_by_path.items():
            if path in self.file_buffers:
                content = self.file_buffers[path]
            elif path.exists():
                content = _read_text(path)
            else:
                continue

            original_lines = content.splitlines(True)
            kept_lines = [
                line_text
                for index, line_text in enumerate(original_lines, start=1)
                if index not in lines_to_remove
            ]
            new_content = "".join(kept_lines)
            self.file_buffers[path] = new_content
            path.write_text(new_content, encoding="utf-8")
            if self.current_edit_path == path:
                affected_current_file = True

        if affected_current_file and self.current_edit_path is not None:
            new_content = self.file_buffers[self.current_edit_path]
            self.editor_text.delete("1.0", "end")
            self.editor_text.insert("1.0", new_content)

        self.status_var.set("Removed object subtree")
        self._refresh_startup_tree()

    def _edit_selected_object(self, preferred_field: str = "") -> None:
        from tkinter import messagebox

        obj = self._selected_editable_object()
        if obj is None:
            messagebox.showinfo("Invalid selection", "Select an object node to edit.")
            return
        editable_kinds = self._editable_object_kinds()
        if obj.kind not in editable_kinds:
            messagebox.showinfo("Unsupported object", "Editing is not implemented for {} objects.".format(obj.kind))
            return

        current_values = self._object_detail_map(obj)
        self._apply_object_update(obj, current_values, preferred_field=preferred_field)

    def _edit_selected_parameter(self, event=None) -> Optional[str]:
        from tkinter import simpledialog, messagebox

        if event is not None:
            row_id = self.param_tree.identify_row(event.y)
            if row_id:
                self.param_tree.selection_set(row_id)
                self.param_tree.focus(row_id)

        obj = self._selected_editable_object()
        if obj is None:
            return "break"

        selected = self.param_tree.selection()
        if not selected:
            return "break"

        item_id = selected[0]
        key = self.param_tree.item(item_id, "text")
        values = self.param_tree.item(item_id, "values")
        current_value = values[0] if values else ""

        protected = {"TYPE", "TITLE", "SOURCE"}
        editable_map = self._object_detail_map(obj)
        if key in protected or key not in editable_map:
            messagebox.showinfo("Read-only parameter", "{} is not directly editable here.".format(key))
            return "break"

        new_value = simpledialog.askstring(
            "Edit Parameter",
            "Set {}:".format(key),
            initialvalue=current_value,
            parent=self.root,
        )
        if new_value is None:
            return "break"

        updated_values = dict(editable_map)
        updated_values[key] = new_value
        self._apply_direct_object_update(obj, updated_values)
        return "break"

    def _edit_selected_tree_entry(self, _event=None) -> Optional[str]:
        from tkinter import simpledialog, messagebox

        selected = self.startup_tree.selection()
        if not selected:
            return "break"

        entry = self.startup_item_map.get(selected[0])
        if entry is None:
            return "break"

        entry_type, payload = entry
        if entry_type in {"object", "detail-group", "linked-detail-group"}:
            self._edit_selected_object()
            return "break"

        if entry_type not in {"detail", "linked-file", "linked-detail"}:
            return "break"

        obj = payload
        editable_map = self._object_detail_map(obj)
        item_id = selected[0]
        key = self._tree_text_logical_name(self.startup_tree.item(item_id, "text"))
        current_values = self.startup_tree.item(item_id, "values")
        current_value = current_values[0] if current_values else ""

        if entry_type == "linked-file":
            key = "FILE"
            current_value = editable_map.get("FILE", current_value)

        if obj.kind == "ecmc_command" and key in {"WRAPPER", "COMMAND"}:
            self._edit_selected_object()
            return "break"

        if key not in editable_map:
            messagebox.showinfo("Read-only entry", "{} is not directly editable here.".format(key))
            return "break"

        new_value = simpledialog.askstring(
            "Edit {}".format(key),
            "Set {}:".format(key),
            initialvalue=current_value,
            parent=self.root,
        )
        if new_value is None:
            return "break"

        updated_values = dict(editable_map)
        updated_values[key] = new_value
        self._apply_direct_object_update(obj, updated_values)
        return "break"

    def _on_startup_tree_selected(self, _event=None) -> None:
        selected = self.startup_tree.selection()
        if not selected:
            return
        entry = self.startup_item_map.get(selected[0])
        if entry is None:
            return
        entry_type, payload = entry
        self._set_editor_tree_highlight(selected[0])
        self._populate_param_tree_for_entry(entry_type, payload)
        self._refresh_context_panel(entry)
        if self._suppress_tree_open_on_select:
            return
        self._show_tree_entry_in_editor(entry)

    def _show_validation_results(self, startup_path: Path, result: ValidationResult) -> None:
        self._populate_issues(startup_path, result)
        error_count = sum(1 for issue in result.issues if issue.severity == "error")
        warning_count = sum(1 for issue in result.issues if issue.severity == "warning")
        self.validation_summary_var.set(
            "{}: {} error(s), {} warning(s), {} reference(s)".format(
                startup_path.name,
                error_count,
                warning_count,
                len(result.references),
            )
        )
        self._set_log_panel_visible(True)

    def _show_latest_results(self) -> None:
        if self.log_visible:
            self._hide_log_panel()
            return
        startup_value = self.startup_var.get().strip()
        if self.latest_result is None or not startup_value:
            self.validation_summary_var.set("No validation results yet.")
            self._set_log_panel_visible(True)
            return
        self._show_validation_results(Path(startup_value).expanduser().resolve(), self.latest_result)

    def _populate_issues(self, startup_path: Path, result: ValidationResult) -> None:
        if self.validation_issue_tree is None:
            return

        self.issue_item_map.clear()
        self.validation_issue_tree.delete(*self.validation_issue_tree.get_children(""))

        sorted_issues = sorted(
            result.issues,
            key=lambda issue: (0 if issue.severity == "error" else 1, str(issue.source), issue.line, issue.message),
        )
        if not sorted_issues:
            sorted_issues = [
                ValidationIssue(
                    severity="info",
                    source=startup_path,
                    line=1,
                    message="No validation issues found.",
                )
            ]

        for index, issue in enumerate(sorted_issues):
            location = "{}:{}".format(self._relative_display(issue.source), issue.line)
            item_id = self.validation_issue_tree.insert(
                "",
                "end",
                values=(issue.severity.upper(), location, issue.message),
                tags=(issue.severity,),
            )
            self.issue_item_map[item_id] = issue
            if index == 0:
                self.validation_issue_tree.selection_set(item_id)

    def _on_issue_selected(self, _event=None) -> None:
        if self.validation_issue_tree is None:
            return
        selected = self.validation_issue_tree.selection()
        if not selected:
            return
        issue = self.issue_item_map.get(selected[0])
        if issue is None or issue.severity == "info":
            return
        self._open_file_in_editor(issue.source, line=issue.line)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ecmc Engineering Studio")
    parser.add_argument(
        "--startup",
        default="",
        help="Startup file to open immediately.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    initial_startup = Path(args.startup).expanduser().resolve() if args.startup else None

    try:
        import tkinter as tk
    except Exception as exc:
        print(f"GUI unavailable: {exc}", file=sys.stderr)
        return 2

    root = tk.Tk()
    ValidatorApp(root, initial_startup=initial_startup)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
