"""GUI-independent command editing for the engineering workspace.

Read values from source rather than the resolved object model: editing one field
must not freeze macro references in the rest of the command.
"""
import re
import ecmc_engineering_studio as core

# Label, startup script, commonly edited fields. Additional macros are retained
# and can be added through the editor even when not listed here.
OBJECT_TYPES = {
    "require": ("Require", "", "MODULE VERSION MACROS"),
    "master": ("Master", "addMaster.cmd", "MASTER_ID"),
    "slave": ("Slave", "addSlave.cmd", "HW_DESC SLAVE_ID SUBST_FILE P_SCRIPT NELM DEFAULT_SUBS DEFAULT_SLAVE_PVS MACROS"),
    "slave_config": ("Slave config", "applySlaveConfig.cmd", "CONFIG LOCAL_CONFIG SLAVE_ID HW_DESC"),
    "apply_config": ("Apply config", "applyConfig.cmd", ""),
    "component": ("Component", "applyComponent.cmd", "COMP EC_COMP_TYPE COMP_S_ID CH_ID MACROS"),
    "axis": ("Axis", "loadYamlAxis.cmd", "FILE AX_NAME AXIS_ID DRV_SID ENC_SID DRV_CH ENC_CH LIMIT_FWD LIMIT_BWD DEV PREFIX"),
    "configured_axis": ("Configure axis", "configureAxis.cmd", "CONFIG DEV CLEAR_VARS_CMD CFG_MACROS"),
    "encoder": ("Encoder", "loadYamlEnc.cmd", "FILE ENC_SID ENC_CH DEV PREFIX"),
    "macro": ("Macro", "", "NAME VALUE"),
    "plc": ("PLC", "loadPLCFile.cmd", "FILE PLC_ID SAMPLE_RATE_MS PLC_MACROS INC DESC"),
    "plcvar_analog": ("PLC analog variable", "ecmcPlcAnalog.db", "P PORT ASYN_NAME REC_NAME TSE T_SMP_MS"),
    "plcvar_binary": ("PLC binary variable", "ecmcPlcBinary.db", "P PORT ASYN_NAME REC_NAME TSE T_SMP_MS"),
    "plugin": ("Plugin", "loadPlugin.cmd", "FILE PLUGIN_ID CONFIG REPORT"),
    "datastorage": ("Data storage", "addDataStorage.cmd", "DS_SIZE DS_ID DS_TYPE SAMPLE_RATE_MS DS_DEBUG DESC"),
    "ecsdo": ("SDO RT", "addEcSdoRT.cmd", "SLAVE_ID INDEX SUBINDEX DT NAME P_SCRIPT"),
    "ecdataitem": ("EtherCAT data item", "addEcDataItem.cmd", "STRT_ENTRY_S_ID STRT_ENTRY_NAME OFFSET_BYTE OFFSET_BITS RW DT NAME P_SCRIPT REC_FIELDS REC_TYPE INIT_VAL"),
    "master_slave_sm": ("Master/slave state machine", "addMasterSlaveSM.cmd", "NAME MST_GRP_NAME SLV_GRP_NAME MST_DISABLE SLV_DISABLE SM_ID"),
    "subst_config": ("Substitution config", "loadSubstConfig.cmd", "FILE MACROS"),
    "pvt_controller": ("PVT controller", "pvtControllerConfig.cmd", "NAME TRG_EC_ENTRY NPOINTS NREADBACK NPULSES MAX_SIZE SOFT_TRG_FLNK"),
    "lut": ("LUT", "loadLUTFile.cmd", "FILE LUT_ID"),
    "ecmc_command": ("ECMC command", "", "WRAPPER COMMAND"),
    "record_update_rate": ("Record update rate", "setRecordUpdateRate.cmd", "RATE_MS"),
    "restore_record_update_rate": ("Restore record rate", "restoreRecordUpdateRate.cmd", ""),
}
CHILD_TYPES = {"slave": ["component", "slave_config", "ecsdo", "ecdataitem"],
               "axis": ["encoder"], "plc": ["plcvar_analog", "plcvar_binary"]}


def read_values(kind, line):
    line = core._strip_inline_comment(line).strip()
    if line.endswith("\\"):
        raise ValueError("Edit continued commands in the source editor.")
    if kind == "require":
        invocation = core._strip_leading_macro_prefixes(line)
        module, end = core._read_token(invocation, len("require"))
        rest = invocation[end:].lstrip(" ,\t")
        token, end = core._read_token(rest, 0)
        value = core._normalize_value(token)
        version, payload = "", ""
        if re.match(r"[A-Za-z_][A-Za-z0-9_]*\s*=", value):
            payload = value
        else:
            version = value
            payload = core._normalize_value(rest[end:].lstrip(" ,\t"))
        return {"MODULE": core._normalize_value(module), "VERSION": version, "MACROS": payload}
    if kind == "macro":
        pair = core._parse_epics_env_assignment(core._strip_leading_macro_prefixes(line))
        if pair is None:
            raise ValueError("Cannot parse macro assignment; edit it in the source editor.")
        return dict(zip(("NAME", "VALUE"), pair))
    if kind == "ecmc_command":
        pair = core._extract_ecmc_config_invocation(line)
        if pair is None:
            raise ValueError("Cannot parse ECMC command; edit it in the source editor.")
        return dict(zip(("WRAPPER", "COMMAND"), pair))
    if kind.startswith("plcvar_"):
        args = core._split_top_level(line[line.find("(") + 1:line.rfind(")")])
        payload = core._normalize_value(args[1]) if len(args) > 1 else ""
    else:
        payload = core._extract_script_call_macro_text(line)
    pairs, malformed = core._parse_macro_payload(payload)
    if malformed or len({k for k, _ in pairs}) != len(pairs):
        raise ValueError("Ambiguous or malformed macros; correct the command in the source editor first.")
    return dict(pairs)


def render_command(kind, values):
    for key, value in values.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError("Invalid field name: " + key)
        if "\n" in value or "\r" in value:
            raise ValueError("Command fields must be single-line values.")
    if kind == "require":
        name = values.get("MODULE", "").strip()
        version = values.get("VERSION", "").strip()
        if not name or any(c.isspace() or c in "\"'" for c in name + version):
            raise ValueError("Enter a module name and optional version without spaces or quotes.")
        payload = values.get("MACROS", "").strip()
        extras = [(key, value) for key, value in values.items() if key not in {"MODULE", "VERSION", "MACROS"}]
        existing, _ = core._parse_macro_payload(payload)
        duplicates = set(dict(existing)) & {key for key, _ in extras}
        if duplicates:
            raise ValueError("Macros already present in MACROS: " + ", ".join(sorted(duplicates)))
        if extras:
            additional = ", ".join("{}={}".format(key, core._quote_startup_value(value)) for key, value in extras)
            payload = ", ".join(part for part in [payload, additional] if part)
        _, malformed = core._parse_macro_payload(payload)
        if malformed or '"' in payload:
            raise ValueError("Invalid require macros: use KEY=value pairs and single quotes for nested values.")
        return "require " + name + (" " + version if version else "") + (' "' + payload + '"' if payload else "")
    if kind == "macro":
        name = values.get("NAME", "").strip()
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name):
            raise ValueError("Enter an uppercase macro name containing letters, numbers or underscores.")
        value = values.get("VALUE", "").replace("\\", "\\\\").replace('"', '\\"')
        return 'epicsEnvSet("{}", "{}")'.format(name, value)
    if kind == "ecmc_command":
        wrapper = values.get("WRAPPER", "ecmcConfigOrDie")
        if wrapper not in {"ecmcConfigOrDie", "ecmcConfig"} or not values.get("COMMAND", "").strip():
            raise ValueError("Select an ECMC wrapper and enter a command.")
        return '{} "{}"'.format(wrapper, values["COMMAND"].replace('"', '\\"'))
    # Empty values explicitly present in source remain present after editing.
    pairs = [(key, value if value else "''") for key, value in values.items()]
    script = OBJECT_TYPES[kind][1]
    if kind.startswith("plcvar_"):
        payload = ", ".join("{}={}".format(key, core._quote_startup_value(value)) for key, value in pairs)
        return 'dbLoadRecords("{}", "{}")'.format(script, payload)
    return core._render_startup_command(script, pairs).rstrip("\n")


def replace_line(original, command):
    """Retain indentation and trailing comments when replacing a command."""
    code = core._strip_inline_comment(original)
    suffix = original[len(code):].rstrip("\r\n")
    indent = original[:len(original) - len(original.lstrip())]
    if command.startswith("${SCRIPTEXEC}"):
        # Keep the original script-execution prefix, target path and separator.
        for marker in ("${SCRIPTEXEC}", "$(SCRIPTEXEC)", "runScript"):
            offset = code.find(marker)
            if offset >= 0:
                _, target_end = core._read_token(code, offset + len(marker))
                _, new_target_end = core._read_token(command, len("${SCRIPTEXEC}"))
                rest = code[target_end:]
                separator = rest[:len(rest) - len(rest.lstrip(" ,\t"))]
                command = code[:target_end].lstrip() + (separator or " ") + command[new_target_end:].lstrip()
                break
        else:
            raise ValueError("Edit this script invocation in the source editor to preserve its calling syntax.")
    else:
        stripped = core._strip_leading_macro_prefixes(code)
        if stripped:
            prefix_length = len(code) - len(stripped)
            command = code[len(indent):prefix_length] + command
    return indent + command.strip() + suffix + ("\n" if original.endswith("\n") else "")
