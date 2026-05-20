import re
from typing import Dict, List

from langchain_community.document_loaders import PyPDFLoader


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2022", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _to_lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


def _canon(line: str) -> str:
    out = re.sub(r"\s+", " ", line.lower()).strip()
    out = out.replace(" #", "#")
    return out


def _looks_like_label_only(line: str, label: str) -> bool:
    c_line = _canon(line)
    c_label = _canon(label)
    return c_line in {c_label, f"{c_label}:", f"{c_label} :"}


def _inline_value(line: str, label: str) -> str:
    c_line = _canon(line)
    c_label = _canon(label)
    if c_line.startswith(f"{c_label}:") or c_line.startswith(f"{c_label} :"):
        raw = line.split(":", 1)[1].strip() if ":" in line else ""
        return raw
    return ""


def _is_generic_label_line(line: str) -> bool:
    labels = {
        "dispatcher",
        "phone",
        "phone#",
        "phone #",
        "fax",
        "fax#",
        "fax #",
        "email",
        "load",
        "date",
        "trailer",
        "notes",
        "carrier name",
        "total amount ( cad )",
        "total charges",
        "hst/gst",
        "terms & conditions",
        "time",
        "order",
        "quantity",
        "weight",
        "temp",
        "stop #1 (pick)",
        "stop #2 (drop)",
    }
    c_line = _canon(line).rstrip(":").strip()
    return c_line in labels


def _next_value_after_label(lines: List[str], label_variants: List[str]) -> str:
    return _next_value_after_label_with_validator(
        lines=lines,
        label_variants=label_variants,
        validator=None
    )


def _next_value_after_label_with_validator(
    lines: List[str],
    label_variants: List[str],
    validator
) -> str:
    for idx, line in enumerate(lines):
        for label in label_variants:
            if _looks_like_label_only(line, label):
                for nxt in lines[idx + 1: idx + 10]:
                    if _is_generic_label_line(nxt):
                        continue
                    if nxt in {"--", "-", ":"}:
                        continue
                    if validator and not validator(nxt):
                        continue
                    return nxt
            inline = _inline_value(line, label)
            if inline and not _is_generic_label_line(inline):
                if validator and not validator(inline):
                    continue
                return inline
    return ""


def _is_date(value: str) -> bool:
    return bool(re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", value))


def _is_time(value: str) -> bool:
    return bool(re.search(r"\b\d{1,2}:\d{2}\s*(AM|PM)?\b", value, flags=re.IGNORECASE))


def _is_phone(value: str) -> bool:
    return bool(re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", value))


def _is_email(value: str) -> bool:
    return bool(re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", value, flags=re.IGNORECASE))


def _is_number(value: str) -> bool:
    return bool(re.fullmatch(r"\d+(\.\d+)?", value.strip()))


def _is_currency(value: str) -> bool:
    return bool(re.search(r"\$\s*\d+(\.\d+)?", value))


def _slice_block(
    lines: List[str],
    start_labels: List[str],
    end_labels: List[str]
) -> List[str]:
    start_idx = -1
    end_idx = len(lines)

    for idx, line in enumerate(lines):
        if any(_canon(line).startswith(_canon(lbl)) for lbl in start_labels):
            start_idx = idx
            break

    if start_idx == -1:
        return []

    for idx in range(start_idx + 1, len(lines)):
        if any(_canon(lines[idx]).startswith(_canon(lbl)) for lbl in end_labels):
            end_idx = idx
            break

    return lines[start_idx:end_idx]


def _extract_first_phone(block_text: str) -> str:
    match = re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", block_text)
    return match.group(0) if match else ""


def _extract_all_phones(block_text: str) -> List[str]:
    return re.findall(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", block_text)


def _extract_first_email(block_text: str) -> str:
    match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", block_text, flags=re.IGNORECASE)
    return match.group(0) if match else ""


def _parse_stop_block(stop_number: str, stop_type: str, stop_body: str) -> Dict[str, str]:
    lines = _to_lines(stop_body)
    cutoff = len(lines)
    for idx, line in enumerate(lines):
        c_line = _canon(line)
        if (
            "please send pod" in c_line
            or "temperature controlled" in c_line
            or "bill of lading" in c_line
        ):
            cutoff = idx
            break
    lines = lines[:cutoff]
    info: Dict[str, str] = {
        "stop_number": stop_number,
        "stop_type": stop_type.lower(),
    }

    label_map = {
        "date": (["Date"], _is_date),
        "time": (["Time"], _is_time),
        "phone": (["Phone"], _is_phone),
        "email": (["Email"], _is_email),
        "order": (["Order"], None),
        "quantity": (["Quantity"], _is_number),
        "weight": (["Weight"], _is_number),
        "temperature": (["Temp"], None),
        "notes": (["Notes"], None),
    }

    for out_key, config in label_map.items():
        labels, validator = config
        value = _next_value_after_label_with_validator(
            lines=lines,
            label_variants=labels,
            validator=validator
        )
        if value:
            info[out_key] = value

    field_start = len(lines)
    for idx, line in enumerate(lines):
        if _canon(line).rstrip(":").strip() in {
            "date", "time", "phone", "email", "order", "quantity", "weight", "temp", "notes"
        }:
            field_start = idx
            break

    location_lines = lines[:field_start]
    if location_lines:
        info["location_name"] = location_lines[0]
    if len(location_lines) > 1:
        info["location_address"] = ", ".join(location_lines[1:])

    numeric_values = [ln for ln in lines if _is_number(ln)]
    if numeric_values:
        large_numeric = [
            val for val in numeric_values
            if "." not in val and int(val) >= 100000
        ]
        small_integers = [
            val for val in numeric_values
            if "." not in val and int(val) <= 1000
        ]
        decimal_values = [val for val in numeric_values if "." in val]

        if large_numeric and "order_reference" not in info:
            info["order_reference"] = large_numeric[0]

        if small_integers:
            info["quantity"] = small_integers[-1]

        if decimal_values:
            info["weight"] = decimal_values[-1]

    return info


def _extract_original_layout(context: str) -> Dict[str, object]:
    text = _normalize_text(context)
    lines = _to_lines(text)

    dispatcher_block = _slice_block(
        lines,
        start_labels=["Dispatcher"],
        end_labels=["Load", "Trailer", "Carrier Name"]
    )
    dispatcher_text = "\n".join(dispatcher_block)
    dispatcher_phones = _extract_all_phones(dispatcher_text)

    carrier_block = _slice_block(
        lines,
        start_labels=["Carrier Name"],
        end_labels=["Total Amount", "Terms & Conditions", "Stop #1"]
    )
    carrier_text = "\n".join(carrier_block)
    carrier_phones = _extract_all_phones(carrier_text)

    total_amount = _next_value_after_label_with_validator(
        lines,
        ["Total Amount ( CAD )", "Total Amount CAD"],
        _is_currency
    )
    total_charges = _next_value_after_label_with_validator(
        lines,
        ["Total Charges"],
        _is_currency
    )
    tax_value = _next_value_after_label(lines, ["HST/GST", "GST/HST"])

    result: Dict[str, object] = {
        "document_type": "carrier_confirmation",
        "load_number": _next_value_after_label_with_validator(
            lines, ["Load"], _is_number
        ),
        "confirmation_date": _next_value_after_label_with_validator(
            lines, ["Date"], _is_date
        ),
        "trailer_type": _next_value_after_label(lines, ["Trailer"]),
        "dispatcher_phone": dispatcher_phones[0] if len(dispatcher_phones) > 0 else "",
        "dispatcher_fax": dispatcher_phones[1] if len(dispatcher_phones) > 1 else "",
        "dispatcher_email": _extract_first_email(dispatcher_text),
        "carrier_name": _next_value_after_label(lines, ["Carrier Name"]),
        "carrier_phone": carrier_phones[0] if len(carrier_phones) > 0 else "",
        "total_amount_cad": total_amount,
        "total_charges": total_charges,
        "tax": tax_value,
    }

    if not result["confirmation_date"] or "signature" in result["confirmation_date"].lower():
        result["confirmation_date"] = ""
        all_dates = re.findall(r"\b\d{1,2}/\d{1,2}/\d{4}\b", text)
        if all_dates:
            result["confirmation_date"] = all_dates[0]

    if (
        not result["total_charges"]
        or _is_generic_label_line(result["total_charges"])
    ):
        result["total_charges"] = result["total_amount_cad"]

    stop_pattern = re.compile(
        r"Stop\s*#(?P<num>\d+)\s*\((?P<type>Pick|Drop)\)(?P<body>.*?)(?=Stop\s*#\d+\s*\(|$)",
        flags=re.IGNORECASE | re.DOTALL
    )
    stops: List[Dict[str, str]] = []
    for match in stop_pattern.finditer(text):
        stop_num = match.group("num").strip()
        stop_type = match.group("type").strip().lower()
        stop_body = match.group("body").strip()
        stops.append(_parse_stop_block(stop_num, stop_type, stop_body))

    result["stops"] = stops

    special_notes = []
    for line in text.split("\n"):
        clean_line = line.strip()
        if not clean_line:
            continue
        lowered = clean_line.lower()
        if (
            "please send pod" in lowered
            or "temperature controlled" in lowered
            or "bill of lading" in lowered
            or "trip number" in lowered
        ):
            special_notes.append(clean_line)

    if special_notes:
        result["special_notes"] = special_notes

    return result


def _extract_scrambled_layout(context: str) -> Dict[str, object]:
    text = _normalize_text(context)
    lines = _to_lines(text)

    def _is_generic_label_line(line):
        return _canon(line).rstrip(":").strip() in {
            "date", "time", "phone", "email", "order", "quantity", "weight", "temp", "notes", "qty", "weight", "phone no", "contact", "datetime"
        }

    def get_next_non_empty_line(lines, start_idx, max_offset=5):
        for i in range(1, max_offset + 1):
            if start_idx + i < len(lines):
                line = lines[start_idx + i].strip()
                if line and line != ":":
                    if _is_generic_label_line(line) or line in {"Pickup#", "Delivery#"}:
                        return ""
                    return line
        return ""

    carrier_name = ""
    load_number = ""
    total_amount = ""
    confirmation_date = ""
    dispatcher_email = ""
    dispatcher_phone = ""
    dispatcher_fax = ""
    carrier_phone = ""
    trailer_type = ""

    for idx, ln in enumerate(lines):
        c_ln = _canon(ln)
        if c_ln in {"carrier", "carrier name"}:
            carrier_name = get_next_non_empty_line(lines, idx)
        elif c_ln in {"trip #", "load", "load #", "carrier confirmation no"}:
            load_number = get_next_non_empty_line(lines, idx)
        elif c_ln in {"settled amount", "total amount ( cad )", "total charges"}:
            total_amount = get_next_non_empty_line(lines, idx)
        elif c_ln in {"trip created", "date", "confirmation date"}:
            confirmation_date = get_next_non_empty_line(lines, idx)
        elif c_ln == "trailer":
            trailer_type = get_next_non_empty_line(lines, idx)
        elif c_ln == "phone #":
            # If we don't have carrier phone yet, assign it
            val = get_next_non_empty_line(lines, idx)
            if val and not carrier_phone:
                carrier_phone = val

    # Emails
    emails = re.findall(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, flags=re.IGNORECASE)
    if emails:
        dispatcher_email = next((e for e in emails if "skverma" not in e), emails[0])

    # Phones
    all_phones = _extract_all_phones(text)
    if all_phones:
        dispatcher_phone = all_phones[0]
        if len(all_phones) > 1:
            dispatcher_fax = all_phones[1]

    # Parse stops
    stops = []
    stop_count = 0
    for idx, ln in enumerate(lines):
        c_ln = _canon(ln)
        if c_ln in {"pickup", "delivery"}:
            stop_count += 1
            stop_type = "pick" if c_ln == "pickup" else "drop"
            
            stop_lines = []
            for i in range(idx, len(lines)):
                line = lines[i]
                c_line = _canon(line)
                if i > idx and (c_line in {"pickup", "delivery", "broker name"} or "stop #" in c_line):
                    break
                stop_lines.append(line)
                
            info = {
                "stop_number": str(stop_count),
                "stop_type": stop_type,
            }
            if len(stop_lines) > 1:
                info["location_name"] = stop_lines[1]
                
            for s_idx, s_ln in enumerate(stop_lines):
                c_s_ln = _canon(s_ln)
                if c_s_ln == "datetime":
                    dt_val = get_next_non_empty_line(stop_lines, s_idx)
                    if dt_val:
                        parts = dt_val.split(" ", 1)
                        info["date"] = parts[0]
                        if len(parts) > 1:
                            info["time"] = parts[1]
                elif c_s_ln == "qty":
                    qty = get_next_non_empty_line(stop_lines, s_idx)
                    if qty: info["quantity"] = qty
                elif c_s_ln == "weight":
                    wt = get_next_non_empty_line(stop_lines, s_idx)
                    if wt: info["weight"] = wt
                elif c_s_ln == "phone no":
                    ph = get_next_non_empty_line(stop_lines, s_idx)
                    if ph: info["phone"] = ph
                elif c_s_ln == "contact":
                    ct = get_next_non_empty_line(stop_lines, s_idx)
                    if ct: info["contact"] = ct
                elif c_s_ln in {"temperature", "reefer temp"}:
                    tmp = get_next_non_empty_line(stop_lines, s_idx)
                    if tmp: info["temperature"] = tmp
                    
            # Extract address: it is the line after DateTime value
            for s_idx, s_ln in enumerate(stop_lines):
                if _canon(s_ln) == "datetime":
                    offset = 1
                    while s_idx + offset < len(stop_lines):
                        val = stop_lines[s_idx + offset].strip()
                        if val and val != ":" and val != info.get("date", "") and not val.startswith(info.get("date", "")):
                            if not _is_generic_label_line(val) and val != "Pickup#" and val != "Delivery#":
                                info["location_address"] = val
                            break
                        offset += 1
            stops.append(info)

    special_notes = []
    for line in text.split("\n"):
        clean_line = line.strip()
        if not clean_line:
            continue
        lowered = clean_line.lower()
        if (
            "please send pod" in lowered
            or "temperature controlled" in lowered
            or "bill of lading" in lowered
            or "trip number" in lowered
        ):
            special_notes.append(clean_line)

    result = {
        "document_type": "carrier_confirmation",
        "load_number": load_number,
        "confirmation_date": confirmation_date,
        "trailer_type": trailer_type,
        "dispatcher_phone": dispatcher_phone,
        "dispatcher_fax": dispatcher_fax,
        "dispatcher_email": dispatcher_email,
        "carrier_name": carrier_name,
        "carrier_phone": carrier_phone,
        "total_amount_cad": total_amount,
        "total_charges": total_amount,
        "tax": "N/A",
        "stops": stops,
    }

    if special_notes:
        result["special_notes"] = special_notes

    return result


def extract_carrier_confirmation_json_from_text(context: str) -> Dict[str, object]:
    res = _extract_original_layout(context)
    # If the original parser failed to find carrier_name or did not extract any stops,
    # fall back to the robust scrambled layout parser
    if not res.get("carrier_name") or not res.get("stops"):
        scrambled_res = _extract_scrambled_layout(context)
        if scrambled_res.get("carrier_name") or scrambled_res.get("stops"):
            return scrambled_res
    return res


def extract_carrier_confirmation_json_from_pdf(file_path: str) -> Dict[str, object]:
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text = "\n".join(page.page_content for page in pages if page.page_content)
    return extract_carrier_confirmation_json_from_text(text)
