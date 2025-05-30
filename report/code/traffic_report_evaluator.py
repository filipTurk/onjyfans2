"""
traffic_report_evaluator.py

A modular script for extracting structured information from traffic reports and comparing human/LLM-generated reports for accuracy.
"""
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
import re

@dataclass
class TrafficEvent:
    event_type: Optional[str] = None
    road: Optional[str] = None
    direction: Optional[str] = None
    location: Optional[str] = None
    cause: Optional[str] = None
    consequence: Optional[str] = None
    time: Optional[str] = None
    delay: Optional[str] = None
    border_crossing: Optional[str] = None
    weather: Optional[str] = None
    other: Optional[str] = None

class TrafficReportExtractor:
    """
    Rule-based extractor for traffic report fields.
    Extend regexes and logic as needed for your data.
    """
    def extract(self, text: str) -> TrafficEvent:
        # Lowercase
        t = text.lower()
        event_type = self._extract_event_type(t)
        road = self._extract_road(t)
        direction = self._extract_direction(t)
        location = self._extract_location(t)
        cause = self._extract_cause(t)
        consequence = self._extract_consequence(t)
        time = self._extract_time(t)
        delay = self._extract_delay(t)
        border_crossing = self._extract_border_crossing(t)
        weather = self._extract_weather(t)
        other = self._extract_other(t)
        return TrafficEvent(event_type, road, direction, location, cause, consequence, time, delay, border_crossing, weather, other)

    def _extract_event_type(self, t: str) -> Optional[str]:
        for key in ["nesreče", "delo na cesti", "zastoji", "opozorila", "mejni prehodi", "vreme", "prireditve", "ovire", "tovorni promet", "zemeljski plaz"]:
            if key in t:
                return key
        return None

    def _extract_road(self, t: str) -> Optional[str]:
        m = re.search(r"(avtocesti|cesti|obvoznici|hitri cesti|regionalni cesti|glavni cesti) ([^.,;\n]+)", t)
        if m:
            return m.group(0)
        return None

    def _extract_direction(self, t: str) -> Optional[str]:
        # Verjetno bo večkrat "proti", ampak od se tudi pojavi
        m = re.search(r"(proti|od) ([^.,;\n]+)", t)
        if m:
            return m.group(2)
        return None

    def _extract_location(self, t: str) -> Optional[str]:
        # Nekateri pisatelji napišejo tudi npr. "avtocesta skozi Maribor" za označbo lokacije
        m = re.search(r"(pri|skozi) ([^.,;\n]+)", t)
        if m:
            return m.group(2)
        return None

    def _extract_cause(self, t: str) -> Optional[str]:
        for key in ["zaradi nesreče", "zaradi del", "zaradi burje", "zaradi poplav", "zaradi praznikov", "zaradi okvare vozila", "zaradi zemeljskega plazu"]:
            if key in t:
                return key
        return None

    def _extract_consequence(self, t: str) -> Optional[str]:
        for key in ["zaprta", "zaprt", "oviran promet", "promet poteka izmenično enosmerno", "zastoj", "omejitev prometa", "prepovedan promet"]:
            if key in t:
                return key
        return None

    def _extract_time(self, t: str) -> Optional[str]:
        # Primerjava samo številk za uro, datum itd.
        numbers = re.findall(r"\d{1,2}(?:[.,:]\d{1,2})?", t)
        if numbers:
            
            return ' '.join(sorted(numbers))
        return None

    def _extract_delay(self, t: str) -> Optional[str]:
        m = re.search(r"zamuda [0-9]+ ?(- ?[0-9]+)? ?minut", t)
        if m:
            return m.group(0)
        return None

    def _extract_border_crossing(self, t: str) -> Optional[str]:
        if "mejni prehod" in t or "čakalna doba" in t:
            m = re.search(r"mejni prehod[\w\s-]*", t)
            if m:
                return m.group(0)
            return "čakalna doba"
        return None

    def _extract_weather(self, t: str) -> Optional[str]:
        for key in ["burja", "sneg", "megla", "poledica", "voda na vozišču", "zimske razmere"]:
            if key in t:
                return key
        return None

    def _extract_other(self, t: str) -> Optional[str]:
        # Catch-all for other info
        m = re.search(r"obvoz[\w\s,.-]*", t)
        if m:
            return m.group(0)
        return None

    def _normalize_field(self, val: Optional[str], field: str = "") -> Optional[str]:
        if val is None:
            return None
        if field == "time":
            
            nums = re.findall(r"\d{1,2}(?:[.,:]\d{1,2})?", val)
            return ' '.join(sorted(nums)) if nums else None
        # Remove common stopwords and words like 'zaprta', 'zaprt', 'oviran', etc.
        val = re.sub(r"\b(zaprta|zaprt|oviran|promet|je|bo|v|na|in|od|proti|za|do|med|pri|ob|po|s|u|se|da|ki|kot|ali|ter|le|pa|so|z|iz|če|čeprav|ampak|tudi|a|ali|ali pa|ali tudi|ali celo|ali morda|ali pa tudi|ali pa celo|ali pa morda|ali pa še|ali pa kar|ali pa le|ali pa samo|ali pa zgolj|ali pa prav|ali pa res|ali pa še kaj|ali pa še kdo|ali pa še kaj drugega|ali pa še kaj podobnega|ali pa še kaj takega|ali pa še kaj takega kot|ali pa še kaj takega kot je|ali pa še kaj takega kot so|ali pa še kaj takega kot so npr|ali pa še kaj takega kot so na primer|ali pa še kaj takega kot so recimo|ali pa še kaj takega kot so recimo npr|ali pa še kaj takega kot so recimo na primer|ali pa še kaj takega kot so recimo na primer npr|ali pa še kaj takega kot so recimo na primer npr.|ali pa še kaj takega kot so recimo na primer npr.\b)", '', val, flags=re.IGNORECASE)
        # Remove whitespace
        val = re.sub(r"[.,;:!?]", '', val)
        val = re.sub(r"\s+", ' ', val).strip()
        return val.lower()

def compare_reports(human: str, ai: str, debug: bool = True) -> Tuple[float, Dict[str, Tuple[Optional[str], Optional[str]]]]:
    """
    Compare two reports using the schema and extractor.
    Returns a score (0-1) and a dict of field-by-field matches.
    """
    extractor = TrafficReportExtractor()
    h_fields = asdict(extractor.extract(human))
    ai_fields = asdict(extractor.extract(ai))
    def norm(v, f):
        return extractor._normalize_field(v, f)
    matches = {}
    relevant = 0
    correct = 0
    for k in h_fields:
        h_val = h_fields[k]
        ai_val = ai_fields[k]
        h_norm = norm(h_val, k)
        ai_norm = norm(ai_val, k)
        # Ni vedno vseh 11 field-ov applicable.
        if h_norm or ai_norm:
            relevant += 1
            # Count as correct if both are not None and equal, or both are None (true negative)
            if h_norm == ai_norm and (h_norm is not None or ai_norm is not None):
                correct += 1
        matches[k] = (h_val, ai_val)
    score = correct / relevant if relevant else 1.0
    if debug:
        print("\n=== Traffic Report Comparison ===")
        print(f"Score: {score:.2f} ({correct}/{relevant} relevant fields matched)")
        print("\n[OK] Matched fields:")
        for k, (hv, aiv) in matches.items():
            if norm(hv, k) == norm(aiv, k) and (norm(hv, k) is not None or norm(aiv, k) is not None):
                print(f"  - {k}: {truncate(hv)}")
        print("\n[MISMATCH] Fields:")
        for k, (hv, aiv) in matches.items():
            if (norm(hv, k) != norm(aiv, k)) and (norm(hv, k) or norm(aiv, k)):
                print(f"  - {k}:")
                print(f"      human: {truncate(hv)}")
                print(f"      ai:    {truncate(aiv)}\n")
        print("\n--- End of Comparison ---\n")
    return score, matches

def truncate(val, maxlen=60):
    if val is None:
        return None
    val = str(val)
    return val if len(val) <= maxlen else val[:maxlen] + '...'

if __name__ == "__main__":

    h = "" #Sem daš human report
    ai = "" #Sem daš AI report
    compare_reports(h, ai)
