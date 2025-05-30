import re
from datetime import datetime
from typing import Optional, Dict, Any, List # Added List
# Assuming traffic_schema.py contains your Pydantic models as discussed previously
from traffic_schema import TrafficEvent, RoadSection, Direction, RoadType, EventType, WeatherEvent, WeatherCondition # Added WeatherEvent, WeatherCondition

class TrafficReportParser:
    def __init__(self):
        self.road_patterns = {
            # Using your RoadType Enum directly as keys
            RoadType.MOTORWAY: r'(?P<type>avtocest[ai]|AC)\s+(?P<name>[\w\s-]+?)(?=\s+proti|\s+v\s+smeri|\s+med|\s+pri|\s+na|\s+zaradi|$)',
            RoadType.EXPRESSWAY: r'(?P<type>hitr[ai]\s*cest[ai]|HC)\s+(?P<name>[\w\s-]+?)(?=\s+proti|\s+v\s+smeri|\s+med|\s+pri|\s+na|\s+zaradi|$)',
            RoadType.MAIN_ROAD: r'(?P<type>glavn[ai]\s*cest[ai])\s+(?P<name>[\w\s-]+?)(?=\s+proti|\s+v\s+smeri|\s+med|\s+pri|\s+na|\s+zaradi|$)',
            RoadType.REGIONAL_ROAD: r'(?P<type>regionaln[ai]\s*cest[ai])\s+(?P<name>[\w\s-]+?)(?=\s+proti|\s+v\s+smeri|\s+med|\s+pri|\s+na|\s+zaradi|$)',
            RoadType.LOCAL_ROAD: r'(?P<type>lokaln[ai]\s*cest[ai]|cest[ai])\s+(?P<name>[\w\s-]+?)(?=\s+proti|\s+v\s+smeri|\s+med|\s+pri|\s+na|\s+zaradi|$)'
        }
        
        # Your comprehensive road_naming_rules, ljubljana_ring, special_cases
        self.road_naming_rules = { # Populated as in your example
            "PRIMORSKA": {"correct_names": ["PRIMORSKA AVTOCESTA"], "type": RoadType.MOTORWAY, "endpoints": {"LJUBLJANA": "KOPER"}, "directions": ["proti Kopru", "proti Ljubljani"], "incorrect_names": ["primorska hitra cesta"]},
            "ŠTAJERSKA": {"correct_names": ["ŠTAJERSKA AVTOCESTA"], "type": RoadType.MOTORWAY, "endpoints": {"LJUBLJANA": "MARIBOR"}, "directions": ["proti Mariboru", "proti Ljubljani"]},
            "DOLENJSKA": {"correct_names": ["DOLENJSKA AVTOCESTA"], "type": RoadType.MOTORWAY, "endpoints": {"LJUBLJANA": "OBREŽJE"}, "directions": ["proti Obrežju", "proti Ljubljani"]},
            "GORENJSKA": {"correct_names": ["GORENJSKA AVTOCESTA"], "type": RoadType.MOTORWAY, "endpoints": {"LJUBLJANA": "KARAVANKE"}, "directions": ["proti Karavankam", "proti Avstriji", "proti Ljubljani"]}, # Rule [1]
            "POMURSKA": {"correct_names": ["POMURSKA AVTOCESTA"], "type": RoadType.MOTORWAY, "endpoints": {"MARIBOR": "LENDAVA"}, "directions": ["proti Mariboru", "proti Lendavi", "proti Madžarski"]}, # Rule [1]
            "PODRAVSKA": {"correct_names": ["PODRAVSKA AVTOCESTA"], "type": RoadType.MOTORWAY, "endpoints": {"MARIBOR": "GRUŠKOVJE"}, "directions": ["proti Mariboru", "proti Gruškovju", "proti Hrvaški"], "forbidden_directions": ["proti Ptuju"]}, # Rule [1]
            "VIPAVSKA": {"correct_names": ["VIPAVSKA HITRA CESTA"], "type": RoadType.EXPRESSWAY, "endpoints": {"NANOS": "VRTOJBA"}, "directions": ["proti Italiji", "proti Vrtojbi", "proti Nanosu", "proti Razdrtemu"], "incorrect_names": ["primorska hitra cesta"]}, # Rule [5]
            "OBALNA": {"correct_names": ["OBALNA HITRA CESTA"], "type": RoadType.EXPRESSWAY, "endpoints": {"SRMIN": "IZOLA"}, "directions": ["proti Kopru", "proti Portorožu"], "incorrect_names": ["primorska hitra cesta"]}, # Rule [5]
            "KOPER-ŠKOFIJE": {"correct_names": ["HITRA CESTA KOPER-ŠKOFIJE"], "type": RoadType.EXPRESSWAY, "endpoints": {"KOPER": "ŠKOFIJE"}, "directions": ["proti Kopru", "proti Škofijam"], "incorrect_names": ["primorska hitra cesta"]}, # Rule [5, 6]
            "DOLGA VAS": {"correct_names": ["HITRA CESTA MEJNI PREHOD DOLGA VAS-DOLGA VAS"], "type": RoadType.EXPRESSWAY, "endpoints": {"MEJNI PREHOD DOLGA VAS": "POMURSKA AVTOCESTA"}, "directions": ["proti pomurski avtocesti", "proti mejnemu prehodu Dolga vas"]}, # Rule [8, 9]
            "GABRK-FERNETIČI": {"correct_names": ["AVTOCESTNI ODSEK RAZCEP GABRK-FERNETIČI"], "type": RoadType.MOTORWAY, "endpoints": {"RAZCEP GABRK": "FERNETIČI"}, "directions": ["proti Italiji", "proti primorski avtocesti", "proti Kopru", "proti Ljubljani"], "incorrect_names": ["primorska avtocesta"]}, # Rule [2]
            "MARIBOR-ŠENTILJ": {"correct_names": ["AVTOCESTNI ODSEK MARIBOR-ŠENTILJ"], "type": RoadType.MOTORWAY, "endpoints": {"DRAGUČOVA": "ŠENTILJ"}, "directions": ["proti Mariboru", "proti Šentilju"], "incorrect_names": ["štajerska avtocesta"]}, # Rule [2]
            "MARIBORSKA VZHODNA OBVOZNICA": {"correct_names": ["MARIBORSKA VZHODNA OBVOZNICA"], "type": RoadType.MOTORWAY, "endpoints": {"RAZCEP SLIVNICA": "RAZCEP DRAGUČOVA"}, "directions": ["proti Avstriji", "proti Lendavi", "proti Ljubljani"], "forbidden_directions": ["proti Mariboru"]}, # Rule [3]
            "REGIONALNA CESTA BETNAVA-PESNICA": {"correct_names": ["REGIONALNA CESTA BETNAVA-PESNICA"], "type": RoadType.REGIONAL_ROAD, "endpoints": {"BETNAVA": "PESNICA"}, "incorrect_names": ["hitra cesta skozi Maribor", "bivša hitra cesta skozi Maribor"]}, # Rule [4]
            "ŠKOFJA LOKA-GORENJA VAS": {"correct_names": ["REGIONALNA CESTA ŠKOFJA LOKA-GORENJA VAS"], "type": RoadType.REGIONAL_ROAD, "endpoints": {"ŠKOFJA LOKA": "GORENJA VAS"}, "directions": ["proti Ljubljani", "proti Gorenji vasi"], "common_name": "škofjeloška obvoznica"}, # Rule [10]
            "LJUBLJANA-ČRNUČE-TRZIN": {"correct_names": ["GLAVNA CESTA LJUBLJANA-ČRNUČE-TRZIN"], "type": RoadType.MAIN_ROAD, "endpoints": {"LJUBLJANA": "TRZIN"}, "directions": ["proti Trzinu", "proti Ljubljani"], "incorrect_names": ["trzinska obvoznica"]}, # Rule [11]
        }
        # Simplified self.ljubljana_ring to merge with road_naming_rules or handle similarly
        self.ljubljana_ring_sections = {
            "VZHODNA OBVOZNICA": {"correct_names": ["LJUBLJANSKA VZHODNA OBVOZNICA"], "type": RoadType.MOTORWAY, "endpoints": {"RAZCEP MALENCE": "RAZCEP ZADOBROVA"}, "directions": ["proti Novemu mestu", "proti Mariboru"]}, # Rule [5]
            "ZAHODNA OBVOZNICA": {"correct_names": ["LJUBLJANSKA ZAHODNA OBVOZNICA"], "type": RoadType.MOTORWAY, "endpoints": {"RAZCEP KOSEZE": "RAZCEP KOZARJE"}, "directions": ["proti Kranju", "proti Kopru"]}, # Rule [5]
            "SEVERNA OBVOZNICA": {"correct_names": ["LJUBLJANSKA SEVERNA OBVOZNICA"], "type": RoadType.MOTORWAY, "endpoints": {"RAZCEP KOSEZE": "RAZCEP ZADOBROVA"}, "directions": ["proti Kranju", "proti Mariboru"]}, # Rule [5]
            "JUŽNA OBVOZNICA": {"correct_names": ["LJUBLJANSKA JUŽNA OBVOZNICA"], "type": RoadType.MOTORWAY, "endpoints": {"RAZCEP KOZARJE": "RAZCEP MALENCE"}, "directions": ["proti Kopru", "proti Novemu mestu"]}, # Rule [5]
        }
        # Merge all road definitions for easier lookup
        self.all_road_rules = {**self.road_naming_rules, **self.ljubljana_ring_sections}


        self.event_patterns = {
            # Use the EventType enum from your Pydantic schema
            EventType.WRONG_WAY_DRIVER: r'voznik.*napačn[oi]\s+smer[i]?',
            EventType.MOTORWAY_CLOSED: r'zaprta\s+(avtocest[ai]|AC)', # More specific than general closure
            EventType.ACCIDENT_WITH_MOTORWAY_JAM: r'(nesreč[ai].*zastoj\s+na\s+avtocesti)|(zastoj\s+na\s+avtocesti.*nesreč[ai])',
            EventType.ROADWORK_WITH_JAM: r'(zastoj[i]?\s+zaradi\s+del)|(del[a]?\s+na\s+cesti.+zastoj)',
            EventType.OTHER_ROAD_CLOSED_DUE_TO_ACCIDENT: r'zaradi\s+nesreče\s+zaprta\s+(glavn[a|i]?|regionaln[a|i]?)\s+cesta',
            EventType.ACCIDENT: r'prometn[ae]\s+nesreč[ai]',
            EventType.BROKEN_DOWN_VEHICLE_LANE_CLOSED: r'(pokvarjen[o|a]?|v\s+okvari)\s+vozil[o|a]?.+zaprt\s+pas', # Rule [15, 53]
            EventType.ANIMAL_ON_ROAD: r'žival.*na\s+(vozišču|cesti)', # Rule [15, 53]
            EventType.OBJECT_ON_ROAD: r'(predmet|razsut\s+tovor)\s+na\s+(vozišču|cesti)', # Rule [15, 53]
            EventType.HIGH_RISK_ROADWORK: r'del[a]?\s+na\s+avtocesti.+(zaprt\s+prometni\s+pas|pred\s+predorom|v\s+predoru)',
            EventType.BORDER_CROSSING_JAM: r'(zastoj|čakalna\s+doba)\s+na\s+mejn[ei|em]\s+prehod[u]?',
            EventType.WEATHER_CONDITION_EVENT: r'(burja|sneg|poledica|megla|močan\s+veter|voda\s+na\s+vozišču)', # Rule [33, 72] for Burja
            EventType.EXCEPTIONAL_TRANSPORT: r'izredni\s+prevoz',
            EventType.TRUCK_RESTRICTION: r'(omejitev|prepoved)\s+prometa\s+tovornih\s+vozil', # Rule [37, 38, 76, 77]
            EventType.GENERAL_ROAD_CLOSURE: r'cest[a]?\s+zaprta', # General, should be last
            EventType.CONGESTION: r'zastoj', # General congestion
            EventType.EVENT_CLEARED_UPDATE: r'(ni\s+več\s+zastoja|promet\s+normalno|odpravljen[a]?|kon[eč|ana])', # For "ODPOVED"
        }
        # ... rest of your __init__ (priority maps, etc.)
        self.priority_map = {
            EventType.WRONG_WAY_DRIVER: 1, EventType.MOTORWAY_CLOSED: 2, EventType.ACCIDENT_WITH_MOTORWAY_JAM: 3,
            EventType.ROADWORK_WITH_JAM: 4, EventType.OTHER_ROAD_CLOSED_DUE_TO_ACCIDENT: 5, EventType.ACCIDENT: 6,
            EventType.BROKEN_DOWN_VEHICLE_LANE_CLOSED: 7, EventType.ANIMAL_ON_ROAD: 8, EventType.OBJECT_ON_ROAD: 9,
            EventType.HIGH_RISK_ROADWORK: 10, EventType.BORDER_CROSSING_JAM: 11,
            EventType.WEATHER_CONDITION_EVENT: 5, # Example priority
            EventType.EXCEPTIONAL_TRANSPORT: 12, EventType.TRUCK_RESTRICTION: 12,
            EventType.GENERAL_ROAD_CLOSURE: 5, EventType.CONGESTION: 7, EventType.EVENT_CLEARED_UPDATE: 15, # Low priority for updates in list
        }


    def _clean_text(self, text: str) -> str:
        # Basic cleaning for common OCR/encoding issues if necessary
        text = text.replace("?a", "č").replace("?e", "š").replace("?i", "ž")
        # Add more replacements if needed
        return text.strip()

    def split_b1_block(self, b1_text: str) -> List[Dict[str, str]]:
        """Splits a B1 block into individual event lines with context."""
        events = []
        current_category = "GENERAL"
        # Remove "B1:" prefix
        if b1_text.startswith("B1:"):
            b1_text = b1_text[3:].strip()

        # Define category headers, map them to EventTypes or reasons if possible
        category_patterns = {
            r"Nesre[čc]e": EventType.ACCIDENT, # Default to accident if under this header
            r"Zastoj[i]?": EventType.CONGESTION,
            r"Ovire": EventType.OBJECT_ON_ROAD, # Or a more general hazard
            r"Delo na cesti": EventType.HIGH_RISK_ROADWORK, # Or general roadwork
            r"Mejni prehodi": EventType.BORDER_CROSSING_JAM,
            r"Opozorila": "WARNING", # Generic category
            r"Tovorni promet": EventType.TRUCK_RESTRICTION,
            r"Prireditve": "EVENT_CLOSURE", # Road closure due to public event
        }

        # Sentences often end with a period, or are separated by newlines followed by uppercase
        # This is a simplified splitter; more sophisticated NLP might be needed for complex cases.
        # Lines could also be separated by the category headers themselves.
        
        # First, split by known category headers to maintain context
        # This is tricky because headers might be inline with the first event
        # For now, we'll assume each "line" extracted from input is a potential event.
        # A more robust way would be to segment text based on sentence structure and keywords.

        raw_lines = re.split(r'\.(?=\s+[A-ZČŠŽ])|\.(?=$)', b1_text) # Split by periods followed by space and uppercase, or end of string
        
        processed_lines = []
        temp_line = ""
        for line_part in raw_lines:
            line_part = line_part.strip()
            if not line_part:
                continue

            # Check if this line part itself is a category header
            is_header = False
            for header_pattern, context_val in category_patterns.items():
                if re.match(header_pattern, line_part, re.IGNORECASE):
                    if temp_line: # Store previous accumulated line
                        processed_lines.append({"text": temp_line, "category_context": current_category})
                        temp_line = ""
                    current_category = context_val
                    # Remove header from line_part if it's *only* a header
                    # This needs refinement, as header might be part of the sentence.
                    # For simplicity, we assume if it matches, the line is the header or starts with it.
                    # and the actual event text follows or is in the next segments.
                    line_part = re.sub(header_pattern, "", line_part, count=1, flags=re.IGNORECASE).strip()
                    is_header = True
                    break
            
            if line_part: # If there's remaining text after header removal or it's not a header
                if temp_line and not temp_line.endswith(('.', '!', '?')):
                     temp_line += " " + line_part
                else:
                    if temp_line: # store completed previous line
                         processed_lines.append({"text": temp_line, "category_context": current_category})
                    temp_line = line_part
        
        if temp_line: # Add any remaining line
            processed_lines.append({"text": temp_line, "category_context": current_category})

        return processed_lines


    def _extract_road_info(self, text: str) -> Optional[Dict[str, Any]]:
        text_upper = text.upper()
        
        # First, try to match specific road names from rules
        for road_key, rules in self.all_road_rules.items():
            for correct_name_variant in rules["correct_names"]:
                if correct_name_variant.upper() in text_upper:
                    # Found a specific, canonical road
                    return {
                        "road_type": rules["type"],
                        "road_name": correct_name_variant, # Use the canonical name
                        "canonical_endpoints": rules.get("endpoints"),
                        "canonical_directions": rules.get("directions"),
                        "forbidden_directions": rules.get("forbidden_directions"),
                        "matched_rule_key": road_key # To access more rules later
                    }
            if "common_name" in rules and rules["common_name"].upper() in text_upper:
                 return {
                        "road_type": rules["type"],
                        "road_name": rules["correct_names"][0], # Use the first canonical name
                        "canonical_endpoints": rules.get("endpoints"),
                        "canonical_directions": rules.get("directions"),
                        "forbidden_directions": rules.get("forbidden_directions"),
                        "matched_rule_key": road_key
                    }


        # If not a specific match, try generic patterns
        for road_type_enum, pattern in self.road_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                road_name_extracted = match.group("name").strip().upper()
                # Try to find if this extracted name corresponds to any known road for canonical form
                for road_key, rules in self.all_road_rules.items():
                    if road_name_extracted in road_key or road_name_extracted in rules["correct_names"][0]:
                         return {
                            "road_type": rules["type"],
                            "road_name": rules["correct_names"][0],
                            "canonical_endpoints": rules.get("endpoints"),
                            "canonical_directions": rules.get("directions"),
                            "forbidden_directions": rules.get("forbidden_directions"),
                            "matched_rule_key": road_key
                        }
                # Generic match, no specific rule found for this exact name
                return {"road_type": road_type_enum, "road_name": road_name_extracted}
        
        # Fallback for "Cesta X - Y"
        road_between_match = re.search(r'(cest[ai])\s+([\w\s-]+?)\s*-\s*([\w\s-]+?)(?=\s+je\s+zaprta|\s+proti|$)', text, re.IGNORECASE)
        if road_between_match:
            from_loc = road_between_match.group(2).strip().upper()
            to_loc = road_between_match.group(3).strip().upper()
            return {
                "road_type": RoadType.REGIONAL_ROAD, # Assumption for "cesta X-Y"
                "road_name": f"CESTA {from_loc}-{to_loc}",
                "from_location_extracted": from_loc,
                "to_location_extracted": to_loc
            }

        return None

    def _extract_direction(self, text: str, road_info: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[str]]:
        direction_data = {"from_location": None, "to_location": None, "section": None}
        text_lower = text.lower()

        # Pattern for "proti X", "v smeri X"
        # Making "proti" more specific to avoid matching "protiPtuju" if Ptuj is part of section.
        to_match = re.search(r'(proti|v\s+smeri)\s+([\w\č\š\ž\-\s]+?)(?=[.,]|$|\s+in|\s+je|\s+zaradi)', text_lower)
        if to_match:
            direction_data["to_location"] = to_match.group(2).strip().upper()
            # Validate against canonical directions if available
            if road_info and road_info.get("forbidden_directions"):
                if f"proti {direction_data['to_location']}".lower() in [fd.lower() for fd in road_info["forbidden_directions"]]:
                    # This direction is forbidden! Handle appropriately (e.g., log warning, clear it)
                    print(f"Warning: Forbidden direction detected for {road_info['road_name']}: proti {direction_data['to_location']}")
                    direction_data["to_location"] = None # Or raise error/mark as invalid

        # Pattern for "med X in Y" / "od X do Y"
        section_match = re.search(r'(med|od)\s+([\w\č\š\ž\-\s]+?)\s+(in|do)\s+([\w\č\š\ž\-\s]+?)(?=[.,]|$|\s+proti|\s+zaradi)', text_lower)
        if section_match:
            loc1 = section_match.group(2).strip().upper()
            loc2 = section_match.group(4).strip().upper()
            direction_data["section"] = f"med {loc1} in {loc2}"
            # If "od X do Y", these can sometimes act as from/to
            if section_match.group(1) == "od":
                direction_data["from_location"] = loc1
                if not direction_data["to_location"]: # Only if "proti" wasn't found or was cleared
                    direction_data["to_location"] = loc2
        
        # Pattern for "iz smeri X proti Y" (for interchanges)
        interchange_match = re.search(r'iz\s+smeri\s+([\w\č\š\ž\-\s]+?)\s+proti\s+([\w\č\š\ž\-\s]+?)(?=[.,]|$|\s+je|\s+zaradi)', text_lower)
        if interchange_match:
            direction_data["from_location"] = interchange_match.group(1).strip().upper()
            direction_data["to_location"] = interchange_match.group(2).strip().upper()

        # If road_info has canonical endpoints and directions are missing
        if road_info and road_info.get("canonical_endpoints"):
            endpoints = road_info["canonical_endpoints"] # Should be {"FROM_KEY": "TO_KEY"} or similar
            # This logic needs to know which direction the text implies or use the first one as default.
            # For now, if specific from/to were parsed from section or interchange, they take precedence.
            # This part requires more nuanced logic based on how 'endpoints' is structured and used.
            # Example: If text says "na Štajerski avtocesti" and a "proti Mariboru" is found, from_location can be LJUBLJANA.
            if direction_data["to_location"] and not direction_data["from_location"]:
                # Try to infer from_location based on to_location and known endpoints
                for ep_from, ep_to in endpoints.items():
                    if direction_data["to_location"] == ep_to.upper():
                        direction_data["from_location"] = ep_from.upper()
                        break
                    elif direction_data["to_location"] == ep_from.upper(): # Reversed
                        direction_data["from_location"] = ep_to.upper()
                        break
        
        if road_info and road_info.get("from_location_extracted"): # From "Cesta X - Y"
            if not direction_data["from_location"]:
                direction_data["from_location"] = road_info["from_location_extracted"]
            if not direction_data["to_location"]:
                 direction_data["to_location"] = road_info["to_location_extracted"]


        return direction_data

    def _detect_event_type(self, text: str, category_context: Any = None) -> Optional[EventType]:
        # Try category context first if it's already an EventType
        if isinstance(category_context, EventType):
            # Further refine based on text if needed, e.g. ACCIDENT vs ACCIDENT_WITH_MOTORWAY_JAM
            if category_context == EventType.ACCIDENT:
                 if re.search(self.event_patterns[EventType.ACCIDENT_WITH_MOTORWAY_JAM], text, re.IGNORECASE):
                     return EventType.ACCIDENT_WITH_MOTORWAY_JAM
            return category_context

        for event_type, pattern in self.event_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return event_type
        return None


    def _extract_reason(self, text: str, event_type: Optional[EventType]) -> str:
        # If event_type implies reason (e.g., ACCIDENT, ROADWORK_JAM)
        if event_type == EventType.ACCIDENT or \
           event_type == EventType.ACCIDENT_WITH_MOTORWAY_JAM or \
           event_type == EventType.OTHER_ROAD_CLOSED_DUE_TO_ACCIDENT:
            return "prometna nesreča"
        if event_type == EventType.ROADWORK_WITH_JAM or \
           event_type == EventType.HIGH_RISK_ROADWORK:
            return "dela na cesti" # Rule [25] "Zastoji zaradi del na avtocesti"
        if event_type == EventType.BROKEN_DOWN_VEHICLE_LANE_CLOSED:
            return "pokvarjeno vozilo"
        if event_type == EventType.WEATHER_CONDITION_EVENT:
            # Extract specific weather condition
            for cond_str, cond_enum in {"burja": WeatherCondition.WIND, "sneg": WeatherCondition.SNOW, 
                                        "poledica": WeatherCondition.ICE, "megla": WeatherCondition.FOG,
                                        "voda na vozišču": WeatherCondition.FLOOD}.items(): # Add more
                if cond_str in text.lower():
                    return cond_str
            return "vremenske razmere"


        match = re.search(r'zaradi\s+([\w\s\č\š\ž-]+?)(?=\s+je|\s+so|\s+na|\s+pri|\s+v|[.,]|$)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "neznan vzrok"

    def _extract_main_consequence(self, text: str, event_type: Optional[EventType]) -> str:
        if event_type == EventType.MOTORWAY_CLOSED or \
           event_type == EventType.OTHER_ROAD_CLOSED_DUE_TO_ACCIDENT or \
           event_type == EventType.GENERAL_ROAD_CLOSURE:
            return "cesta zaprta"
        if event_type == EventType.WRONG_WAY_DRIVER:
            return "nevarnost voznika v napačni smeri"


        # More specific consequences
        if "zaprt vozni pas" in text.lower(): return "zaprt vozni pas"
        if "zaprt prehitevalni pas" in text.lower(): return "zaprt prehitevalni pas"
        if "zaprta polovica avtoceste" in text.lower(): return "zaprta polovica avtoceste" # Rule [27, 65]
        if "promet poteka izmenično enosmerno" in text.lower(): return "promet poteka izmenično enosmerno" # Rule [26, 64]
        if "promet je oviran" in text.lower() or "oviran promet" in text.lower() : return "oviran promet"
        if "zastoj" in text.lower(): return "zastoj"
        
        return "upočasnjen promet"


    def _extract_additional_details(self, text: str, event_type: Optional[EventType]) -> Dict[str, Any]:
        details = {}
        text_lower = text.lower()

        # Jam length
        jam_match = re.search(r'zastoj(?:[\w\s]+?)?\s(?:dolg[a]?\s*)?(\d+[,.]?\d*)\s*k?m', text_lower)
        if jam_match:
            details["jam_length_km"] = float(jam_match.group(1).replace(',', '.'))
            if details["jam_length_km"] < 1.0 and not re.search(r'pričakujemo|daljšal', text_lower): # Rule [22, 60]
                 # If jam is less than 1km and no indication it will worsen, maybe don't report or flag it
                 print(f"Note: Jam length {details['jam_length_km']}km is less than 1km.")
        
        # Delay time
        delay_match = re.search(r'(zamu[dbeoa]+|čas\s+potovanja\s+podaljša\s+za)\s*(?:približno|okoli|do)?\s*([\d\w\sčšž]+?)(?=\s+minut|\s+ure|[.,]|$)', text_lower)
        if delay_match:
            delay_str = delay_match.group(2).strip()
            # Convert words like "pol ure", "četrt ure" to minutes if possible
            if "pol ure" in delay_str: details["delay_minutes"] = 30
            elif "četrt ure" in delay_str: details["delay_minutes"] = 15
            elif "ura" in delay_str or "uri" in delay_str: # e.g. "1 ura", "2 uri"
                num_match = re.search(r'(\d+)', delay_str)
                if num_match: details["delay_minutes"] = int(num_match.group(1)) * 60
            else: # Assuming it's minutes
                num_match = re.search(r'(\d+)', delay_str)
                if num_match: details["delay_minutes"] = int(num_match.group(1))
        
        # Lanes affected / closed
        if "zaprt vozni pas" in text_lower or "zaprt prehitevalni pas" in text_lower:
            details["is_lane_closed"] = True
            # Try to count how many, if specified e.g. "dva vozna pasova zaprta"
            lanes_closed_match = re.search(r'(\d+|en|dva|trije|štirje)\s+(vozn[i|a]?|prehitevaln[i|a]?)\s+pas[ovi|a]?\s+zaprt[i|a]?', text_lower)
            if lanes_closed_match:
                num_str = lanes_closed_match.group(1)
                if num_str.isdigit(): details["lanes_affected"] = int(num_str)
                elif num_str == "en": details["lanes_affected"] = 1
                elif num_str == "dva": details["lanes_affected"] = 2
                # Add more for three, four if needed
            elif details["is_lane_closed"] and "lanes_affected" not in details: # If lane closed but not specified how many
                 details["lanes_affected"] = 1 # Default to 1 if a lane is closed

        if "zaprta polovica avtoceste" in text_lower: # Rule [27, 65]
            details["is_lane_closed"] = True # Implies lanes are closed
            # This situation means traffic is "urejen le po polovici avtoceste v obe smeri" - might need a specific flag or note.


        # Detour
        detour_match = re.search(r'obvoz\s*(?:je)?\s*(.+?)(?=\.\s+[A-ZČŠŽ]|\.$|$)', text, re.IGNORECASE) # Capture until next sentence or end
        if detour_match:
            details["detour_available"] = True
            details["detour_description"] = detour_match.group(1).strip()
            if "SE LAHKO PREUSMERIJO TUDI" in details["detour_description"].upper(): # Rule [30, 68]
                # This indicates an alternative detour option
                pass # The description already contains it.

        # Emergency corridor
        if "prostor za intervencijska vozila" in text_lower: # Rule [71]
            details["leave_emergency_corridor"] = True
        
        # Weather details
        if event_type == EventType.WEATHER_CONDITION_EVENT:
            weather_data = {"condition": None, "weather_comment": text} # Default comment to full text
            if "burja" in text_lower: 
                weather_data["condition"] = WeatherCondition.WIND
                # Extract specific Burja restrictions (stopnja 1 or 2)
                # Rule [33, 72] "Zaradi burje je na vipavski hitri cesti med razcepom Nanos in priključkom Ajdovščina prepovedan promet za počitniške prikolice, hladilnike in vozila s ponjavami, lažja od 8 ton."
                # Rule [34, 73] "Zaradi burje je na vipavski hitri cesti med razcepom Nanos in Ajdovščino prepovedan promet za hladilnike in vsa vozila s ponjavami."
                # This detailed text could be the weather_comment.
            elif "sneg" in text_lower: weather_data["condition"] = WeatherCondition.SNOW
            elif "poledica" in text_lower: weather_data["condition"] = WeatherCondition.ICE
            elif "megla" in text_lower: weather_data["condition"] = WeatherCondition.FOG
            elif "voda na vozišču" in text_lower: weather_data["condition"] = WeatherCondition.FLOOD
            
            if weather_data["condition"]:
                details["weather_details"] = WeatherEvent(**weather_data)

        # Event cleared message required
        if event_type in [EventType.WRONG_WAY_DRIVER, EventType.ACCIDENT_WITH_MOTORWAY_JAM, EventType.ACCIDENT]: # Rule [32, 70]
            details["event_cleared_message_required"] = True
            
        # Active status for updates
        if event_type == EventType.EVENT_CLEARED_UPDATE:
            details["is_event_active"] = False


        return details

    def parse_report_line(self, report_line_text: str, report_id: str, category_context: Any = None) -> Optional[TrafficEvent]:
        cleaned_text = self._clean_text(report_line_text)
        if not cleaned_text:
            return None

        try:
            road_info = self._extract_road_info(cleaned_text)
            # If no road_info, it might be a general announcement (e.g. truck restrictions not tied to a road)
            # or a border crossing waiting time message.

            detected_event_type = self._detect_event_type(cleaned_text, category_context)
            if not detected_event_type and not (road_info and road_info.get("matched_rule_key")): # If no event and no specific road rule matched either
                # This might not be a standard traffic event, or patterns are missing
                # Could be "Tovorni promet: Zaradi praznikov..."
                if "tovornih vozil" in cleaned_text.lower() and ("omejitev" in cleaned_text.lower() or "prepoved" in cleaned_text.lower()):
                    detected_event_type = EventType.TRUCK_RESTRICTION
                elif category_context == EventType.BORDER_CROSSING_JAM or \
                     ("mejn" in cleaned_text.lower() and ("čakalna doba" in cleaned_text.lower() or "zastoj" in cleaned_text.lower())):
                     detected_event_type = EventType.BORDER_CROSSING_JAM
                else:
                    print(f"Skipping line due to unrecognized event type and no road match: {cleaned_text[:100]}")
                    return None


            direction_data = self._extract_direction(cleaned_text, road_info) if road_info else {}
            
            current_road_section = None
            if road_info:
                current_road_section = RoadSection(
                    road_type=road_info["road_type"],
                    road_name=road_info["road_name"],
                    direction=Direction(
                        from_location=direction_data.get("from_location"),
                        to_location=direction_data.get("to_location"),
                        section=direction_data.get("section")
                    )
                )
            elif detected_event_type not in [EventType.TRUCK_RESTRICTION, EventType.BORDER_CROSSING_JAM]: # These might not have a road section
                 print(f"Skipping line as no road information found for event type {detected_event_type}: {cleaned_text[:100]}")
                 return None # Or handle as a general announcement without road_section if schema allows


            reason = self._extract_reason(cleaned_text, detected_event_type)
            consequence = self._extract_main_consequence(cleaned_text, detected_event_type)
            additional_details = self._extract_additional_details(cleaned_text, detected_event_type)

            # Assign priority
            priority = self.priority_map.get(detected_event_type, 99)


            # Construct the event
            event_data = {
                "id": report_id,
                "timestamp": datetime.now(),
                "event_type": detected_event_type,
                "priority": priority,
                "reason": reason,
                "consequence": consequence,
                "source_text_segment": report_line_text, # Store original line
                **additional_details
            }
            if current_road_section:
                event_data["road_section"] = current_road_section
            
            # For events like TRUCK_RESTRICTION, road_section might be optional or very generic (e.g., "Slovenija")
            # The Pydantic model needs to allow road_section to be Optional if such events are to be parsed.
            # For now, we assume events that reach here (past the road_info check) should have a road_section
            # unless it's one of the few exceptions.

            if not current_road_section and detected_event_type not in [EventType.TRUCK_RESTRICTION]:
                # BORDER_CROSSING_JAM might or might not have a specific road, often just the crossing name.
                # If border crossing, the "road_name" could be the border crossing itself.
                if detected_event_type == EventType.BORDER_CROSSING_JAM:
                    # Try to extract border crossing name
                    bc_match = re.search(r'na\s+mejnem\s+prehodu\s+([\w\s\č\š\ž]+?)(?=[.,]|$)', cleaned_text, re.IGNORECASE)
                    if bc_match:
                        event_data["road_section"] = RoadSection(
                            road_type=RoadType.LOCAL_ROAD, # Or a special type for border crossings
                            road_name=f"MEJNI PREHOD {bc_match.group(1).strip().upper()}",
                            direction=Direction(to_location=None) # Direction might be implicit (vstop/izstop)
                        )
                    else: # If no specific border crossing name, this event might be too generic
                        pass # Let it be created without road_section if schema allows or handle error
                # else: # Other event types that should have a road section but don't
                #     return None


            return TrafficEvent(**event_data)

        except Exception as e:
            print(f"Error parsing report line '{report_line_text[:100]}...': {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def process_b1_input(self, full_b1_text: str) -> List[TrafficEvent]:
        """Processes a full B1 input string, splits it, and parses each event line."""
        all_events = []
        event_lines_with_context = self.split_b1_block(full_b1_text)
        
        for i, line_data in enumerate(event_lines_with_context):
            event_text = line_data["text"]
            category_context = line_data["category_context"]
            report_id = f"B1-EVT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{i+1}" # Generate unique ID
            
            parsed_event = self.parse_report_line(event_text, report_id, category_context)
            if parsed_event:
                all_events.append(parsed_event)
        return all_events