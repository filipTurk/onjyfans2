import re
from datetime import datetime
from typing import Optional, Dict, Any
from traffic_schema import TrafficEvent, RoadSection, Direction, RoadType, EventType

class TrafficReportParser:
    def __init__(self):

        self.road_patterns = {
            RoadType.MOTORWAY: r'(avtocest[ai]|AC)\s*([\w\s-]+)',
            RoadType.EXPRESSWAY: r'(hitr[ai]\s*cest[ai]|HC)\s*([\w\s-]+)',
            RoadType.MAIN_ROAD: r'(glavn[ai]\s*cest[ai])\s*([\w\s-]+)',
            RoadType.REGIONAL_ROAD: r'(regionaln[ai]\s*cest[ai])\s*([\w\s-]+)',
            RoadType.LOCAL_ROAD: r'(lokaln[ai]\s*cest[ai])\s*([\w\s-]+)'
        }
        
        self.road_naming_rules = {
            "PRIMORSKA": {
                "correct_names": ["PRIMORSKA AVTOCESTA"],
                "incorrect_names": ["primorska hitra cesta"],
                "directions": ["proti Kopru", "proti Ljubljani"],
                "endpoints": {"from": "LJUBLJANA", "to": "KOPER"},
                "notes": "Standard motorway naming"
            },
            "ŠTAJERSKA": {
                "correct_names": ["ŠTAJERSKA AVTOCESTA"],
                "incorrect_names": [],
                "directions": ["proti Mariboru", "proti Ljubljani"],
                "endpoints": {"from": "LJUBLJANA", "to": "MARIBOR"},
                "notes": "Standard motorway naming"
            },
            "DOLENJSKA": {
                "correct_names": ["DOLENJSKA AVTOCESTA"],
                "incorrect_names": [],
                "directions": ["proti Obrežju", "proti Ljubljani"],
                "endpoints": {"from": "LJUBLJANA", "to": "OBREŽJE"},
                "notes": "Standard motorway naming"
            },
            "GORENJSKA": {
                "correct_names": ["GORENJSKA AVTOCESTA"],
                "incorrect_names": [],
                "directions": ["proti Karavankam", "proti Avstriji", "proti Ljubljani"],
                "endpoints": {"from": "LJUBLJANA", "to": "KARAVANKE"},
                "notes": "Standard motorway naming"
            },
            "POMURSKA": {
                "correct_names": ["POMURSKA AVTOCESTA"],
                "incorrect_names": [],
                "directions": ["proti Lendavi", "proti Madžarski", "proti Mariboru"],
                "endpoints": {"from": "MARIBOR", "to": "LENDAVA"},
                "notes": "Standard motorway naming"
            },
            "PODRAVSKA": {
                "correct_names": ["PODRAVSKA AVTOCESTA"],
                "incorrect_names": [],
                "directions": ["proti Gruškovju", "proti Hrvaški", "proti Mariboru"],
                "endpoints": {"from": "MARIBOR", "to": "GRUŠKOVJE"},
                "notes": "Never use 'proti Ptuju'",
                "forbidden_directions": ["proti Ptuju"]
            },
            
            # Expressways (Hitre ceste)
            "VIPAVSKA": {
                "correct_names": ["VIPAVSKA HITRA CESTA"],
                "incorrect_names": ["primorska hitra cesta"],
                "directions": ["proti Italiji", "proti Vrtojbi", "proti Nanosu", "proti Razdrtemu"],
                "endpoints": {"from": "NANOS", "to": "VRTOJBA"},
                "notes": "Never use 'primorska hitra cesta'"
            },
            "OBALNA": {
                "correct_names": ["OBALNA HITRA CESTA"],
                "incorrect_names": ["primorska hitra cesta"],
                "directions": ["proti Kopru", "proti Portorožu"],
                "endpoints": {"from": "SRMIN", "to": "IZOLA"},
                "notes": "Never use 'primorska hitra cesta'"
            },
            "KOPER-ŠKOFIJE": {
                "correct_names": ["HITRA CESTA KOPER-ŠKOFIJE"],
                "incorrect_names": ["primorska hitra cesta"],
                "directions": ["proti Kopru", "proti Škofijam"],
                "endpoints": {"from": "KOPER", "to": "ŠKOFIJE"},
                "notes": "Name by locations, never 'primorska hitra cesta'"
            },
            "DOLGA VAS": {
                "correct_names": ["HITRA CESTA DOLGA VAS"],
                "incorrect_names": [],
                "directions": ["proti pomurski avtocesti", "proti mejnemu prehodu Dolga vas"],
                "endpoints": {"from": "MEJNI PREHOD", "to": "DOLGA VAS"},
                "notes": "Small section before border crossing"
            },
            
            # Special Sections
            "GABRK-FERNETIČI": {
                "correct_names": ["AVTOCESTNI ODSEK GABRK-FERNETIČI"],
                "incorrect_names": ["primorska avtocesta"],
                "directions": ["proti Italiji", "proti primorski avtocesti", "proti Kopru", "proti Ljubljani"],
                "endpoints": {"from": "GABRK", "to": "FERNETIČI"},
                "notes": "Not primorska avtocesta"
            },
            "MARIBOR-ŠENTILJ": {
                "correct_names": ["AVTOCESTNI ODSEK MARIBOR-ŠENTILJ"],
                "incorrect_names": ["štajerska avtocesta"],
                "directions": ["proti Mariboru", "proti Šentilju"],
                "endpoints": {"from": "MARIBOR", "to": "ŠENTILJ"},
                "notes": "Not štajerska avtocesta"
            },
            "SLIVNICA-DRAGUČOVA": {
                "correct_names": ["MARIBORSKA VZHODNA OBVOZNICA"],
                "incorrect_names": [],
                "directions": ["proti Avstriji", "proti Lendavi", "proti Ljubljani"],
                "endpoints": {"from": "SLIVNICA", "to": "DRAGUČOVA"},
                "notes": "Never use 'proti Mariboru'",
                "forbidden_directions": ["proti Mariboru"]
            },
            
            # Regional and Main Roads
            "ŠKOFJELOŠKA": {
                "correct_names": ["ŠKOFJELOŠKA OBVOZNICA", "REGIONALNA CESTA ŠKOFJA LOKA-GORENJA VAS"],
                "incorrect_names": [],
                "directions": ["proti Ljubljani", "proti Gorenji vasi"],
                "endpoints": {"from": "ŠKOFJA LOKA", "to": "GORENJA VAS"},
                "notes": "Important due to Stén tunnel"
            },
            "TRZINSKA": {
                "correct_names": ["GLAVNA CESTA LJUBLJANA-ČRNUČE-TRZIN"],
                "incorrect_names": ["trzinska obvoznica"],
                "directions": ["proti Trzinu", "proti Ljubljani"],
                "endpoints": {"from": "LJUBLJANA", "to": "TRZIN"},
                "notes": "Use 'glavna cesta', not 'trzinska obvoznica'"
            }
        }
        
        # Ljubljana Ring Road Sections
        self.ljubljana_ring = {
            "VZHODNA": {
                "correct_names": ["LJUBLJANSKA VZHODNA OBVOZNICA"],
                "endpoints": {"from": "MALENCE", "to": "ZADOBROVA"},
                "directions": ["proti Novemu mestu", "proti Mariboru"],
                "notes": "Part of Ljubljana ring road"
            },
            "ZAHODNA": {
                "correct_names": ["LJUBLJANSKA ZAHODNA OBVOZNICA"],
                "endpoints": {"from": "KOSEZE", "to": "KOZARJE"},
                "directions": ["proti Kranju", "proti Kopru"],
                "notes": "Part of Ljubljana ring road"
            },
            "SEVERNA": {
                "correct_names": ["LJUBLJANSKA SEVERNA OBVOZNICA"],
                "endpoints": {"from": "KOSEZE", "to": "ZADOBROVA"},
                "directions": ["proti Mariboru", "proti Kranju"],
                "notes": "Part of Ljubljana ring road"
            },
            "JUŽNA": {
                "correct_names": ["LJUBLJANSKA JUŽNA OBVOZNICA"],
                "endpoints": {"from": "KOZARJE", "to": "MALENCE"},
                "directions": ["proti Kopru", "proti Novemu mestu"],
                "notes": "Part of Ljubljana ring road"
            }
        }
        
        # Special Cases
        self.special_cases = {
            "MARIBOR_BETNAVA-PESNICA": {
                "correct_names": ["REGIONALNA CESTA BETNAVA-PESNICA"],
                "incorrect_names": ["hitra cesta skozi Maribor", "bivša hitra cesta skozi Maribor"],
                "notes": "Former expressway through Maribor, now regional road"
            }
        }

        # Event type patterns
        self.event_patterns = {
            EventType.WRONG_WAY_DRIVER: r'(vozi[l]?\s+v\s+napačn[oi]?\s+smer)|voznik.*napačn[oi]',
            EventType.CLOSED_MOTORWAY: r'zapr[ta]\s+(avtocest[ai]|AC)',
            EventType.ACCIDENT_WITH_JAM: r'(prometn[ai]\s+nesreč[ai].*zastoj)|(zastoj.*prometn[ai]\s+nesreč[ai])',
            EventType.ROADWORK_JAM: r'(zastoj.*del[ao])|(del[ao].*zastoj)',
            EventType.ACCIDENT: r'prometn[ai]\s+nesreč[ai]',
            EventType.BROKEN_VEHICLE: r'(pokvarjen|v\s+okvari).*vozil',
            EventType.ANIMAL: r'žival',
            EventType.OBJECT_ON_ROAD: r'(predmet|razsut\s+tovor)',
            EventType.BORDER_JAM: r'zastoj.*mejn'
        }
#Prosimo voznike, naj se razvrstijo na skrajni levi rob in desni rob vozišča oziroma odstavni pas, v sredini pa pustijo prostor za intervencijska vozila!

        self.event_detail_instructions = {
            EventType.WRONG_WAY_DRIVER: "Opozarjamo vse voznike, ki vozijo v smeri proti vozniku, da je na njihovo polovico avtoceste zašel voznik, ki vozi v napačno smer. Vozite skrajno desno in ne prehitevajte.",
            }

        self.emergency_resolution_message_required = {
            EventType.WRONG_WAY_DRIVER: True,
            EventType.ACCIDENT_WITH_JAM: True,
        }

        self.event_resolution_messages_instruction = {
            EventType.WRONG_WAY_DRIVER: "Promet na pomurski avtocesti iz smeri Dragučove proti Pernici ni več ogrožen zaradi voznika, ki je vozil po napačni polovici avtoceste",
        }


    def _extract_road_info(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract road type and name from text."""
        for road_type, pattern in self.road_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                road_name = match.group(2).strip().upper()
                # Check if it's a standard road
                for std_name, locations in self.standard_roads.items():
                    if std_name in road_name:
                        return {
                            "road_type": road_type,
                            "road_name": f"{std_name} AVTOCESTA",
                            "from_location": locations["from"],
                            "to_location": locations["to"]
                        }
                return {"road_type": road_type, "road_name": road_name}
        return None

    def _extract_direction(self, text: str) -> Optional[Dict[str, str]]:
        """Extract direction information from text."""
        # Look for direction patterns like "proti Ljubljani", "v smeri Maribora"
        direction_pattern = r'(proti|v\s+smeri)\s+(\w+)'
        section_pattern = r'med\s+([\w\s]+)\s+in\s+([\w\s]+)'
        
        direction_match = re.search(direction_pattern, text, re.IGNORECASE)
        section_match = re.search(section_pattern, text, re.IGNORECASE)
        
        result = {}
        if direction_match:
            result["to_location"] = direction_match.group(2).upper()
        if section_match:
            result["section"] = f"med {section_match.group(1)} in {section_match.group(2)}"
        
        return result if result else None

    def _detect_event_type(self, text: str) -> Optional[EventType]:
        """Detect the type of traffic event from text."""
        for event_type, pattern in self.event_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return event_type
        return None

    def _extract_consequences(self, text: str) -> Dict[str, Any]:
        """Extract consequences of the traffic event."""
        consequences = {}
        
        # Check for lane closures
        lane_pattern = r'zaprt[ai]?\s+(\d+)\s+([prometnai\s]+)pas'
        lane_match = re.search(lane_pattern, text, re.IGNORECASE)
        if lane_match:
            consequences["lanes_affected"] = int(lane_match.group(1))
        
        # Check for detours
        detour_pattern = r'[Oo]bvoz\s+([^\.]+)'
        detour_match = re.search(detour_pattern, text)
        if detour_match:
            consequences["detour_available"] = True
            consequences["detour_description"] = detour_match.group(1).strip()
        
        # Check for traffic jam length
        jam_pattern = r'zastoj\s+dolg\s+(\d+(?:,\d+)?)\s*(?:km|kilometer)'
        jam_match = re.search(jam_pattern, text, re.IGNORECASE)
        if jam_match:
            consequences["jam_length"] = float(jam_match.group(1).replace(',', '.'))
        
        return consequences

    def parse_report(self, text: str, report_id: str) -> Optional[TrafficEvent]:
        """Parse a traffic report text and return a TrafficEvent object."""
        try:
            # Extract road information
            road_info = self._extract_road_info(text)
            if not road_info:
                return None
            
            # Extract direction
            direction_info = self._extract_direction(text)
            if not direction_info:
                direction_info = {
                    "from_location": road_info.pop("from_location", ""),
                    "to_location": road_info.pop("to_location", "")
                }
            
            # Create direction object
            direction = Direction(**direction_info)
            
            # Create road section object
            road_section = RoadSection(
                road_type=road_info["road_type"],
                road_name=road_info["road_name"],
                direction=direction
            )
            
            # Detect event type
            event_type = self._detect_event_type(text)
            if not event_type:
                return None
            
            # Extract consequences
            consequences = self._extract_consequences(text)
            
            # Create traffic event
            event = TrafficEvent(
                id=report_id,
                timestamp=datetime.now(),
                event_type=event_type,
                priority=self._get_priority(event_type),
                road_section=road_section,
                reason=self._extract_reason(text),
                consequence=self._extract_main_consequence(text),
                **consequences
            )
            
            return event
            
        except Exception as e:
            print(f"Error parsing report: {str(e)}")
            return None

    def _get_priority(self, event_type: EventType) -> int:
        """Get priority number based on event type."""
        priority_map = {
            EventType.WRONG_WAY_DRIVER: 1,
            EventType.CLOSED_MOTORWAY: 2,
            EventType.ACCIDENT_WITH_JAM: 3,
            EventType.ROADWORK_JAM: 4,
            EventType.CLOSED_OTHER_ROAD: 5,
            EventType.ACCIDENT: 6,
            EventType.BROKEN_VEHICLE: 7,
            EventType.ANIMAL: 8,
            EventType.OBJECT_ON_ROAD: 9,
            EventType.RISKY_ROADWORK: 10,
            EventType.BORDER_JAM: 11
        }
        return priority_map.get(event_type, 99)

    def _extract_reason(self, text: str) -> str:
        """Extract the main reason for the traffic event."""
        reason_patterns = [
            (r'zaradi\s+([\w\s]+?)(?=\s+(?:je|so|na|pri|v))', 1),
            (r'(prometn[ai]\s+nesreč[ai])', 0),
            (r'(del[ao]\s+na\s+cest[ii])', 0)
        ]
        
        for pattern, group in reason_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(group).strip()
        
        return "neznano"

    def _extract_main_consequence(self, text: str) -> str:
        """Extract the main consequence of the traffic event."""
        consequence_patterns = [
            r'(?:je|so)\s+(zaprt[ai][\w\s]+)',
            r'(nastajajo\s+zastoji)',
            r'(oviran\s+promet)',
        ]
        
        for pattern in consequence_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "oviran promet"

# Example usage:
"""
parser = TrafficReportParser()
report = "Na štajerski avtocesti med Domžalami in Krtino proti Mariboru je zaradi prometne nesreče zaprt vozni pas. Nastaja zastoj dolg 2,5 km. Obvoz je možen po regionalni cesti."
event = parser.parse_report(report, "ACC-2024-001")
if event:
    print(event.json(indent=2))
""" 