from enum import Enum
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

class RoadType(str, Enum):
    MOTORWAY = "A"  # avtocesta
    EXPRESSWAY = "H"  # hitra cesta
    MAIN_ROAD = "G"  # glavna cesta
    REGIONAL_ROAD = "R"  # regionalna cesta
    LOCAL_ROAD = "L"  # lokalna cesta

class EventType(str, Enum):
    WRONG_WAY_DRIVER = "wrong_way_driver"
    CLOSED_MOTORWAY = "closed_motorway"
    ACCIDENT_WITH_JAM_ON_MOTORWAY = "accident_with_jam" #short jams can cause long jams
    ROADWORK_JAM = "roadwork_jam"
    CLOSED_OTHER_ROAD = "closed_other_road"
    ACCIDENT = "accident"
    BROKEN_VEHICLE = "broken_vehicle" # one pas is blocked
    ANIMAL = "animal"
    OBJECT_ON_ROAD = "object_on_road"
    RISKY_ROADWORK = "risky_roadwork" #kjer je večja nevarnost naleta (zaprt prometni pas, pred predori, v predorih, …)
    BORDER_JAM = "border_jam" #Zastoj pred Karavankami in mejnimi prehodi
    WEATHER = "weather" #sneženje, dež, led, megla, močan veter, plazovi, poplave, …

class CommonEvents(str, Enum):
    MORNING_RUSH_HOUR = "morning_rush_hour"  # common traffic jams in the morning
    EVENING_RUSH_HOUR = "evening_rush_hour"  # common traffic jams in the evening
    HOLIDAY_TRAFFIC = "holiday_traffic"  # common traffic jams during holidays


class Direction(BaseModel):
    from_location: str
    to_location: str
    section: Optional[str] = None  # specific section if applicable

class RoadSection(BaseModel):
    road_type: RoadType
    road_name: str
    direction: Direction

class WeatherCondition(str, Enum):
    SNOW = "snow"
    RAIN = "rain"
    FOG = "fog"
    ICE = "ice"
    FLOOD = "flood"
    SLIPPERY_ROAD = "slippery_road"  # general slippery conditions
    WIND = "wind"  # strong winds

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    
class WeatherEvent(BaseModel):
    condition: WeatherCondition
    severity: Optional[Severity] = None
    weather_comment: Optional[str] = None  # additional comments or details                             #not okay prolly 

class WeatherCommentsExampleList:
    SNOW = [
        "Cesta je zasnežena in spolzka.",
        "Zaradi snega priporočamo zimsko opremo.",
        "Sneženje otežuje vidljivost in vožnjo."
    ]
    
    RAIN = [
        "Močan dež zmanjšuje vidljivost.",
        "Vozna površina je mokra in spolzka.",
        "Poplavljena vozišča – priporočamo obvoz."
    ]

    FOG = [
        "Gosta megla zmanjšuje vidljivost.",
        "Uporabite meglenke in zmanjšajte hitrost.",
        "Vidljivost je manjša od 50 metrov."
    ]

    ICE = [
        "Pojavlja se poledica na cestišču.",
        "Mostovi in senčne lege so poledenele.",
        "Zaradi ledu priporočamo skrajno previdnost."
    ]

    FLOOD = [
        "Vozišča so ponekod zalita z vodo.",
        "Poplavne vode ovirajo promet.",
        "Cesta zaprta zaradi poplav."
    ]

    SLIPPERY_ROAD = [
        "Cesta je spolzka zaradi vremenskih razmer.",
        "Zaradi spolzkega vozišča zmanjšajte hitrost.",
        "Opozorilo: močno zmanjšan oprijem ceste."
    ]

    WIND = [
        "Močni sunki vetra ovirajo promet.",
        "Zaradi burje je prepovedan promet za vozila s ponjavami.",
        "Na izpostavljenih mestih možni močni sunki vetra.",
        "Zaradi burje je na vipavski hitri cesti med razcepom Nanos in priključkom Ajdovščina prepovedan promet za počitniške prikolice, hladilnike in vozila s ponjavami, lažja od 8 ton.",
        "Zaradi burje je na vipavski hitri cesti med razcepom Nanos in Ajdovščino prepovedan promet za hladilnike in vsa vozila s ponjavami.",
        "Na vipavski hitri cesti in na regionalni cesti Ajdovščina - Podnanos ni več prepovedi prometa zaradi burje."
    ]


class TrafficEvent(BaseModel):
    """Main schema for traffic events following RTVSlo rules"""
    id: str
    timestamp: datetime
    event_type: EventType
    priority: int  # 1-10 based on hierarchy in rules
    road_section: RoadSection
    
    # Event details
    reason: str  # what caused the event
    consequence: str  # effect on traffic
    
    # Additional information
    lanes_affected: Optional[int] = None
    detour_available: bool = False
    detour_description: Optional[str] = None
    
    # For accidents and incidents
    emergency_response: Optional[bool] = None
    estimated_duration: Optional[int] = None  # in minutes
    emergency_resolution_message_required: Optional[bool] = None
    
    # For traffic jams
    jam_length: Optional[float] = None  # in kilometers
    delay_time: Optional[int] = None  # in minutes
    
    class Config:
        schema_extra = {
            "example": {
                "id": "ACC-2024-001",
                "timestamp": "2024-03-20T10:30:00",
                "event_type": "ACCIDENT_WITH_JAM",
                "priority": 3,
                "road_section": {
                    "road_type": "A",
                    "road_name": "ŠTAJERSKA AVTOCESTA",
                    "direction": {
                        "from_location": "LJUBLJANA",
                        "to_location": "MARIBOR",
                        "section": "med priključkoma Domžale in Krtina"
                    }
                },
                "reason": "prometna nesreča",
                "consequence": "zaprt vozni pas",
                "lanes_affected": 1,
                "detour_available": True,
                "detour_description": "Obvoz je po vzporedni regionalni cesti",
                "emergency_response": True,
                "estimated_duration": 60,
                "jam_length": 2.5,
                "delay_time": 15
            }
        } 