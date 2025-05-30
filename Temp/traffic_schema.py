from enum import Enum
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
class RoadType(str, Enum):
    MOTORWAY = "A"  # avtocesta
    EXPRESSWAY = "H"  # hitra cesta
    MAIN_ROAD = "G"  # glavna cesta
    REGIONAL_ROAD = "R"  # regionalna cesta
    LOCAL_ROAD = "L"  # lokalna cesta

class EventType(str, Enum):
    WRONG_WAY_DRIVER = "wrong_way_driver"                             # Voznik v napačno smer [cite: 25, 63]
    MOTORWAY_CLOSED = "motorway_closed"                               # Zaprta avtocesta [cite: 25, 63]
    ACCIDENT_WITH_MOTORWAY_JAM = "accident_with_motorway_jam"         # Nesreča z zastojem na avtocesti [cite: 25, 63]
    ROADWORK_WITH_JAM = "roadwork_with_jam"                           # Zastoji zaradi del na avtocesti [cite: 25, 63]
    OTHER_ROAD_CLOSED_DUE_TO_ACCIDENT = "other_road_closed_due_to_accident" # Zaradi nesreče zaprta glavna ali regionalna cesta [cite: 25, 63]
    ACCIDENT = "accident"                                             # Nesreče na avtocestah in drugih cestah [cite: 25, 63]
    BROKEN_DOWN_VEHICLE_LANE_CLOSED = "broken_down_vehicle_lane_closed" # Pokvarjena vozila, ko je zaprt vsaj en prometni pas [cite: 25, 53, 63]
    ANIMAL_ON_ROAD = "animal_on_road"                                 # Žival, ki je zašla na vozišče [cite: 25, 53, 63]
    OBJECT_ON_ROAD = "object_on_road"                                 # Predmet/razsut tovor na avtocesti [cite: 25, 53, 63]
    HIGH_RISK_ROADWORK = "high_risk_roadwork"                         # Dela na avtocesti, kjer je večja nevarnost naleta [cite: 25, 63]
    BORDER_CROSSING_JAM = "border_crossing_jam"                       # Zastoj pred Karavankami in mejnimi prehodi [cite: 63]
    WEATHER_CONDITION_EVENT = "weather_condition_event"               # Burja, sneg, poledica itd. [cite: 33, 72]

    # Suggested additions:
    GENERAL_ROAD_CLOSURE = "general_road_closure"                 # For road closures not specified by other types
    CONGESTION = "congestion"                                     # Splošni zastoj, ki ni neposredno posledica drugega dogodka (npr. daljši zastoji [cite: 14, 52])
    EXCEPTIONAL_TRANSPORT = "exceptional_transport"               # Izredni prevoz
    TRUCK_RESTRICTION = "truck_restriction"                       # Omejitev prometa tovornih vozil [cite: 37, 38, 76, 77]
    EVENT_CLEARED_UPDATE = "event_cleared_update"                 # For "ODPOVED" messages or event resolutions [cite: 31, 32, 70]
    OTHER_HAZARD = "other_hazard"                                 # For other unclassified hazards

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
    id: str = Field(..., description="Unique identifier for the event, e.g., ACC-2024-001")
    timestamp: datetime = Field(..., description="Timestamp of when the event was reported or occurred")
    event_type: EventType
    priority: int = Field(..., description="Priority based on hierarchy in rules (1-11+). Lower is more urgent.")
    road_section: RoadSection

    reason: str = Field(..., description="Primary cause of the event (e.g., 'prometna nesreča', 'dela na cesti')")
    consequence: str = Field(..., description="Effect on traffic (e.g., 'zaprt vozni pas', 'zastoj', 'promet preusmerjen')")

    lanes_affected: Optional[int] = Field(None, description="Number of lanes affected")
    is_lane_closed: Optional[bool] = Field(None, description="Indicates if at least one traffic lane is closed. From rule [cite: 15, 53]")

    detour_available: Optional[bool] = Field(False, description="Is a detour available?")
    detour_description: Optional[str] = Field(None, description="Description of the detour, e.g., 'Obvoz je po vzporedni regionalni cesti' [cite: 30, 68]")

    emergency_services_on_site: Optional[bool] = Field(None, description="Are emergency services (police, ambulance, fire) on site?")
    leave_emergency_corridor: Optional[bool] = Field(None, description="Instruction to leave emergency corridor. [cite: 71]")

    estimated_duration_minutes: Optional[int] = Field(None, description="Estimated duration of the event in minutes")
    event_cleared_message_required: Optional[bool] = Field(None, description="Is a specific 'all clear' message required for this event type? [cite: 31, 32, 70]")

    # For traffic jams/congestion
    jam_length_km: Optional[float] = Field(None, description="Length of the traffic jam in kilometers. Report if >= 1km unless special circumstances. [cite: 22, 60]")
    delay_minutes: Optional[int] = Field(None, description="Estimated delay time in minutes")

    # Specific details for weather events
    weather_details: Optional[WeatherEvent] = Field(None, description="Specific details if event_type is WEATHER")

    # Optional field for event updates or cancellations referring to the original event
    references_event_id: Optional[str] = Field(None, description="If this event is an update or cancellation, ID of the original event")
    is_event_active: bool = Field(True, description="Indicates if the event is currently active. False if it's a cancellation/cleared message for a previous event.")

    # Additional context
    source_text_segment: Optional[str] = Field(None, description="The raw text segment from which this event was parsed, for traceability")

    class Config:
        schema_extra = {
            "example": {
                "id": "ACC-2024-001",
                "timestamp": "2024-03-20T10:30:00",
                "event_type": "ACCIDENT_WITH_JAM_ON_MOTORWAY",
                "priority": 3,
                "road_section": {
                    "road_type": "A",
                    "road_name": "ŠTAJERSKA AVTOCESTA",
                    "direction": {
                        "from_location": "LJUBLJANA", # Or a specific junction like "Razcep Zadobrova"
                        "to_location": "MARIBOR",     # Final destination as per rules [cite: 12, 50]
                        "section": "med priključkoma Domžale in Krtina" # Wider section for tunnels/rest areas [cite: 29, 67]
                    }
                },
                "reason": "prometna nesreča",
                "consequence": "zaprt vozni pas, zastoj",
                "lanes_affected": 1,
                "is_lane_closed": True,
                "detour_available": True,
                "detour_description": "Obvoz je po vzporedni regionalni cesti. [cite: 30, 68]",
                "emergency_services_on_site": True,
                "leave_emergency_corridor": True,
                "estimated_duration_minutes": 60,
                "event_cleared_message_required": True,
                "jam_length_km": 2.5,
                "delay_minutes": 15,
                "is_event_active": True
            }
        }