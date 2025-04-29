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
    ACCIDENT_WITH_JAM = "accident_with_jam"
    ROADWORK_JAM = "roadwork_jam"
    CLOSED_OTHER_ROAD = "closed_other_road"
    ACCIDENT = "accident"
    BROKEN_VEHICLE = "broken_vehicle"
    ANIMAL = "animal"
    OBJECT_ON_ROAD = "object_on_road"
    RISKY_ROADWORK = "risky_roadwork"
    BORDER_JAM = "border_jam"

class Direction(BaseModel):
    from_location: str
    to_location: str
    section: Optional[str] = None  # specific section if applicable

class RoadSection(BaseModel):
    road_type: RoadType
    road_name: str
    direction: Direction
    
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