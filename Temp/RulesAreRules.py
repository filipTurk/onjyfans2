road_references = {
    "PRIMORSKA_AVTOCESTA": {
        "proper_name": "primorska avtocesta",
        "directions": ["proti Kopru", "proti Ljubljani"]
    },
    "DOLENJSKA_AVTOCESTA": {
        "proper_name": "dolenjska avtocesta",
        "directions": ["proti Obrežju", "proti Ljubljani"]
    },
    "GORENJSKA_AVTOCESTA": {
        "proper_name": "gorenjska avtocesta",
        "directions": ["proti Karavankam", "proti Avstriji", "proti Ljubljani"]
    },
    "ŠTAJERSKA_AVTOCESTA": {
        "proper_name": "štajerska avtocesta",
        "directions": ["proti Mariboru", "proti Ljubljani"]
    },
    "POMURSKA_AVTOCESTA": {
        "proper_name": "pomurska avtocesta",
        "directions": ["proti Lendavi", "proti Madžarski", "proti Mariboru"]
    },
    "PODRAVSKA_AVTOCESTA": {
        "proper_name": "podravska avtocesta",
        "directions": ["proti Gruškovju", "proti Hrvaški", "proti Mariboru"]
    },
    "GABRK_FERNETIČI": {
        "proper_name": "avtocestni odsek Gaberk–Fernetici",
        "directions": ["proti Italiji", "proti primorski avtocesti", "proti Kopru", "proti Ljubljani"]
    },
    "MARIBOR_ŠENTILJ": {
        "proper_name": "avtocestni odsek Maribor–Šentilj",
        "directions": ["proti Šentilju", "proti Mariboru"]
    },
    "MARIBORSKA_VZHODNA_OBVOZNICA": {
        "proper_name": "mariborska vzhodna obvoznica",
        "directions": ["proti Avstriji", "proti Lendavi", "proti Ljubljani"]
    },
    "VIPAVSKA_HITRA_CESTA": {
        "proper_name": "vipavska hitra cesta",
        "directions": ["proti Italiji", "proti Vrtojbi", "proti Nanosu", "proti Razdrtemu"]
    },
    "OBALNA_HITRA_CESTA": {
        "proper_name": "obalna hitra cesta",
        "directions": ["proti Kopru", "proti Portorožu"]
    },
    "KOPER_ŠKOFIJE_HITRA_CESTA": {
        "proper_name": "hitra cesta Koper–Škofije",
        "directions": ["proti Škofijam", "proti Kopru"]
    },
    "DOLGA_VAS_HITRA_CESTA": {
        "proper_name": "hitra cesta Dolga vas",
        "directions": ["proti mejnemu prehodu Dolga vas", "proti pomurski avtocesti"]
    },
    "ŠKOFJA_LOKA_GORENJA_VAS": {
        "proper_name": "regionalna cesta Škofja Loka–Gorenja vas",
        "directions": ["proti Gorenji vasi", "proti Ljubljani"]
    },
    "LJUBLJANA_TRZIN": {
        "proper_name": "glavna cesta Ljubljana–Trzin",
        "directions": ["proti Trzinu", "proti Ljubljani"]
    },
    "LJUBLJANSKA_OBVOZNICA_VZHODNA": {
        "proper_name": "vzhodna ljubljanska obvoznica",
        "directions": ["proti Novemu mestu", "proti Mariboru"]
    },
    "LJUBLJANSKA_OBVOZNICA_ZAHODNA": {
        "proper_name": "zahodna ljubljanska obvoznica",
        "directions": ["proti Kranju", "proti Kopru"]
    },
    "LJUBLJANSKA_OBVOZNICA_SEVERNA": {
        "proper_name": "severna ljubljanska obvoznica",
        "directions": ["proti Kranju", "proti Mariboru"]
    },
    "LJUBLJANSKA_OBVOZNICA_JUŽNA": {
        "proper_name": "južna ljubljanska obvoznica",
        "directions": ["proti Kopru", "proti Novemu mestu"]
    }
}

event_priority = [
    "voznik_v_napacni_smeri",
    "zaprta_avtocesta",
    "nesreca_z_zastojem_na_avtocesti",
    # Include all other events in priority order
]

announcement_templates = {
    "voznik_v_napacni_smeri": "Opozarjamo vse voznike, ki vozijo po {road} od {location_from} proti {location_to}, torej v smeri proti {direction}, da je na njihovo polovico avtoceste zašel voznik, ki vozi v napačno smer. Vozite skrajno desno in ne prehitevajte.",
    "burja_stopnja_1": "Zaradi burje je na vipavski hitri cesti med razcepom Nanos in priključkom Ajdovščina prepovedan promet za počitniške prikolice, hladilnike in vozila s ponjavami, lažja od 8 ton.",
    # Add all other templates
}

announcement_templates_pomembno = {
    
}

def create_prompt(traffic_data):
    prompt = """
    Ustvari prometno informacijo za slovensko radijsko oddajo po naslednjih pravilih:
    
    1. Uporabi pravilna imena cest:
       - Primorska avtocesta je proti Kopru ali proti Ljubljani
       - Ljubljanska obvoznica ima štiri krake: vzhodno, zahodno, severno in južno
       
    2. Sestava prometne informacije:
       - Formulacija 1: Cesta in smer + razlog + posledica in odsek
       - Formulacija 2: Razlog + cesta in smer + posledica in odsek
       
    3. Prioriteta dogodkov:
       - Najprej navedi voznike v napačni smeri
       - Nato zaprte avtoceste
       - Nato nesreče z zastoji
       
    4. Za zastoje:
       - Objavi samo zastoje daljše od 1 kilometra
       - Ne objavljaj pričakovanih prometnih konic
    
    Podatki o prometu:
    Datum: {date}
    Vreme: {weather}
    Nesreče: {accidents}
    Zastoji: {traffic_jams}
    Dela na cesti: {roadwork}
    
    Generiraj besedilo v slovenščini kot za prometno poročilo na radiu.
    """
    
    # Fill in the prompt with actual data
    filled_prompt = prompt.format(
        date=traffic_data.get("Datum", ""),
        weather=traffic_data.get("ContentVremeSLO", ""),
        accidents=traffic_data.get("ContentNesreceSLO", ""),
        traffic_jams=traffic_data.get("ContentZastojiSLO", ""),
        roadwork=traffic_data.get("ContentDeloNaCestiSLO", "")
    )
    
    return filled_prompt

def prioritize_events(events):
    """Sort events by priority level"""
    return sorted(events, key=lambda x: event_priority.index(x["type"]) if x["type"] in event_priority else len(event_priority))



def validate_broadcast_text(text):
    """Validate that the generated text follows all guidelines"""
    issues = []
    
    # Check for incorrect road naming
    if "hitra cesta primorska" in text.lower():
        issues.append("Incorrect naming: should be 'vipavska hitra cesta' not 'primorska hitra cesta'")
    
    # Check for improper direction references
    if "proti Ptuju" in text and "podravska avtocesta" in text.lower():
        issues.append("Incorrect direction: should not use 'proti Ptuju' for podravska avtocesta")
    
    # Add more validation rules
    
    return issues

# Example training data that demonstrates correct usage
training_examples = [
    {
        "input": "Nesreča na primorski avtocesti med Vrhniko in Logatcem, zastoj 2km",
        "output": "Na primorski avtocesti med priključkoma Vrhnika in Logatec je zaradi prometne nesreče nastal zastoj dolg dva kilometra v smeri proti Kopru."
    },
    {
        "input": "Burja stopnja 1 na hitri cesti Nanos-Vrtojba",
        "output": "Zaradi burje je na vipavski hitri cesti med razcepom Nanos in priključkom Ajdovščina prepovedan promet za počitniške prikolice, hladilnike in vozila s ponjavami, lažja od 8 ton."
    },
    # Add more examples that follow the rules
]

def test_road_naming(model):
    """Test that the model uses correct road naming"""
    test_cases = [
        {
            "input": "Zastoj na hitri cesti Nanos-Vrtojba",
            "expected_phrases": ["vipavska hitra cesta", "proti Italiji"],
            "forbidden_phrases": ["primorska hitra cesta"]
        },
        # Add more test cases
    ]
    
    for case in test_cases:
        output = model.generate(case["input"])
        
        for phrase in case["expected_phrases"]:
            assert phrase in output.lower(), f"Missing required phrase: {phrase}"
            
        for phrase in case["forbidden_phrases"]:
            assert phrase not in output.lower(), f"Found forbidden phrase: {phrase}"

"""
def process_traffic_data(csv_data):
    # Load and parse CSV data
    df = pd.read_csv(csv_data)
    
    for _, row in df.iterrows():
        # Extract data
        traffic_info = extract_information(row)
        
        # Prioritize events
        events = parse_events(traffic_info)
        prioritized_events = prioritize_events(events)
        
        # Generate prompt with rules
        prompt = create_prompt(traffic_info)
        
        # Generate text with LLM
        generated_text = call_llm_model(prompt)
        
        # Validate output
        issues = validate_broadcast_text(generated_text)
        if issues:
            # Either fix issues automatically or flag for review
            generated_text = fix_issues(generated_text, issues)
        
        # Format as RTF
        rtf_content = create_rtf(generated_text, traffic_info["date"])
        
        # Save output
        save_broadcast_file(rtf_content, traffic_info["date"])
"""