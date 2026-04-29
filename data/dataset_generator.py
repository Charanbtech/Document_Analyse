"""
Synthetic Dataset Generator for Document Classification
Generates realistic training data that mirrors the 20 Newsgroups distribution.
Used when the dataset cannot be downloaded from external sources.
"""

import random
import numpy as np
from typing import Tuple, List

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Category templates ────────────────────────────────────────────────────────

TEMPLATES = {
    "alt.atheism": {
        "words": [
            "atheism", "god", "religion", "belief", "faith", "bible", "church",
            "evolution", "morality", "agnostic", "secular", "prayer", "divine",
            "existence", "proof", "supernatural", "theist", "skeptic", "dogma",
            "scripture", "rationalism", "humanism", "creationism", "cosmos"
        ],
        "sentences": [
            "Many atheists argue that the existence of god cannot be proven empirically.",
            "The debate between religion and science continues in modern society.",
            "Secular humanism provides a moral framework without supernatural beliefs.",
            "The burden of proof lies with those who make extraordinary claims.",
            "Evolution is a well-established scientific theory supported by evidence.",
            "Religious faith often provides comfort but lacks empirical justification.",
            "Morality can exist independently of religious doctrine and scripture.",
            "The origins of the universe do not require a supernatural explanation.",
            "Many cultures have developed distinct religious traditions over centuries.",
            "Critical thinking and skepticism are cornerstones of the atheist worldview.",
        ]
    },
    "comp.graphics": {
        "words": [
            "graphics", "rendering", "pixel", "texture", "shader", "opengl", "3d",
            "image", "resolution", "vector", "rasterization", "polygon", "mesh",
            "animation", "ray-tracing", "algorithm", "display", "framerate", "gpu",
            "blender", "antialiasing", "normal-map", "lighting", "vertex", "buffer"
        ],
        "sentences": [
            "The new GPU supports real-time ray tracing for photorealistic rendering.",
            "OpenGL shaders allow fine-grained control over the graphics pipeline.",
            "Texture mapping applies 2D images to 3D polygon surfaces.",
            "Anti-aliasing techniques reduce jagged edges in rasterized images.",
            "The rendering engine uses a deferred shading approach for efficiency.",
            "Blender's Cycles renderer produces physically accurate lighting simulations.",
            "Vertex buffers store geometry data for fast GPU-accelerated rendering.",
            "Normal maps simulate surface detail without adding extra polygons.",
            "Screen-space ambient occlusion adds depth cues to 3D scenes.",
            "The framerate dropped below 60fps when rendering complex particle systems.",
        ]
    },
    "comp.sys.ibm.pc.hardware": {
        "words": [
            "cpu", "motherboard", "ram", "bios", "hard-drive", "ibm", "pc",
            "processor", "memory", "cache", "interrupt", "bus", "isa", "pci",
            "benchmark", "overclock", "cooling", "thermal", "slot", "driver",
            "firmware", "chipset", "voltage", "clock-speed", "compatible"
        ],
        "sentences": [
            "The motherboard supports dual-channel DDR4 RAM for better bandwidth.",
            "Overclocking the CPU requires adequate cooling and voltage adjustments.",
            "The BIOS update fixed compatibility issues with newer PCIe devices.",
            "ISA slots are obsolete but some legacy hardware still relies on them.",
            "A thermal paste reapplication reduced CPU temperatures by 10 degrees.",
            "The system posted successfully after reseating the RAM modules.",
            "PCIe 4.0 doubles the bandwidth available compared to the previous generation.",
            "Benchmark scores improved significantly after upgrading the processor.",
            "The hard drive controller firmware needed updating for stable operation.",
            "Interrupt conflicts between legacy devices caused system instability.",
        ]
    },
    "misc.forsale": {
        "words": [
            "sell", "sale", "price", "offer", "asking", "obo", "condition",
            "buy", "shipping", "payment", "used", "brand-new", "negotiable",
            "contact", "email", "pickup", "paypal", "item", "listing", "deal",
            "excellent", "mint", "working", "boxed", "original"
        ],
        "sentences": [
            "Selling my barely-used laptop, asking $600 OBO, includes charger.",
            "Brand new in box, never opened, willing to ship at buyer's expense.",
            "Item is in excellent condition, only used a few times since purchase.",
            "Contact me by email if interested, local pickup preferred.",
            "Price is negotiable, looking to sell quickly before the weekend.",
            "Accepting PayPal only, will ship within 2 business days of payment.",
            "Original box and all accessories included with this item for sale.",
            "Used for about 6 months, works perfectly, no scratches or dents.",
            "Selling due to upgrade, the item performs great but I need newer model.",
            "First come first served, cash or PayPal accepted for this listing.",
        ]
    },
    "rec.autos": {
        "words": [
            "car", "engine", "transmission", "brake", "mileage", "mpg", "horsepower",
            "dealership", "tire", "suspension", "mechanic", "oil", "exhaust",
            "turbo", "acceleration", "torque", "manual", "automatic", "highway",
            "rally", "sports", "sedan", "suv", "hybrid", "diesel"
        ],
        "sentences": [
            "The turbocharged engine produces 400 horsepower with excellent fuel economy.",
            "Replacing the brake pads and rotors improved stopping distance noticeably.",
            "A manual transmission gives the driver more control over gear selection.",
            "The suspension was upgraded with coilovers for better handling on corners.",
            "Regular oil changes every 5000 miles prolong engine life significantly.",
            "The dealership offered a competitive financing rate on the new sedan.",
            "Fuel economy dropped when switching from highway to city driving conditions.",
            "The mechanic diagnosed a faulty oxygen sensor causing rough idling.",
            "All-season tires offer a balance between wet and dry performance.",
            "The hybrid powertrain regenerates energy during braking to charge the battery.",
        ]
    },
    "rec.sport.hockey": {
        "words": [
            "hockey", "nhl", "puck", "goalie", "penalty", "power-play", "overtime",
            "playoffs", "stanley-cup", "goal", "assist", "ice", "rink", "shot",
            "trade", "forward", "defense", "team", "score", "season", "coach",
            "faceoff", "bodycheck", "save", "hat-trick"
        ],
        "sentences": [
            "The goalie made 45 saves in overtime to keep his team in the playoffs.",
            "A power-play goal in the final minute tied the game and sent it to overtime.",
            "The team announced a blockbuster trade acquiring a top defenseman.",
            "Hat tricks are rare but this forward has scored three this season.",
            "The Stanley Cup playoffs begin next week with eight teams competing.",
            "A bodycheck along the boards sent the puck directly to the winger.",
            "The coach pulled the goalie with two minutes left to add an extra attacker.",
            "Faceoff wins in the offensive zone are crucial for sustained pressure.",
            "The penalty kill unit successfully defended all four shorthanded situations.",
            "Back-to-back games on consecutive nights tested the team's endurance.",
        ]
    },
    "sci.med": {
        "words": [
            "medical", "patient", "drug", "treatment", "clinical", "diagnosis",
            "symptom", "disease", "therapy", "hospital", "surgery", "medicine",
            "vaccine", "antibody", "protein", "cell", "trial", "dosage", "side-effect",
            "chronic", "acute", "prescription", "immune", "pathogen", "physician"
        ],
        "sentences": [
            "The clinical trial showed a statistically significant reduction in symptoms.",
            "Patients with chronic conditions require careful medication management.",
            "The vaccine stimulates antibody production against the target pathogen.",
            "Surgery was recommended after conservative treatments failed to help.",
            "Side effects of the drug include nausea, dizziness, and mild headache.",
            "The physician ordered additional tests to confirm the preliminary diagnosis.",
            "Immune system suppression increases susceptibility to opportunistic infections.",
            "A new protein target was identified for treating the aggressive cancer type.",
            "The dosage was adjusted based on the patient's kidney function levels.",
            "Early detection of the disease dramatically improves treatment outcomes.",
        ]
    },
    "sci.space": {
        "words": [
            "nasa", "space", "rocket", "orbit", "satellite", "telescope", "galaxy",
            "planet", "astronaut", "mission", "launch", "solar", "asteroid", "probe",
            "cosmos", "universe", "gravity", "black-hole", "mars", "moon", "iss",
            "hubble", "exoplanet", "nebula", "spacecraft"
        ],
        "sentences": [
            "NASA's latest mission aims to collect samples from a near-Earth asteroid.",
            "The Hubble Space Telescope captured detailed images of a distant nebula.",
            "Astronauts aboard the ISS conducted experiments in microgravity conditions.",
            "The rocket successfully placed its payload into geostationary orbit.",
            "Scientists detected an exoplanet in the habitable zone of a nearby star.",
            "Mars rover data suggests the presence of ancient riverbeds on the surface.",
            "A black hole merger produced gravitational waves detected on Earth.",
            "The solar probe gathered unprecedented data about the sun's corona.",
            "Telescope arrays allow astronomers to study galaxies billions of light-years away.",
            "The spacecraft adjusted its trajectory using a gravity assist maneuver.",
        ]
    },
    "soc.religion.christian": {
        "words": [
            "christian", "church", "jesus", "bible", "faith", "prayer", "gospel",
            "salvation", "sin", "grace", "worship", "sermon", "baptism", "holy",
            "spirit", "cross", "resurrection", "covenant", "ministry", "congregation",
            "testimony", "disciple", "scripture", "redemption", "blessing"
        ],
        "sentences": [
            "The sermon this Sunday focused on the themes of grace and redemption.",
            "Scripture teaches that faith without works is considered incomplete.",
            "The congregation gathered for a special baptism ceremony at the river.",
            "Prayer is central to the Christian practice of seeking divine guidance.",
            "The gospel message of salvation has spread across cultures and centuries.",
            "The church ministry focuses on serving the poor and vulnerable community.",
            "Disciples of Jesus are called to love their neighbors as themselves.",
            "The resurrection of Jesus is the cornerstone of Christian doctrine.",
            "A testimony of personal transformation through faith inspired many.",
            "The holy spirit is believed to guide believers in their daily lives.",
        ]
    },
    "talk.politics.guns": {
        "words": [
            "gun", "firearm", "amendment", "rights", "control", "legislation",
            "weapon", "background-check", "ban", "regulation", "nra", "safety",
            "constitution", "permit", "concealed", "violence", "policy", "senate",
            "law", "crime", "enforcement", "militia", "license", "ownership", "rifle"
        ],
        "sentences": [
            "The Senate debated new gun control legislation following recent incidents.",
            "Background checks are required for all firearms purchases at licensed dealers.",
            "Second Amendment advocates argue that gun ownership is a constitutional right.",
            "The proposed ban on assault-style weapons faces strong opposition in Congress.",
            "Concealed carry permits require safety training and a background check.",
            "Law enforcement agencies have mixed views on stricter gun regulations.",
            "The NRA lobbied heavily against the proposed universal background check bill.",
            "Statistics on gun violence are cited by both sides of the policy debate.",
            "Many gun owners support responsible ownership and oppose illegal modifications.",
            "The state legislature passed a red flag law allowing temporary firearm removal.",
        ]
    }
}


def generate_document(category: str, min_sentences: int = 4, max_sentences: int = 10) -> str:
    """Generate a synthetic document for a given category."""
    tmpl = TEMPLATES[category]
    sentences = tmpl["sentences"]
    words = tmpl["words"]

    n = random.randint(min_sentences, max_sentences)
    chosen = random.choices(sentences, k=n)

    # Add word-based noise sentences
    extra_words = random.sample(words, min(5, len(words)))
    noise = f"Topics often discussed include {', '.join(extra_words[:3])} and related matters."
    chosen.insert(random.randint(0, len(chosen)), noise)

    return " ".join(chosen)


def generate_dataset(
    samples_per_class: int = 200,
    test_ratio: float = 0.2
) -> Tuple[List[str], List[str], List[int], List[int], List[str]]:
    """
    Generate train/test splits.

    Returns:
        X_train, X_test, y_train, y_test, class_names
    """
    categories = list(TEMPLATES.keys())
    X, y = [], []

    for label_idx, cat in enumerate(categories):
        for _ in range(samples_per_class):
            doc = generate_document(cat)
            X.append(doc)
            y.append(label_idx)

    X, y = np.array(X), np.array(y)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(len(X) * (1 - test_ratio))
    return (
        list(X[:split]), list(X[split:]),
        list(y[:split]), list(y[split:]),
        categories
    )


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, cats = generate_dataset(samples_per_class=100)
    print(f"Train: {len(X_tr)}, Test: {len(X_te)}, Classes: {len(cats)}")
    print("Sample:", X_tr[0][:120])
