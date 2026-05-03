import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()
random.seed(42)


LINES_OF_BUSINESS = [
    "Commercial Property",
    "General Liability",
    "Cyber",
    "Marine & Energy",
    "Professional Liability",
    "Workers Compensation",
    "Property Catastrophe Reinsurance",
    "Casualty Treaty Reinsurance",
]


LOSS_TYPES = {
    "Commercial Property": ["Fire", "Water Damage", "Wind", "Hail", "Theft"],
    "General Liability": ["Bodily Injury", "Slip and Fall", "Property Damage"],
    "Cyber": ["Ransomware", "Data Breach", "Business Interruption"],
    "Marine & Energy": ["Cargo Damage", "Hull Damage", "Energy Equipment Failure"],
    "Professional Liability": ["Negligence", "Misrepresentation", "Service Error"],
    "Workers Compensation": ["Back Injury", "Slip Injury", "Equipment Injury"],
    "Property Catastrophe Reinsurance": ["Hurricane", "Wildfire", "Flood", "Hailstorm"],
    "Casualty Treaty Reinsurance": ["Large Liability Loss", "Umbrella Claim", "Mass Tort"],
}


STATES = ["TX", "FL", "CA", "NY", "IL", "GA", "LA", "NJ"]
CAT_STATES = {"TX", "FL", "CA", "LA"}
TREATY_TYPES = ["None", "Quota Share", "Excess of Loss", "Catastrophe XOL"]


def generate_policy(policy_id: str | None = None) -> dict:
    lob = random.choice(LINES_OF_BUSINESS)

    return {
        "policy_id": policy_id or f"POL-{random.randint(100000, 999999)}",
        "insured_name": fake.company(),
        "insured_type": random.choice(["Small Business", "Middle Market", "Large Commercial"]),
        "line_of_business": lob,
        "policy_limit": random.choice([250000, 500000, 1000000, 5000000, 10000000]),
        "deductible": random.choice([1000, 2500, 5000, 10000, 25000]),
        "state": random.choice(STATES),
        "effective_date": str(datetime.today().date() - timedelta(days=random.randint(30, 500))),
        "expiration_date": str(datetime.today().date() + timedelta(days=random.randint(30, 365))),
    }


def generate_treaty(treaty_id: str | None = None) -> dict:
    treaty_type = random.choice(TREATY_TYPES)

    return {
        "treaty_id": treaty_id or f"TRT-{random.randint(10000, 99999)}",
        "treaty_type": treaty_type,
        "retention": random.choice([100000, 250000, 500000, 1000000]),
        "treaty_limit": random.choice([1000000, 2500000, 5000000, 10000000]),
        "quota_share_percent": random.choice([0, 10, 20, 30, 40]) if treaty_type == "Quota Share" else 0,
        "reinsurer": random.choice(["Global Re", "Atlantic Re", "Summit Re", "Harbor Re"]),
    }


def generate_exposure(policy_id: str | None = None) -> dict:
    state = random.choice(STATES)

    return {
        "policy_id": policy_id or f"POL-{random.randint(100000, 999999)}",
        "location_id": f"LOC-{random.randint(10000, 99999)}",
        "state": state,
        "zip_code": fake.zipcode(),
        "insured_value": random.choice([100000, 250000, 500000, 1000000, 5000000]),
        "construction_type": random.choice(["Frame", "Masonry", "Steel", "Concrete"]),
        "occupancy_type": random.choice(["Retail", "Warehouse", "Office", "Manufacturing"]),
        "cat_zone": int(state in CAT_STATES),
    }


def generate_cat_event() -> dict:
    return {
        "event_id": f"CAT-{random.randint(1000, 9999)}",
        "event_type": random.choice(["Hurricane", "Wildfire", "Flood", "Hailstorm"]),
        "state": random.choice(["TX", "FL", "CA", "LA"]),
        "severity": random.choice(["Low", "Medium", "High", "Extreme"]),
        "event_date": str(datetime.today().date() - timedelta(days=random.randint(1, 90))),
    }


def generate_claim() -> dict:
    policy = generate_policy()
    treaty = generate_treaty()
    exposure = generate_exposure(policy["policy_id"])

    lob = policy["line_of_business"]
    loss_type = random.choice(LOSS_TYPES[lob])

    prior_claim_count = random.randint(0, 5)
    litigation_flag = random.choice([0, 1])

    base_amount = random.uniform(5000, 600000)

    cat_exposure = exposure["cat_zone"] and loss_type in [
        "Hurricane",
        "Wildfire",
        "Flood",
        "Hailstorm",
        "Wind",
        "Hail",
    ]

    severity_multiplier = (
        1
        + 0.25 * prior_claim_count
        + 0.50 * litigation_flag
        + 0.75 * int(cat_exposure)
    )

    claim_amount = round(base_amount * severity_multiplier, 2)

    reinsurance_triggered = (
        treaty["treaty_type"] != "None"
        and claim_amount > treaty["retention"]
    )

    ceded_loss = (
        max(0, min(claim_amount - treaty["retention"], treaty["treaty_limit"]))
        if reinsurance_triggered
        else 0
    )

    severity_label = (
        "High"
        if claim_amount >= 300000
        else "Medium"
        if claim_amount >= 75000
        else "Low"
    )

    suspicious_flag = int(
        prior_claim_count >= 3
        or litigation_flag == 1
        or claim_amount > policy["policy_limit"] * 0.8
    )

    claim_id = f"CLM-{random.randint(100000, 999999)}"

    return {
        "claim_id": claim_id,
        "policy_id": policy["policy_id"],
        "treaty_id": treaty["treaty_id"],
        "claimant_id": f"CUST-{random.randint(10000, 99999)}",
        "claimant_name": fake.name(),
        "claimant_email": fake.email(),
        "claimant_phone": fake.phone_number(),
        "claimant_address": fake.address().replace("\n", ", "),
        "line_of_business": lob,
        "loss_type": loss_type,
        "state": policy["state"],
        "loss_date": str(datetime.today().date() - timedelta(days=random.randint(1, 365))),
        "claim_amount": claim_amount,
        "reserve_amount": round(claim_amount * random.uniform(0.85, 1.3), 2),
        "prior_claim_count": prior_claim_count,
        "litigation_flag": litigation_flag,
        "cat_exposure": int(cat_exposure),
        "suspicious_flag": suspicious_flag,
        "severity_label": severity_label,
        "treaty_type": treaty["treaty_type"],
        "retention": treaty["retention"],
        "treaty_limit": treaty["treaty_limit"],
        "reinsurance_triggered": int(reinsurance_triggered),
        "ceded_loss": round(ceded_loss, 2),
        "claim_note": (
            f"Claim {claim_id} reported under {lob}. "
            f"Loss type is {loss_type}. Initial loss estimate is ${claim_amount:,.0f}. "
            f"Litigation flag is {litigation_flag}. Prior claim count is {prior_claim_count}. "
            f"Review coverage, reserve adequacy, and reinsurance impact."
        ),
    }


def generate_claims(n: int = 10) -> list[dict]:
    return [generate_claim() for _ in range(n)]