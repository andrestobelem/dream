"""Controles prospectivos separados para cumplimiento y fidelidad.

Las referencias anteriores, ya inspeccionadas, se usan para desarrollo.
Estas narraciones no aparecen en las demostraciones del prompt.
"""

CASES = [
    (
        "observatory",
        "I was in an observatory. My uncle adjusted the telescope while I watched. I did not touch the telescope.",
        [
            ("The narrator is in an observatory.", "preserved"),
            ("The narrator feels frightened.", "omitted"),
            ("The narrator adjusts the telescope.", "contradicted"),
        ],
    ),
    (
        "swimming_pool",
        "My coach swam across the pool. I stood on the dry tiles and did not enter the water.",
        [
            ("The coach swims across the pool.", "preserved"),
            ("The coach wears a striped swimming cap.", "omitted"),
            ("The narrator enters the water.", "contradicted"),
        ],
    ),
    (
        "bakery",
        "The owner handed me a loaf in the bakery. I stayed behind the counter the whole time and never went outside.",
        [
            ("The bakery owner hands the narrator a loaf.", "preserved"),
            ("The narrator feels calm.", "omitted"),
            ("The narrator goes outside.", "contradicted"),
        ],
    ),
    (
        "space_capsule",
        "A red light blinked above my seat in the sealed space capsule. I watched it and did not leave the capsule.",
        [
            ("A red light blinks above the narrator's seat.", "preserved"),
            ("A robot accompanies the narrator.", "omitted"),
            ("The narrator leaves the capsule.", "contradicted"),
        ],
    ),
    (
        "rock_climbing",
        "I waited on a narrow ledge. My cousin lowered a rope and pulled me up. I did not pull him up.",
        [
            ("The cousin pulls the narrator up.", "preserved"),
            ("The narrator feels relieved.", "omitted"),
            ("The narrator pulls their cousin up.", "contradicted"),
        ],
    ),
    (
        "recording_studio",
        "The drummer played in the recording studio. I listened from a chair and did not play the drums.",
        [
            ("A drummer plays in the recording studio.", "preserved"),
            ("The studio smells of oranges.", "omitted"),
            ("The narrator plays the drums.", "contradicted"),
        ],
    ),
]


def examples():
    strict = [
        {
            "id": f"prospective-{name}-{i}",
            "source_group": name,
            "requirement": requirement,
            "candidate": text,
            "expected": expected,
            "reviewer": "synthetic_control",
            "dimension": "relation_control",
        }
        for name, text, requirements in CASES
        for i, (requirement, expected) in enumerate(requirements, 1)
    ]
    fidelity_cases = [
        (
            "museum",
            "I walked through a museum with my nephew. We looked at a statue. I did not touch the statue.",
            [
                ("The narrator walks quietly through a museum with their nephew.", "preserved"),
                ("The narrator finds a lost wallet.", "omitted"),
                ("The narrator touches the statue.", "contradicted"),
            ],
        ),
        (
            "farm",
            "I was at a farm with my grandfather. He fed the animals while I stood by the gate. I did not feed them.",
            [
                ("The narrator is at an old farm at dawn with their grandfather.", "preserved"),
                ("The narrator knows the barn will close.", "omitted"),
                ("The narrator feeds the animals.", "contradicted"),
            ],
        ),
        (
            "concert_hall",
            "I sat in a concert hall. A violinist played nearby. I felt nervous, not relaxed.",
            [
                (
                    "The narrator listens quietly to a violinist in a large concert hall.",
                    "preserved",
                ),
                ("The narrator loses their coat.", "omitted"),
                ("The narrator feels relaxed.", "contradicted"),
            ],
        ),
    ]
    return strict + [
        {
            "id": f"prospective-fidelity-{name}-{i}",
            "source_group": name,
            "requirement": requirement,
            "candidate": text,
            "expected": expected,
            "reviewer": "synthetic_control",
            "dimension": "fidelidad",
        }
        for name, text, requirements in fidelity_cases
        for i, (requirement, expected) in enumerate(requirements, 1)
    ]
