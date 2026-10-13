def get_screening_notes(is_potable):
    """Return general educational notes for the model prediction."""
    if is_potable:
        return [
            "The model assigned this sample to the potable class in its training dataset.",
            "This is not a guarantee of drinking-water safety; some contaminants may not be included in these features.",
            "For real-world decisions, use an accredited laboratory and follow local public-health guidance.",
        ]
    return [
        "The model assigned this sample to the non-potable class in its training dataset.",
        "Treat this as a screening result, not a definitive measurement of water safety.",
        "Arrange laboratory testing and consult local water-quality authorities before use.",
    ]
