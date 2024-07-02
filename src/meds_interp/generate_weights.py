
def get_ranges(modalities):
    modality_weights = []
    for modality in modalities:
        modality_weights.append(interval(0, 1))
    return modality_weights


def main():
    """Generates all of the ranges of modality weights to iterate through."""
    modalities = ["modality_1", "modality_2", "modality_3", "modality_4"]
    get_ranges(modalities)


if __name__ == "__main__":
    main()
