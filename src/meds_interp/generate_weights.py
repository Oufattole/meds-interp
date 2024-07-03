import sys


def get_ranges(modalities):
    modality_weights = ["modalities=[" + ",".join(modalities) + "]"]
    for modality in modalities:
        modality_weights.append(f"+weights.{modality}=interval(0,1)")
    return modality_weights


def main():
    """Generates all of the ranges of modality weights to iterate through."""
    modalities = list(sys.argv[1].strip("[]").split(","))
    print(" ".join(get_ranges(modalities)))


if __name__ == "__main__":
    main()
