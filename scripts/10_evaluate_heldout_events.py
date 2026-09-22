TRAINING_ALLOWED = False


def main():
    if TRAINING_ALLOWED:
        raise RuntimeError("Training is forbidden in held-out evaluation.")
    print("Held-out evaluator guard is active.")
    print("For exact paper reproduction, restore the final Phase7D-D prediction/result artifacts and use reproduce_results.py.")
    print("Re-running inference is a separate medium-cost reproduction route.")


if __name__ == "__main__":
    main()
