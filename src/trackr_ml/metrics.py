from __future__ import annotations


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def compute_binary_metrics(
    y_true: list[int], y_pred: list[int]
) -> dict[str, float]:
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(tp + tn, len(y_true))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_abstention_metrics(
    probabilities: list[float],
    y_true: list[int],
    lower_bound: float,
    upper_bound: float,
) -> dict[str, float]:
    covered = 0
    correct = 0
    unknown = 0

    for truth, probability in zip(y_true, probabilities):
        if probability >= upper_bound:
            prediction = 1
        elif probability <= lower_bound:
            prediction = 0
        else:
            unknown += 1
            continue

        covered += 1
        if prediction == truth:
            correct += 1

    return {
        "coverage": safe_divide(covered, len(y_true)),
        "selective_accuracy": safe_divide(correct, covered),
        "unknown_rate": safe_divide(unknown, len(y_true)),
    }
