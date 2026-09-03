"""
Matthews Correlation Coefficient (MCC) calculation module.
"""
import math


def matthews_correlation_coefficient(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Calculate the Matthews Correlation Coefficient.
    
    Args:
        tp: True positives
        tn: True negatives
        fp: False positives
        fn: False negatives
    
    Returns:
        The MCC score as a float between -1 and 1.
    """
    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    if denominator == 0:
        return 0.0
      
    return numerator / denominator