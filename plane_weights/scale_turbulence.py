def scale_turbulence(pilot_reported_turb, plane_weight):
    """
    Scale the pilot-reported turbulence intensity based on aircraft weight.
    
    Intensities range from 0 to 7.
    Weights:
        - 'L' (Light): No change
        - 'M' (Medium): +1
        - 'H' (Heavy): +2
        - 'U' (Unknown): No change 

    Parameters:
        pilot_reported_turb (int): Turbulence intensity (1-7).
        plane_weight (str): Aircraft weight category ('L', 'M', 'H').

    Returns:
        int: Scaled turbulence intensity.

    Assumptions:
        Note for unknown plane weights, we will leave turb intensity unchanged
        since almost all unknown plane weights are little small unique planes
    """
    
    # Ensure turbulence is within the valid range
    if not (0 <= pilot_reported_turb <= 7):
        raise ValueError("Turbulence intensity must be between 1 and 7.")
    
    # Ensure weight category is valid
    if plane_weight not in {'L', 'M', 'H', 'U'}:
        raise ValueError("Plane weight must be 'L', 'M', 'H' or 'U.")

    # Scale turbulence based on weight
    weight_adjustments = {'L': 0, 'M': 1, 'H': 2, 'U': 0 }
    scaled_turbulence = pilot_reported_turb + weight_adjustments[plane_weight]

    # Ensure the scaled turbulence does not exceed the max value of 7
    return scaled_turbulence
