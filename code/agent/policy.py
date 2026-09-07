import numpy as np

class PhysicalParameterAgentEngine:
    """
    Proactive Autonomous Agent that monitors physical turbofan parameters
    (T30 HPC Outlet Temp, P30 HPC Pressure, Fan Speed Nf) and predicts RUL shifts
    to issue dynamic control & maintenance actions.
    """
    def __init__(self, egt_limit_degR=1580.0, max_p30_limit=555.0):
        self.egt_limit_degR = egt_limit_degR
        self.max_p30_limit = max_p30_limit

    def evaluate_physical_telemetry(self, predicted_rul, current_cycle, avg_t30, avg_p30, temp_offset=0.0, pressure_offset=0.0):
        """
        Evaluates physical sensor parameters and RUL predictions to issue targeted physical control recommendations.
        """
        effective_t30 = avg_t30 + temp_offset
        effective_p30 = avg_p30 + pressure_offset

        # Default values
        life_gained = 0.0
        egt_drop = 0.0

        # Calculate dynamic derate recommendation based on temperature spike & RUL
        if effective_t30 >= self.egt_limit_degR or predicted_rul <= 15.0:
            action_type = "SCHEDULE_INSPECTION"
            recommended_derate_pct = 12.5
            action_headline = f"SCHEDULE IMMEDIATE INSPECTION WITHIN {max(1, int(predicted_rul))} CYCLES"
            derate_factor = 0.0
            rationale = (
                f"Physical Stress Alert: HPC Temperature T30 reached {effective_t30:.1f}°R "
                f"(limit: {self.egt_limit_degR}°R) and predicted RUL dropped to {predicted_rul:.1f} cycles. "
                f"High thermal creep detected on High-Pressure Turbine blades. Ground engine for physical overhaul."
            )
        elif temp_offset > 15.0 or predicted_rul <= 35.0:
            recommended_derate_pct = round(min(10.0, 5.0 + temp_offset * 0.15 + (35.0 - predicted_rul) * 0.1), 1)
            action_type = "REDUCE_THRUST_HEAVY"
            action_headline = f"REDUCE THRUST BY {recommended_derate_pct}%"
            derate_factor = 1.0 - (recommended_derate_pct / 100.0)
            egt_drop = round(recommended_derate_pct * 3.2, 1)
            life_gained = round(predicted_rul * ((derate_factor ** -2.5) - 1.0), 1) if derate_factor > 0 else 0.0
            rationale = (
                f"Thermal Stress Warning: Elevated HPC Outlet Temp T30 = {effective_t30:.1f}°R (+{temp_offset:.1f}°R offset). "
                f"Predicted RUL decreased to {predicted_rul:.1f} cycles. "
                f"Proactively reducing thrust by {recommended_derate_pct}% lowers T30 by ~{egt_drop}°R, "
                f"extending time-on-wing by +{life_gained} cycles."
            )
        elif temp_offset > 5.0 or predicted_rul <= 60.0:
            recommended_derate_pct = round(4.0 + temp_offset * 0.1, 1)
            action_type = "REDUCE_THRUST_LIGHT"
            action_headline = f"REDUCE THRUST BY {recommended_derate_pct}%"
            derate_factor = 1.0 - (recommended_derate_pct / 100.0)
            egt_drop = round(recommended_derate_pct * 3.0, 1)
            life_gained = round(predicted_rul * ((derate_factor ** -2.5) - 1.0), 1) if derate_factor > 0 else 0.0
            rationale = (
                f"Moderate Load Advisory: Operating under T30 = {effective_t30:.1f}°R (+{temp_offset:.1f}°R offset). "
                f"Reducing thrust by {recommended_derate_pct}% mitigates thermal wear, gaining +{life_gained} cycles."
            )
        else:
            recommended_derate_pct = 0.0
            action_type = "NOMINAL"
            action_headline = "CONTINUE NOMINAL OPERATION (100% RATING)"
            derate_factor = 1.0
            egt_drop = 0.0
            life_gained = 0.0
            rationale = (
                f"Nominal Parameters: T30 = {effective_t30:.1f}°R, P30 = {effective_p30:.1f} psia. "
                f"Predicted RUL = {predicted_rul:.1f} cycles. Engine operating within healthy envelope."
            )

        nom_life = max(0.0, predicted_rul)
        ext_life = nom_life + life_gained

        counterfactual_matrix = [
            {
                "Candidate Operating Mode": "1. Standard Power (0% Derate)",
                "Physical T30 Temp": f"{effective_t30:.1f} °R",
                "Predicted RUL": f"{nom_life:.1f} cycles",
                "Thermal Wear Rate": "100% (High Thermal Stress)"
            },
            {
                "Candidate Operating Mode": f"2. Apply {recommended_derate_pct}% Thrust Derate (Recommended)",
                "Physical T30 Temp": f"{effective_t30 - egt_drop:.1f} °R (-{egt_drop}°R)",
                "Predicted RUL": f"{ext_life:.1f} cycles (+{life_gained} cycles)",
                "Thermal Wear Rate": f"{round((derate_factor**2.5)*100, 1)}% (Extended Life)"
            },
            {
                "Candidate Operating Mode": "3. Immediate Engine Grounding",
                "Physical T30 Temp": "Ambient (Engine Off)",
                "Predicted RUL": "0 cycles",
                "Thermal Wear Rate": "0% (High Downtime Penalty)"
            }
        ]

        return {
            "cycle": current_cycle,
            "predicted_rul": round(predicted_rul, 1),
            "effective_t30": round(effective_t30, 1),
            "effective_p30": round(effective_p30, 1),
            "temp_offset": temp_offset,
            "pressure_offset": pressure_offset,
            "action_type": action_type,
            "action_headline": action_headline,
            "derate_pct": recommended_derate_pct,
            "derate_factor": derate_factor,
            "gained_cycles": life_gained,
            "egt_temp_drop": egt_drop,
            "rationale": rationale,
            "counterfactual_matrix": counterfactual_matrix,
            "status": "MAINTENANCE_TRIGGERED" if action_type == "SCHEDULE_INSPECTION" else "IN_SERVICE"
        }
