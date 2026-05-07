"""Reward functions for different legal domains."""

from legal_lm.rewards.administrative import build_administrative_reward_functions
from legal_lm.rewards.civil import build_civil_reward_functions
from legal_lm.rewards.common import build_common_reward_functions
from legal_lm.rewards.criminal import build_criminal_reward_functions

SUPPORTED_REWARD_PROFILES = {"general", "criminal", "civil", "administrative"}


def build_reward_functions(profile: str) -> list:
    reward_functions = build_common_reward_functions()
    profile = (profile or "general").lower()
    if profile not in SUPPORTED_REWARD_PROFILES:
        supported = ", ".join(sorted(SUPPORTED_REWARD_PROFILES))
        raise ValueError(f"Unsupported reward profile: {profile}. Expected one of: {supported}")
    if profile == "criminal":
        reward_functions.extend(build_criminal_reward_functions())
    elif profile == "civil":
        reward_functions.extend(build_civil_reward_functions())
    elif profile == "administrative":
        reward_functions.extend(build_administrative_reward_functions())
    return reward_functions


__all__ = ["SUPPORTED_REWARD_PROFILES", "build_reward_functions"]
