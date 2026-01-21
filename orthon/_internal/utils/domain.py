"""
PRISM Domain Utilities
======================

Domain selection and validation for PRISM pipelines.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict

import yaml


def get_domains_config_path() -> Path:
    """Get path to domains.yaml config."""
    return Path(__file__).parent.parent.parent / "config" / "domains.yaml"


def load_domains() -> Dict[str, dict]:
    """Load available domains from config/domains.yaml."""
    config_path = get_domains_config_path()
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config.get('domains', {})


def list_domains() -> List[str]:
    """Get list of available domain names."""
    return list(load_domains().keys())


def get_domain_info(domain: str) -> Optional[dict]:
    """Get info for a specific domain."""
    domains = load_domains()
    return domains.get(domain)


def validate_domain(domain: str) -> bool:
    """Check if domain exists in registry."""
    return domain in load_domains()


def prompt_for_domain(message: str = "Select domain") -> str:
    """
    Interactive domain picker.

    Shows numbered list of available domains and prompts for selection.
    Returns selected domain name.
    """
    domains = load_domains()

    if not domains:
        print("No domains configured in config/domains.yaml")
        sys.exit(1)

    print(f"\n{message}:")
    print("-" * 50)

    domain_list = list(domains.keys())
    for i, name in enumerate(domain_list, 1):
        info = domains[name]
        desc = info.get('description', '')
        print(f"  [{i}] {name:<12} - {desc}")

    print("-" * 50)

    while True:
        try:
            choice = input(f"Enter number (1-{len(domain_list)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(domain_list):
                selected = domain_list[idx]
                print(f"Selected: {selected}\n")
                return selected
            else:
                print(f"Invalid choice. Enter 1-{len(domain_list)}")
        except ValueError:
            # Allow typing domain name directly
            if choice in domain_list:
                print(f"Selected: {choice}\n")
                return choice
            print(f"Invalid input. Enter number or domain name.")
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            sys.exit(0)


def require_domain(domain: Optional[str], prompt_message: str = "Select domain") -> str:
    """
    Require a valid domain - prompt if not provided.

    Args:
        domain: Domain name (may be None)
        prompt_message: Message to show when prompting

    Returns:
        Valid domain name (from arg or interactive selection)
    """
    if domain:
        if not validate_domain(domain):
            print(f"Unknown domain: {domain}")
            print(f"Available: {', '.join(list_domains())}")
            sys.exit(1)
        return domain

    return prompt_for_domain(prompt_message)
