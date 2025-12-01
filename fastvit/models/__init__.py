#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All rights reserved.
#
"""
FastViT Model Registry

This module provides both original and Performer variants of FastViT models.

Original Models (always available):
    - fastvit_t8, fastvit_t12, fastvit_s12
    - fastvit_sa12, fastvit_sa24, fastvit_sa36, fastvit_ma36

Performer Models (requires performer_pytorch):
    - fastvit_t8_P (and potentially more)
    
    To enable performer models: pip install performer-pytorch

Usage:
    # Check if performer models are available
    from fastvit.models import has_performer_models
    if has_performer_models():
        print("Performer models available!")
    
    # List all available models
    from fastvit.models import list_available_models
    models = list_available_models()
    print(f"Original: {models['original']}")
    print(f"Performer: {models['performer']}")
"""

# Original FastViT models (always available)
from .fastvit import (
    fastvit_t8,
    fastvit_t12,
    fastvit_s12,
    fastvit_sa12,
    fastvit_sa24,
    fastvit_sa36,
    fastvit_ma36,
)

# Optional import for Performer variants (requires performer_pytorch)
# Performer models use linear attention instead of standard attention
# To use performer models, install: pip install performer-pytorch
_has_performer = False
try:
    from .fastvit_performer import *
    _has_performer = True
except ImportError:
    # performer_pytorch not installed, performer models unavailable
    pass


def has_performer_models():
    """Check if Performer model variants are available.
    
    Returns:
        bool: True if performer_pytorch is installed and performer models are available.
    """
    return _has_performer


def list_available_models():
    """List all available model variants.
    
    Returns:
        dict: Dictionary with 'original' and 'performer' keys listing available models.
    """
    models = {
        'original': [
            'fastvit_t8',
            'fastvit_t12',
            'fastvit_s12',
            'fastvit_sa12',
            'fastvit_sa24',
            'fastvit_sa36',
            'fastvit_ma36',
        ],
        'performer': []
    }
    
    if _has_performer:
        # Performer models typically have _P suffix or similar naming
        # This would need to be updated if more performer models are added
        try:
            from .fastvit_performer import fastvit_t8_P
            from .fastvit_performer import fastvit_sa12_P
            models['performer'].append('fastvit_t8_P')
            models['performer'].append('fastvit_sa12_P')
        except (ImportError, AttributeError):
            pass
    
    return models  
