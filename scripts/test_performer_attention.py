#!/usr/bin/env python3
"""
Test script to validate Performer attention implementation in FastViT.

This script checks:
1. If PerformerSelfAttention includes output projection
2. If double projection exists in MHSA
3. Shape compatibility
4. Forward pass correctness
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn

print("=" * 70)
print("Testing Performer Attention Implementation")
print("=" * 70)

# Test 1: Check if performer-pytorch is installed
print("\n[Test 1] Checking performer-pytorch installation...")
try:
    from performer_pytorch import SelfAttention as PerformerSelfAttention
    print("✓ performer-pytorch is installed")
except ImportError as e:
    print(f"✗ ERROR: performer-pytorch not installed: {e}")
    print("  Install with: pip install performer-pytorch")
    sys.exit(1)

# Test 2: Check PerformerSelfAttention structure
print("\n[Test 2] Analyzing PerformerSelfAttention structure...")
try:
    import inspect
    attn = PerformerSelfAttention(dim=512, heads=8, dim_head=64)
    
    # Check if it has output projection
    has_to_out = hasattr(attn, 'to_out') or hasattr(attn, 'to_out_proj') or hasattr(attn, 'out_proj')
    has_proj = hasattr(attn, 'proj')
    
    print(f"  PerformerSelfAttention attributes:")
    print(f"    - has 'to_out' or similar: {has_to_out}")
    print(f"    - has 'proj': {has_proj}")
    
    # List all attributes
    attn_attrs = [attr for attr in dir(attn) if not attr.startswith('_') and not callable(getattr(attn, attr, None))]
    print(f"    - All attributes: {attn_attrs[:10]}...")  # Show first 10
    
    # Check parameters
    num_params = sum(p.numel() for p in attn.parameters())
    print(f"    - Total parameters: {num_params:,}")
    
    if has_to_out or has_proj:
        print("  ⚠ WARNING: PerformerSelfAttention likely includes output projection!")
        print("    Adding another projection in MHSA would cause double projection.")
    else:
        print("  ✓ PerformerSelfAttention does NOT include output projection")
        print("    Adding projection in MHSA is correct.")
        
except Exception as e:
    print(f"✗ ERROR analyzing PerformerSelfAttention: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Test input/output shapes
print("\n[Test 3] Testing input/output shapes...")
try:
    attn = PerformerSelfAttention(dim=512, heads=8, dim_head=64)
    
    # Test 3D input (B, N, C)
    x_3d = torch.randn(2, 49, 512)  # (batch, tokens, dim)
    out_3d = attn(x_3d)
    print(f"  3D input:  {x_3d.shape} -> {out_3d.shape}")
    assert out_3d.shape == x_3d.shape, f"Shape mismatch: {out_3d.shape} != {x_3d.shape}"
    print("  ✓ 3D shape test passed")
    
except Exception as e:
    print(f"✗ ERROR in shape test: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test FastViT MHSA implementation
print("\n[Test 4] Testing FastViT Performer MHSA implementation...")
try:
    import fastvit.models as models
    from fastvit.models.fastvit_performer import MHSA
    
    mhsa = MHSA(dim=512, head_dim=64)
    
    # Test 4D input (B, C, H, W) - as used in FastViT
    x_4d = torch.randn(2, 512, 7, 7)  # (batch, channels, height, width)
    out_4d = mhsa(x_4d)
    print(f"  4D input:  {x_4d.shape} -> {out_4d.shape}")
    assert out_4d.shape == x_4d.shape, f"Shape mismatch: {out_4d.shape} != {x_4d.shape}"
    print("  ✓ 4D shape test passed")
    
    # Test 3D input
    x_3d = torch.randn(2, 49, 512)
    out_3d = mhsa(x_3d)
    print(f"  3D input:  {x_3d.shape} -> {out_3d.shape}")
    assert out_3d.shape == x_3d.shape, f"Shape mismatch: {out_3d.shape} != {x_3d.shape}"
    print("  ✓ 3D shape test passed")
    
    # Count parameters
    num_params = sum(p.numel() for p in mhsa.parameters())
    print(f"  Total parameters in MHSA: {num_params:,}")
    
    # Check if there's double projection
    has_attn_proj = hasattr(mhsa.attn, 'to_out') or hasattr(mhsa.attn, 'to_out_proj') or hasattr(mhsa.attn, 'out_proj')
    has_mhsa_proj = hasattr(mhsa, 'proj')
    
    if has_attn_proj and has_mhsa_proj:
        print("  ⚠ WARNING: Potential double projection detected!")
        print("    - PerformerSelfAttention has output projection")
        print("    - MHSA also has output projection (self.proj)")
        print("    - This may cause unexpected behavior")
    elif has_mhsa_proj:
        print("  ✓ Single projection in MHSA (correct if PerformerSelfAttention has no output projection)")
    else:
        print("  ⚠ WARNING: No output projection in MHSA!")
        
except Exception as e:
    print(f"✗ ERROR testing MHSA: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Compare with original MHSA
print("\n[Test 5] Comparing with original MHSA...")
try:
    from fastvit.models.fastvit import MHSA as OriginalMHSA
    
    orig_mhsa = OriginalMHSA(dim=512, head_dim=64)
    perf_mhsa = MHSA(dim=512, head_dim=64)
    
    x = torch.randn(2, 512, 7, 7)
    
    with torch.no_grad():
        orig_out = orig_mhsa(x)
        perf_out = perf_mhsa(x)
    
    print(f"  Original MHSA output shape: {orig_out.shape}")
    print(f"  Performer MHSA output shape: {perf_out.shape}")
    print(f"  Shapes match: {orig_out.shape == perf_out.shape}")
    
    orig_params = sum(p.numel() for p in orig_mhsa.parameters())
    perf_params = sum(p.numel() for p in perf_mhsa.parameters())
    print(f"  Original MHSA parameters: {orig_params:,}")
    print(f"  Performer MHSA parameters: {perf_params:,}")
    print(f"  Parameter difference: {perf_params - orig_params:,}")
    
except Exception as e:
    print(f"✗ ERROR comparing MHSA: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Test full AttentionBlock
print("\n[Test 6] Testing AttentionBlock with Performer...")
try:
    from fastvit.models.fastvit_performer import AttentionBlock
    
    block = AttentionBlock(dim=512, mlp_ratio=4.0)
    x = torch.randn(2, 512, 7, 7)
    
    with torch.no_grad():
        out = block(x)
    
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print("  ✓ AttentionBlock shape test passed")
    
except Exception as e:
    print(f"✗ ERROR testing AttentionBlock: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Test full model creation
print("\n[Test 7] Testing full model creation...")
try:
    from timm.models import create_model
    
    model = create_model("fastvit_sa12_P", pretrained=False, num_classes=37)
    print(f"  ✓ Model created successfully")
    print(f"  Model type: {type(model).__name__}")
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  ✓ Forward pass successful")
    
except Exception as e:
    print(f"✗ ERROR creating model: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Test Summary")
print("=" * 70)
print("If all tests passed, the Performer attention implementation is valid.")
print("Check warnings above for potential double projection issues.")
print("=" * 70)

