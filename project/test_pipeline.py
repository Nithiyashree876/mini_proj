import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fusion.multimodal_fusion import MultimodalFusion
from language.context_engine import ContextEngine
from language.notification_generator import NotificationGenerator
from utils.database import IdentityDatabase

def run_headless_test():
    print("=" * 60)
    print("  MULTIMODAL IDENTITY RECOGNITION - PIPELINE TEST  ")
    print("=" * 60)
    
    db = IdentityDatabase()
    fusion = MultimodalFusion()
    context = ContextEngine()
    notifier = NotificationGenerator()
    
    # ── Fetch Alice's ID for simulation ───────────────────────
    alice_pid = None
    for pid, info in db.get_all_identities().items():
        if info['name'] == 'Alice':
            alice_pid = pid
            break
            
    if not alice_pid:
        print("Demo identities not found. Please run setup_demo.py first.")
        return

    # ── TEST 1: Both modalities match (Alice) ─────────────────
    print("\n[TEST 1] SIMULATING FACE & VOICE MATCH (ALICE)")
    face_res = {'name': 'Alice', 'person_id': alice_pid, 'confidence': 0.95}
    voice_res = {'name': 'Alice', 'person_id': alice_pid, 'confidence': 0.88}
    
    print(" ➔ Face says: Alice (95%)")
    print(" ➔ Voice says: Alice (88%)")
    
    fusion_res = fusion.fuse(face_res, voice_res, None)
    ctx = context.update(fusion_res)
    notif = notifier.generate(ctx, fusion_res)
    
    print(f"\n >>> {notif['icon']} [{notif['level']}] {notif['message']}")
    print(f"     Details: {notif['detail']}")

    # ── TEST 2: Someone unknown walks in ──────────────────────
    print("\n" + "-" * 50)
    print("\n[TEST 2] SIMULATING UNKNOWN PERSON")
    
    face_unk = {'name': 'Unknown', 'person_id': None, 'confidence': 0.12}
    print(" ➔ Face says: Unknown (12%)")
    print(" ➔ Voice says: None")
    
    fusion_unk = fusion.fuse(face_unk, None, None)
    ctx_unk = context.update(fusion_unk)
    notif_unk = notifier.generate(ctx_unk, fusion_unk)
    
    print(f"\n >>> {notif_unk['icon']} [{notif_unk['level']}] {notif_unk['message']}")
    print(f"     Details: {notif_unk['detail']}")

    # ── TEST 3: Spoof attempt detected ────────────────────────
    print("\n" + "-" * 50)
    print("\n[TEST 3] SIMULATING ANTI-SPOOF TRIGGER")
    
    face_spoof = {'name': 'Bob', 'person_id': 'person_002', 'confidence': 0.90}
    spoof_info = {
        'is_spoof': True, 
        'spoof_confidence': 0.85, 
        'reasons': ["Low texture variance (possible printed photo)", "No natural micro-movements detected"]
    }
    
    print(" ➔ Face says: Bob (90%) - BUT Spoofing detected!")
    
    fusion_sp = fusion.fuse(face_spoof, None, spoof_info)
    ctx_sp = context.update(fusion_sp)
    notif_sp = notifier.generate(ctx_sp, fusion_sp)
    
    print(f"\n >>> {notif_sp['icon']} [{notif_sp['level']}] {notif_sp['message']}")
    print(f"     Details: {notif_sp['detail']}")

    print("\n" + "=" * 60)

if __name__ == '__main__':
    run_headless_test()
