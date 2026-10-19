# Neon Rider
Night-city highway motorbike dodger. C++17 + OpenGL ES 3.0 (NativeActivity), APK built by GitHub Actions.

**Get the APK:** Actions tab -> latest "Build APK" run -> Artifacts -> `NeonRider-apk`.

## Controls
- GAS (right thumb) accelerates, BRAKE slows, NITRO boosts. No reverse.
- Steer: tilt (default), drag a finger, or on-screen arrows. CAM button switches 3rd/1st person.
- BRAKE (bottom-left) / NITRO (bottom-right). Left 2 lanes are oncoming traffic: 2x score but dangerous.
- Near-misses build a combo (up to x8).

## Performance tiers (auto-detected by RAM/cores, override via GFX in menu)
| Tier | Bike tris | Cars | Render scale | FPS cap |
|------|-----------|------|--------------|---------|
| LOW  | ~2k  | 9  | 45-62% (dynamic) | 30 |
| MID  | ~6k  | 13 | 55-85% (dynamic) | 60 |
| HIGH | ~14k | 18 | 70-100% (dynamic) + radial speed blur | 60 |

## Re-generating the compact meshes
`pip install trimesh numpy pillow pyfqmr` then `python3 tools/convert_models.py <dir with the 3 .glb> assets`
