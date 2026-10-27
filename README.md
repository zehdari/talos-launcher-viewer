# Talos launcher viewer

Web viewer for the Talos torpedo/marker launcher, live at https://zehdari.github.io/talos-launcher-viewer/

- **V11**: clean native SolidWorks rebuild. The meshes are exported per component from `LV11 Launcher.SLDASM` and placed on the robot mounting rails.
- **Fire**: the servo angle drives the gear train. Servo 27T to shaft 18T is 1.5:1, and the 20:20 helix pair drives the marker shaft. Each payload rolls off its lock as the pusher reaches it, then launches at its release angle.
- **Assembly procedure**: 12 steps. Each part flies in along its insertion direction.
- **Section cut**: cuts along the torpedo axis, with solid caps.

Rebuild the assets: `build_glb_v11.py` (in `~/torpedo_work/viewer`).
