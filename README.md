# EE22005 Mouse Race Lap Time Simulator
**Author:** ma3403  
**Date:** 20 March 2026

## Overview
Progressive three-stage Python simulation predicting lap time 
for the EE22005 Project Week 3 mouse race competition.

## Files in this Repository
| File | Description |
|---|---|
| mouse_simulation.py | Main simulation script |
| test_plan.xlsx | Comprehensive test plan with 23 test cases |
| flowchart.png | Mermaid flowchart of program logic |
| uml_diagram.png | PlantUML class diagram |

## Simulation Stages
- Stage 1: Kinematic constant-speed model
- Stage 2: DC motor torque and acceleration model
- Stage 3: PID differential steering model

## How to Run
python mouse_simulation.py

## Change Log
| Commit | Change | Reason |
|---|---|---|
| Initial commit | Added mouse_simulation.py | Base simulation uploaded |
| Add supporting docs | Added test plan, flowchart, UML diagram | Project documentation |
| Update RPM | MOTOR_NO_LOAD_RPM 200 to 592 | Lab tachometer measurement |
| Update torque | MOTOR_STALL_TORQUE_NM 0.05 to 0.03 | Datasheet review |


