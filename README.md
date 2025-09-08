# AiM-IGTL-Connection
```mermaid
sequenceDiagram

title System Sequence Diagram

participant TVN as TheraVision Console
participant R as Robot
participant MRIGT  as MR IGTL Bridge
participant MRS as MR Scanner

alt Start Up
activate R
activate TVN
TVN->>R:Connect/Command via IGTL(STRING, START_UP)
R->>TVN:TRANSFORM(measured_cp, [-0.77  -0.64  -0.01  -24.59  0.02  -0.00  -1.00  57.85  0.64  -0.77  0.01  114.67  0.00  0.00  0.00  1.00  ])
R->>TVN:TRANSFORM(desired_cp, [-0.76  -0.65  -0.01  -24.85  0.00  0.01  -1.00  39.45  0.65  -0.76  -0.01  115.20  0.00  0.00  0.00  1.00  ])
R->>TVN:TRANSFORM(scanner_to_robot_reg, [1.00  -0.01  0.02  7.75  0.01  1.00  -0.02  -126.24  -0.02  0.02  1.00  27.91  0.00  0.00  0.00  1.00  ])
R->>TVN:POINT(entry_point, [0.144625   -37.1517   78.288])
note left of TVN: Update applicator transformation based on measure_cp
R->>TVN:POINT(target_point, [-0.263161   -91.5912   78.288])
R->>TVN:STRING(cannula_to_treatment, [5.000000])
R->>TVN:STRING(treatment_to_tip, [0.000000])
R->>TVN:STRING(robot_to_entry, [5.000000])
R->>TVN:STRING(robot_treatment_home, [41.000000])
R->>TVN:STRING(measured_jp, [0.028902,-0.672112,0.025132,0.000300,0.000000,-0.012700,-0.038100,])
R->>TVN:STRING(desired_jp, [11,-880,16,61470,1112,0,0,])
R->>TVN:STRING(measured_jv, [0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,])
deactivate R
deactivate TVN
end

alt Update information
activate R
activate TVN
activate MRS
MRS->>TVN:Registration DICOM image via TVN image server
note left of TVN:⠀Auto generate registration matrix
TVN->>R:TRANSFORM(REGISTRATION, [ 1.00 -0.01 0.02 7.75   0.01 1.00 -0.02 -126.24   -0.02 0.02 1.00 27.91   0.00 0.00 0.00 1.00 ])
MRS->>TVN:Patient DICOM image via TVN image server
note left of TVN:⠀Select entry and target point
TVN->>R:TRANSFORM(TARGET_POINT, [ 0.59 0.81 0.00 -17.62   0.05 -0.03 -1.00 -16.45   -0.80 0.59 -0.06 104.80   0.00 0.00 0.00 1.00   ])
R->>TVN:POINT(target_point, [-17.62   -16.45   104.8])
R->>TVN:STRING(desired_jp, [-38,2859,6,140524,330,-1361,-807,])
R->>TVN:TRANSFORM(desired_cp, [0.59  0.81  0.00  -25.37  0.05  -0.03  -1.00  42.05  -0.81  0.59  -0.06  101.97  0.00  0.00  0.00  1.00  ])
TVN->>R:TRANSFORM(ENTRY_POINT, [ 1.00 0.00 0.00 -17.88   0.00 1.00 0.00 45.71   0.00 0.00 1.00 104.81   0.00 0.00 0.00 1.00 ])
R->>TVN:POINT(entry_point, [-17.8831   45.7067   104.806])
R->>TVN:STRING(desired_jp, [12,-865,7,87210,325,-1023,-463,])
R->>TVN:TRANSFORM(desired_cp, [-0.77  -0.64  0.00  -25.27  -0.01  0.00  -1.00  39.90  0.64  -0.77  -0.01  105.85  0.00  0.00  0.00  1.00  ])
note left of TVN: Update applicator transformation based on measure_cp
deactivate R
deactivate TVN
deactivate MRS
end

loop When scan is active or robot is active
MRS->>R:DICOM Image
R->>MRIGT:MRTI image generation
MRIGT->>TVN: MRTI image
R->>TVN:STRING(measured_jv, [0.000000,1.475065,0.000000,0.000000,0.000000,0.000000,0.000000,])
R->>TVN:TRANSFORM(measured_cp, [-0.77  -0.64  -0.01  -24.03  -0.18  0.24  -0.95  42.82  0.61  -0.73  -0.30  108.96  0.00  0.00  0.00  1.00  ])
R->>TVN:STRING(measured_jp, [-12.854989,-0.669000,0.015079,38.645623,1.121931,-12.369825,-5.473711,])
end

Note over TVN,R:
    TVN→Robot IGTL connection.<br/>
    TVN=client, Robot=server.<br/>
    Server IP:192.168.88.250, Port:18936.<br/>
    Scanner IP:10.0.1.1, Port:15002.<br/><br/>
    Robot front end→TVN IGTL MRTI.<br/>
    TVN=client, Frontend=server.<br/>
    Server IP:10.0.1.229, Port:18944.<br/>
    Acquisition rate:10
