# Pedestrian Trajectory Analysis - PETS S2.L1
## Computer Vision Project

### Project Structure
```
project/
├── main.m                      % Main orchestrator script
├── loadGroundTruth.m           % Load and parse gt.txt
├── detectImagePattern.m        % Auto-detect image naming
├── task1_plotGT.m              % Task 1: GT bounding box visualization
├── task2_detectAndTrack.m      % Task 2: Detection + tracking (HOG/BG sub)
├── task3_plotTrajectories.m    % Task 3: Dynamic trajectory plots
├── task4_consistentLabels.m    % Task 4: Consistent ID assignment (Hungarian)
├── task5_heatmaps.m            % Task 5: Static & dynamic heatmaps
├── task6_EManalysis.m          % Task 6: EM/GMM statistical analysis
├── task7_evaluation.m          % Task 7: IoU success plot, FP/FN metrics
├── task8_DNNcomparison.m       % Task 8: DNN comparison
├── README.md                   % This file
└── PETS-S2L1/                  % Dataset (you must provide this)
    ├── gt/
    │   └── gt.txt
    └── View001/
        ├── frame_0001.jpg
        ├── frame_0002.jpg
        └── ... (795 frames)
```

### Requirements
- MATLAB R2019b or later
- Computer Vision Toolbox (for insertShape, VideoWriter, peopleDetectorACF)
- Statistics and Machine Learning Toolbox (for fitgmdist in Task 6)
- Deep Learning Toolbox (optional, for Task 8 DNN comparison)

### Setup
1. Download the PETS S2.L1 dataset from the course GitHub (Crowd-PETS directory)
2. Place the View001 frames in `./PETS-S2L1/View001/`
3. Place `gt.txt` in `./PETS-S2L1/gt/`
4. The script auto-detects image naming patterns

### Running
```matlab
>> main
```
This runs all 8 tasks sequentially. Results are saved in `./results/`.

### Algorithms Used
- **Task 2**: Adaptive Gaussian background subtraction + morphological filtering
  + connected component analysis + aspect ratio filtering
- **Task 4**: Hungarian algorithm with combined IoU + centroid distance cost
- **Task 5**: Gaussian kernel density estimation for heatmaps
- **Task 6**: GMM via EM algorithm (fitgmdist) with BIC model selection
- **Task 7**: IoU-based matching with greedy assignment
- **Task 8**: ACF detector (or YOLO v4 if Deep Learning Toolbox available)

### Notes
- Without images, Tasks 2/8 use GT-based simulated detections for testing
- The code handles missing images gracefully with placeholder visualizations
- All outputs (images, videos, reports) are saved to ./results/
