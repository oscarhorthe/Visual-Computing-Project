%% ========================================================================
%  ANALYSIS OF STUDENT ACTIVITIES ON A UNIVERSITY CAMPUS
%  Main Project Script - PETS S2.L1 View001
%  ========================================================================
%  This script orchestrates all 8 tasks of the project.
%  Make sure the dataset is in the correct folder structure:
%    ./PETS-S2L1/
%       gt/gt.txt
%       View001/
%          frame_0001.jpg, frame_0002.jpg, ... frame_0795.jpg
%  ========================================================================
clc; clear; close all;

%% ---- Configuration -----------------------------------------------------
config.dataDir     = './PETS-S2L1';
config.gtFile      = fullfile(config.dataDir, 'gt', 'gt.txt');
config.imgDir      = fullfile(config.dataDir, 'View001');
config.outputDir   = './results';
config.numFrames   = 795;
config.imgExt      = '.jpg';  % Change to .png if needed

% Create output directory
if ~exist(config.outputDir, 'dir')
    mkdir(config.outputDir);
end

%% ---- Load Ground Truth -------------------------------------------------
fprintf('Loading ground truth...\n');
gt = loadGroundTruth(config.gtFile);
fprintf('  Loaded %d GT entries across %d frames, %d unique IDs.\n', ...
    size(gt, 1), numel(unique(gt(:,1))), numel(unique(gt(:,2))));

%% ---- Detect Image Naming Pattern ---------------------------------------
config.imgPattern = detectImagePattern(config.imgDir, config.imgExt);
fprintf('  Image pattern: %s\n', config.imgPattern);

%% ========================================================================
%  TASK 1: Plot GT bounding boxes on each frame (3.0v)
%  ========================================================================
fprintf('\n===== TASK 1: Ground Truth Visualization =====\n');
task1_plotGT(gt, config);

%% ========================================================================
%  TASK 2: Pedestrian detection + tracking with labels (4.0v)
%  ========================================================================
fprintf('\n===== TASK 2: Pedestrian Detection & Tracking =====\n');
detections = task2_detectAndTrack(gt, config);

%% ========================================================================
%  TASK 3: Plot trajectories dynamically (4.0v)
%  ========================================================================
fprintf('\n===== TASK 3: Trajectory Visualization =====\n');
task3_plotTrajectories(detections, config);

%% ========================================================================
%  TASK 4: Consistent label assignment (2.0v)
%  ========================================================================
fprintf('\n===== TASK 4: Consistent Label Assignment =====\n');
detections_consistent = task4_consistentLabels(detections, config);

%% ========================================================================
%  TASK 5: Heatmaps - static & dynamic (2.0v)
%  ========================================================================
fprintf('\n===== TASK 5: Heatmap Generation =====\n');
task5_heatmaps(detections_consistent, config);

%% ========================================================================
%  TASK 6: EM statistical analysis of trajectories (1.5v)
%  ========================================================================
fprintf('\n===== TASK 6: EM Statistical Analysis =====\n');
task6_EManalysis(detections_consistent, config);

%% ========================================================================
%  TASK 7: Evaluation metrics - IoU success plot, FP, FN (2.5v)
%  ========================================================================
fprintf('\n===== TASK 7: Evaluation Metrics =====\n');
task7_evaluation(gt, detections_consistent, config);

%% ========================================================================
%  TASK 8: DNN comparison (1.0v)
%  ========================================================================
fprintf('\n===== TASK 8: DNN Comparison =====\n');
task8_DNNcomparison(gt, detections_consistent, config);

fprintf('\n===== ALL TASKS COMPLETED =====\n');
fprintf('Results saved in: %s\n', config.outputDir);
