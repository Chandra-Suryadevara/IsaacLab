@echo off
ECHO =================================================================
ECHO Initializing Visual Studio and Conda Environment for VIDEO...
ECHO =================================================================

:: 1. Load your specific Visual Studio Developer environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"

:: 2. Activate your conda environment
call conda activate F:\envs\detectron2

:: 3. Set the PYTHONPATH to find the Detectron2 library
set PYTHONPATH=F:\detectron2_src

:: Create output directory if it doesn't exist
IF NOT EXIST "F:\cowinj\output" (
    ECHO Creating output directory...
    mkdir "F:\cowinj\output"
)

ECHO =================================================================
ECHO Environment is ready. Running the NECK DETECTION model...
ECHO =================================================================

:: 4. Run your Python video script with the new model and adjusted threshold
python F:\cowinj\process_neck_video.py ^
    --config-file "F:\cowinj\config.yaml" ^
    --input "F:\cowinj\Screen_Recording_2025-07-07_132050.mp4" ^
    --output "F:\cowinj\output\CowVid_neck_processed.mp4" ^
    --confidence-threshold 0.85 ^
    --opts MODEL.WEIGHTS "F:\cowinj\model_final.pth"

ECHO =================================================================
ECHO Video demo has finished. A video has been saved in the 'output' folder.
ECHO Press any key to close this window.
ECHO =================================================================

pause >nul
