@echo off

REM ============================
REM Activate virtual environment
REM ============================
IF NOT DEFINED VIRTUAL_ENV (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) ELSE (
    echo Virtual environment already activated.
)

REM ============================
REM Clean previous outputs
REM ============================
echo Cleaning old results...

IF EXIST results (
    rmdir /S /Q results
)

IF EXIST datasets\pav_features (
    rmdir /S /Q datasets\pav_features
)

REM ============================
REM Run pipeline
REM ============================
echo Running pipeline...

python pipeline\0_extract_features.py --num_partitions 1 || goto :error
python pipeline\1_train_val_split.py || goto :error

python pipeline\2_ppca.py --latent-dim 2   || goto :error
python pipeline\2_ppca.py --latent-dim 16  || goto :error
python pipeline\2_ppca.py --latent-dim 32  || goto :error
python pipeline\2_ppca.py --latent-dim 128 || goto :error

python pipeline\3_naive.py || goto :error

echo Pipeline completed successfully.
goto :eof

:error
echo.
echo ERROR: Pipeline failed.
exit /b 1
