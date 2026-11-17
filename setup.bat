@echo off
REM ============================================================
REM  Installation en un double-clic (Windows)
REM  Crée l'environnement virtuel .venv et installe tout.
REM ============================================================
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    set "PY=py -3"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo [ERREUR] Python est introuvable. Installez Python 3.10+ depuis https://www.python.org
        echo          en cochant "Add python.exe to PATH".
        pause
        exit /b 1
    )
    set "PY=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creation de l'environnement virtuel .venv ...
    %PY% -m venv .venv || (echo [ERREUR] Creation du venv impossible & pause & exit /b 1)
) else (
    echo [1/3] Environnement .venv deja present.
)

echo [2/3] Installation des dependances (quelques minutes la premiere fois) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo [ERREUR] Installation echouee & pause & exit /b 1)

echo [3/3] Enregistrement du noyau Jupyter "Python (quantum-classifier)" ...
".venv\Scripts\python.exe" -m ipykernel install --user --name quantum-classifier --display-name "Python (quantum-classifier)"

echo.
echo Installation terminee.
echo   Activer l'environnement :   .venv\Scripts\activate
echo   Lancer les tests       :   python -m pytest
echo   Lancer le benchmark    :   python run_benchmark.py
echo   Ouvrir les notebooks   :   jupyter lab
pause
