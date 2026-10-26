@echo off
echo ============================================================
echo OfficeBot RL - Training All Algorithms
echo ============================================================
echo Training all algorithms with 3 seeds each...
echo.

for %%A in (ppo dqn a2c) do (
    for %%S in (42 123 999) do (
        echo.
        echo ============================================================
        echo Training %%A seed %%S...
        echo ============================================================
        python training/train.py --algo %%A --seed %%S --config configs/%%A_config.yaml
        echo Done: %%A seed %%S
    )
)

echo.
echo ============================================================
echo All training complete!
echo ============================================================
