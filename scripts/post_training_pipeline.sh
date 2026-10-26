#!/bin/bash
# Post-training evaluation and visualization pipeline

echo "========================================="
echo "OfficeBot RL - Post-Training Pipeline"
echo "========================================="
echo ""

# Wait for all training to complete
echo "Waiting for training to complete..."
echo ""

# Evaluate all three algorithms
echo "Step 1: Evaluating trained models..."
echo ""

for algo in ppo dqn a2c; do
    echo "Evaluating $algo (seed 42)..."
    PYTHONPATH=. python evaluation/evaluate.py --algo $algo --seed 42 --episodes 50
    echo ""
done

# Compare algorithms
echo "Step 2: Comparing algorithms..."
echo ""
PYTHONPATH=. python evaluation/compare.py

# Generate plots
echo "Step 3: Generating visualization plots..."
echo ""
PYTHONPATH=. python visualization/plot_results.py --algo all --seeds 42

echo ""
echo "========================================="
echo "Pipeline Complete!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - results/comparison_table.csv"
echo "  - results/experiment_summary.json"
echo "  - results/plots/"
echo ""
