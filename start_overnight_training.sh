#!/bin/bash
# Start Overnight Training Script - 500 Videos
# Run this before bed for maximum training time

echo "🌙 VideoLLaMA3 Overnight Training Setup"
echo "Training 500 videos with blind navigation annotations"
echo "Optimized for M3 Max - Sleep Training Configuration"
echo "=" * 60

# Check system status
echo "🔍 System Check:"
echo "Date: $(date)"
echo "Available RAM: $(vm_stat | grep free | awk '{print $3}' | sed 's/\.//' | awk '{print $1 * 4 / 1024 / 1024}') GB"
echo "Battery: $(pmset -g batt | grep -o '[0-9]*%')"

# Prevent sleep during training
echo "⚡ Preventing system sleep..."
caffeinate -d -i -m -s &
CAFFEINATE_PID=$!
echo "Caffeinate PID: $CAFFEINATE_PID"

# Set up environment
echo "🔧 Setting up environment..."
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=8

# Start training with comprehensive logging
echo "🚀 Starting overnight training..."
echo "Training will continue while you sleep 😴"
echo "Check progress with: tail -f logs/overnight_training_*.log"

# Run training and capture exit code
python train_500_videos_overnight.py 2>&1 | tee "logs/overnight_training_$(date +%Y%m%d_%H%M%S)_console.log"
TRAINING_EXIT_CODE=$?

# Kill caffeinate process
kill $CAFFEINATE_PID 2>/dev/null

# Final status
echo ""
echo "🏁 Training Session Completed"
echo "Exit code: $TRAINING_EXIT_CODE"
echo "Completion time: $(date)"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "Model saved to: ./trained_models/blind_navigation_500_videos_final"
    echo ""
    echo "📋 Next steps:"
    echo "1. Check model quality: python test_trained_model.py"
    echo "2. Push to GitHub when ready"
    echo "3. Test on UC Merced cluster"
else
    echo "❌ Training failed with exit code: $TRAINING_EXIT_CODE"
    echo "Check logs for error details"
    echo "Emergency checkpoint may be available"
fi

echo ""
echo "🔋 System can now sleep normally"