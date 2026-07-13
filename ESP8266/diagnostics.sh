#!/usr/bin/env bash
# diagnostics.sh - Multi-Architecture Local Serial Monitor Engine

echo "🔍 Initializing UniversalBit Core Hardware Diagnostics Pipeline..."

# 1. Verification for tool binary
if command -v arduino-cli &> /dev/null; then
    CLI_EXEC="arduino-cli"
elif [ -f "./bin/arduino-cli" ]; then
    CLI_EXEC="./bin/arduino-cli"
else
    echo "📥 'arduino-cli' is missing. Initiating automated installation..."
    if curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh; then
        CLI_EXEC="./bin/arduino-cli"
        echo "✅ 'arduino-cli' successfully prepared locally."
    else
        echo "❌ Critical Error: Failed to provision the toolchain."
        exit 1
    fi
fi

# 2. Hardware Port Detection
PORT=$(ls /dev/ttyUSB* 2>/dev/null | head -n 1)
if [ -z "$PORT" ]; then
    PORT=$(ls /dev/ttyACM* 2>/dev/null | head -n 1)
fi

if [ -z "$PORT" ]; then
    echo "❌ Error: No connected microcontroller detected on /dev/ttyUSB* or /dev/ttyACM*."
    exit 1
fi

echo "🔌 Target connection confirmed at: $PORT"
echo "------------------------------------------------------------"

# 3. Contextualized Environment Menu
echo "💡 Choose Target Device Diagnostic Profile:"
echo "1) Arduino Classic Baseline (9600 Baud - e.g., Uno / Nano / Mega)"
echo "2) ESP32 Environment       (115200 Baud - e.g., NodeMCU-32S / GRBL CNC)"
echo "3) ESP8266 Environment     (115200 Baud - e.g., D1 Mini / NodeMCU)"
echo "4) ESP8266 Boot ROM Debug  (74880 Baud - Raw Reset Boot Diagnostics)"
echo "5) Generic Terminal        (Custom Entry - Define your own speed)"
echo "6) Exit Diagnostics"
read -rp "Select profile option [1-6]: " choice

case $choice in
    1)
        BAUD=9600
        ;;
    2|3)
        BAUD=115200
        ;;
    4)
        BAUD=74880
        ;;
    5)
        echo "------------------------------------------------------------"
        read -rp "⌨️  Enter custom Baud Rate speed (e.g., 4800, 19200, 57600, 230400): " custom_baud
        BAUD=$custom_baud
        ;;
    6|*)
        echo "👋 Diagnostics cancelled."
        exit 0
        ;;
esac

echo "------------------------------------------------------------"
echo "🚀 Spawning Live Terminal Pipeline on $PORT @ $BAUD Baud..."
echo "👉 Press [CTRL + C] at any time to kill the monitor feed."
echo "------------------------------------------------------------"

# 4. Local execution bypasses all core package check dependencies 
$CLI_EXEC monitor -p "$PORT" --config baudrate="$BAUD" --additional-urls ""
