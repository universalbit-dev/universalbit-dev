#!/usr/bin/env bash
# ==============================================================================
# fib_esp8266_firmware.sh
# UniversalBit Project - Fibonacci NTP Clock Firmware Uploader Tool
# Supports Arduino Classic (Uno), ESP8266 (D1 Mini / R3 D1), and ESP32
# ==============================================================================

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Resolve true user directory even when executing with sudo
REAL_USER="${SUDO_USER:-$USER}"
REAL_HOME="$(eval echo "~${REAL_USER}")"

CLI_DIR="${REAL_HOME}/.local/bin"
ARDUINO_CLI="${CLI_DIR}/arduino-cli"

echo -e "${GREEN}========================================================${NC}"
echo -e "${GREEN}     UniversalBit Fibonacci NTP Clock Uploader          ${NC}"
echo -e "${GREEN}========================================================${NC}"

# 1. Dynamically scan for all .ino files in the current workspace
mapfile -t sketches < <(ls *.ino 2>/dev/null || true)

if [ ${#sketches[@]} -eq 0 ]; then
    echo -e "${RED}Error: No .ino files found in this directory!${NC}"
    echo -e "Please ensure fibonacci_ntp.ino is in: $(pwd)${NC}"
    exit 1
fi

echo -e "${YELLOW}Select the NTP Clock sketch to compile and deploy:${NC}"
for i in "${!sketches[@]}"; do
    echo "$((i+1))) ${sketches[$i]}"
done

read -rp "Selection [1-${#sketches[@]}]: " SKETCH_CHOICE

# Validate choice
if [[ ! "$SKETCH_CHOICE" =~ ^[0-9]+$ ]] || [ "$SKETCH_CHOICE" -lt 1 ] || [ "$SKETCH_CHOICE" -gt "${#sketches[@]}" ]; then
    echo -e "${RED}Invalid selection. Exiting.${NC}"
    exit 1
fi

SKETCH_PATH="${sketches[$((SKETCH_CHOICE-1))]}"
echo -e "${GREEN}✔ Target Selected: $SKETCH_PATH${NC}"

# 2. Install Arduino CLI locally to user path if it does not exist
if [ ! -f "$ARDUINO_CLI" ]; then
    echo -e "${YELLOW}Arduino CLI not found. Installing locally to $CLI_DIR...${NC}"
    mkdir -p "$CLI_DIR"
    curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | BINDIR="$CLI_DIR" sh
    chown -R "${REAL_USER}:${REAL_USER}" "${REAL_HOME}/.local"
else
    echo -e "${GREEN}Using existing local Arduino CLI installation at $ARDUINO_CLI.${NC}"
fi

# 3. Reset and initialize configuration under the actual user context
sudo -u "$REAL_USER" -H "$ARDUINO_CLI" config init --overwrite > /dev/null

# Expand network timeouts to prevent download drops during heavy toolchain installs
sudo -u "$REAL_USER" -H "$ARDUINO_CLI" config set network.connection_timeout 600s

# 4. Add stable ESP32 and ESP8266 repository URLs
echo -e "${YELLOW}Adding board manager index URLs...${NC}"
sudo -u "$REAL_USER" -H "$ARDUINO_CLI" config add board_manager.additional_urls "https://espressif.github.io/arduino-esp32/package_esp32_index.json"
sudo -u "$REAL_USER" -H "$ARDUINO_CLI" config add board_manager.additional_urls "https://arduino.esp8266.com/stable/package_esp8266com_index.json"

echo -e "${YELLOW}Updating package index...${NC}"
sudo -u "$REAL_USER" -H "$ARDUINO_CLI" core update-index

# 5. Auto-Detect connected Serial Port
echo -e "\n${YELLOW}Scanning for connected devices...${NC}"
PORT=""
candidates=($(compgen -G "/dev/ttyUSB*" || true) $(compgen -G "/dev/ttyACM*" || true))

if [ ${#candidates[@]} -gt 0 ]; then
    PORT="${candidates[0]}"
    echo -e "${GREEN}Auto-detected serial port: $PORT${NC}"
    read -rp "Press Enter to use $PORT, or type your preferred port: " USER_PORT
    if [ -n "$USER_PORT" ]; then
        PORT="$USER_PORT"
    fi
else
    echo -e "${YELLOW}No serial ports automatically detected.${NC}"
    read -rp "Enter the port your device is connected to (e.g. /dev/ttyUSB0 or COM3): " PORT
fi

if [ -z "$PORT" ]; then
    echo -e "${RED}Error: Port cannot be empty!${NC}"
    exit 1
fi

# 6. Hardware Auto-Probing using esptool
echo -e "${YELLOW}Probing hardware architecture on $PORT...${NC}"
DETECTED_CHIP="unknown"

if command -v esptool &> /dev/null; then
    PROBE_OUT=$(esptool --port "$PORT" --baud 115200 chip_id 2>&1 || true)
    if echo "$PROBE_OUT" | grep -qi "ESP8266"; then
        DETECTED_CHIP="esp8266"
    elif echo "$PROBE_OUT" | grep -qi "ESP32"; then
        DETECTED_CHIP="esp32"
    else
        DETECTED_CHIP="avr"
    fi
fi

# 7. Core and FQBN Setup
FQBN=""
CORE=""

if [ "$DETECTED_CHIP" = "esp8266" ]; then
    echo -e "${GREEN}✔ Auto-detected Hardware: ESP8266${NC}"
    echo "Please select your exact ESP8266 board variation:"
    echo "1) Wemos D1 Mini (or compatible)"
    echo "2) NodeMCU 1.0 (ESP-12E Module) / WeMos D1 R3"
    read -rp "Selection [1-2]: " ESP_CHOICE
    case "$ESP_CHOICE" in
        1) FQBN="esp8266:esp8266:d1_mini"; CORE="esp8266:esp8266" ;;
        2) FQBN="esp8266:esp8266:nodemcuv2"; CORE="esp8266:esp8266" ;;
        *) FQBN="esp8266:esp8266:d1_mini"; CORE="esp8266:esp8266" ;; # Default Fallback
    esac
elif [ "$DETECTED_CHIP" = "esp32" ]; then
    echo -e "${GREEN}✔ Auto-detected Hardware: ESP32 board${NC}"
    FQBN="esp32:esp32:esp32"
    CORE="esp32:esp32@2.0.17"
else
    echo -e "${YELLOW}Hardware auto-detection inconclusive. Please select manually:${NC}"
    echo "1) Arduino Classic (Uno R3 / Classic)"
    echo "2) ESP8266 Wemos D1 Mini"
    echo "3) ESP8266 WeMos D1 R3 (NodeMCU 1.0)"
    echo "4) ESP32 Dev Module (Stable 2.x Core)"
    read -rp "Selection [1-4]: " BOARD_CHOICE

    case "$BOARD_CHOICE" in
        1) FQBN="arduino:avr:uno"; CORE="arduino:avr" ;;
        2) FQBN="esp8266:esp8266:d1_mini"; CORE="esp8266:esp8266" ;;
        3) FQBN="esp8266:esp8266:nodemcuv2"; CORE="esp8266:esp8266" ;;
        4) FQBN="esp32:esp32:esp32"; CORE="esp32:esp32@2.0.17" ;;
        *) echo -e "${RED}Invalid selection. Exiting.${NC}"; exit 1 ;;
    esac
fi

# 8. Install compiler core with fallback retries
echo -e "${YELLOW}Installing/Updating compiler core platform: $CORE...${NC}"
max_retries=3
count=0
success=false

while [ $count -lt $max_retries ]; do
    if sudo -u "$REAL_USER" -H "$ARDUINO_CLI" core install "$CORE"; then
        success=true
        break
    else
        count=$((count+1))
        echo -e "${YELLOW}Network glitch detected. Retrying toolchain download ($count/$max_retries)...${NC}"
        sleep 3
    fi
done

if [ "$success" = false ]; then
    echo -e "${RED}Error: Failed to install compiler core toolchain after $max_retries attempts.${NC}"
    exit 1
fi

# 9. Set up Isolated Sandbox Directory to avoid compiler conflicts
BUILD_SANDBOX="/tmp/arduino_build_sandbox"
rm -rf "$BUILD_SANDBOX"
mkdir -p "$BUILD_SANDBOX"

SKETCH_NAME="${SKETCH_PATH%.*}"
mkdir -p "$BUILD_SANDBOX/$SKETCH_NAME"
cp "$SKETCH_PATH" "$BUILD_SANDBOX/$SKETCH_NAME/$SKETCH_NAME.ino"
chown -R "${REAL_USER}:${REAL_USER}" "$BUILD_SANDBOX"

# 10. Compile and Upload
echo -e "\n${YELLOW}Compiling $SKETCH_PATH in isolated sandbox...${NC}"
sudo -u "$REAL_USER" -H "$ARDUINO_CLI" compile --fqbn "$FQBN" "$BUILD_SANDBOX/$SKETCH_NAME"

echo -e "${YELLOW}Uploading compiled binary to $PORT...${NC}"
sudo -u "$REAL_USER" -H "$ARDUINO_CLI" upload -p "$PORT" --fqbn "$FQBN" "$BUILD_SANDBOX/$SKETCH_NAME"

# Clean up
rm -rf "$BUILD_SANDBOX"

echo -e "${GREEN}✔ Successfully compiled and uploaded $SKETCH_PATH to your device!${NC}"
