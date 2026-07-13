#!/usr/bin/env bash
# ==============================================================================
# ntp.sh - Automated Build, Flash Deployment & Serial Diagnostics
# Platform Target: ESP8266 (Wemos D1 Mini / NodeMCU)
# ==============================================================================
set -Eeuo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { printf "%b[INFO]%b %s\n" "${GREEN}" "${NC}" "$*"; }
warn() { printf "%b[WARN]%b %s\n" "${YELLOW}" "${NC}" "$*"; }
err()  { printf "%b[ERR ]%b %s\n" "${RED}" "${NC}" "$*" >&2; }
step() { printf "%b[STEP]%b %s\n" "${BLUE}" "${NC}" "$*"; }

trap 'err "Script failed at line $LINENO (exit code: $?)"' ERR

BOARD_PROFILE="${BOARD_PROFILE:-esp8266:esp8266:d1_mini}"
BAUD_FLASH="${BAUD_FLASH:-460800}"
BAUD_FLASH_FALLBACK="${BAUD_FLASH_FALLBACK:-115200}"
BAUD_MONITOR="${BAUD_MONITOR:-115200}"
BUILD_DIR="${BUILD_DIR:-./build_out}"
LOCAL_BIN="${HOME}/.local/bin"

require_cmd() { command -v "$1" >/dev/null 2>&1; }

ensure_path() {
  mkdir -p "$LOCAL_BIN"
  export PATH="${PATH}:${LOCAL_BIN}"
}

install_arduino_cli_linux() {
  if require_cmd arduino-cli; then return 0; fi
  warn "arduino-cli not found."
  read -rp "Install arduino-cli automatically now? [y/N]: " ans
  case "${ans:-N}" in
    y|Y|yes|YES)
      step "Installing arduino-cli to ${LOCAL_BIN}..."
      curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | BINDIR="${LOCAL_BIN}" sh
      require_cmd arduino-cli || { err "arduino-cli install failed."; exit 1; }
      
      log "Configuring ESP8266 hardware platform indices..."
      arduino-cli config init --overwrite
      arduino-cli config set board_manager.additional_urls https://arduino.esp8266.com/stable/package_esp8266com_index.json
      ;;
    *) err "arduino-cli is required. Install it and rerun."; exit 1 ;;
  esac
}

ensure_esp8266_core() {
  step "Checking Arduino core index..."
  arduino-cli core update-index >/dev/null 2>&1 || true
  if ! arduino-cli core list | grep -q '^esp8266:esp8266'; then
    warn "ESP8266 core not installed."
    log "Installing esp8266:esp8266 ..."
    arduino-cli core install esp8266:esp8266
  fi
}

detect_serial_port() {
  local p=""
  for d in /dev/ttyUSB*; do [[ -e "$d" ]] || continue; p="$d"; break; done
  if [[ -z "$p" && "${ALLOW_ACM_PORTS:-0}" == "1" ]]; then
    for d in /dev/ttyACM*; do [[ -e "$d" ]] || continue; p="$d"; break; done
  fi
  [[ -n "$p" ]] || return 1
  printf "%s\n" "$p"
}

check_port_permissions() {
  local port="$1"
  if [[ ! -r "$port" || ! -w "$port" ]]; then
    warn "Insufficient permissions on ${port}."
    log "Recommended:"
    echo "  sudo usermod -aG dialout \$USER"
    echo "  newgrp dialout   # or log out/in"
    read -rp "Attempt temporary sudo chmod 666 ${port}? [y/N]: " ans
    case "${ans:-N}" in
      y|Y|yes|YES) sudo chmod 666 "$port" ;;
      *) warn "Skipping chmod. Flash may fail without proper permissions." ;;
    esac
  fi
}

run_esptool_flash_id() {
  local port="$1"
  if esptool --port "$port" flash-id >/dev/null 2>&1; then
    return 0
  fi

  warn "flash-id default reset failed, retrying with no_reset control lines..."
  if esptool --port "$port" --before no_reset --after no_reset flash-id >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

probe_chip() {
  local port="$1"
  step "Probing ESP8266 chip on ${port}..."
  if run_esptool_flash_id "$port"; then
    log "Chip communication OK."
    return 0
  fi
  err "Failed chip probe. Try: disconnect/reconnect USB, close serial monitors, or set ESP_PORT explicitly."
  return 1
}

find_firmware_bin() {
  local target_name="$1"
  local expected_file="${BUILD_DIR}/${target_name}.bin"
  
  if [[ -f "$expected_file" ]]; then
    printf "%s\n" "$expected_file"
    return 0
  fi
  
  # Fallback to scanning timestamp if standard names break
  local latest
  latest="$(find "$BUILD_DIR" -maxdepth 2 -type f -name '*.bin' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
  [[ -n "$latest" ]] || return 1
  printf "%s\n" "$latest"
}

compile_single_ino_via_temp_sketch() {
  local ino_path="$1"        # ./esp8266_ntp.ino
  local sketch_name="$2"     # esp8266_ntp
  local temp_dir="${BUILD_DIR}/sketch_sandbox/${sketch_name}"

  [[ -f "$ino_path" ]] || { err "Missing sketch file: $ino_path"; exit 1; }

  rm -rf "$temp_dir"
  mkdir -p "$temp_dir"
  cp "$ino_path" "${temp_dir}/${sketch_name}.ino"

  step "Compiling ${ino_path} via standard isolated layout folder..."
  arduino-cli compile --fqbn "$BOARD_PROFILE" --output-dir "$BUILD_DIR" "$temp_dir"
}

compile_sketch_dir() {
  local dir_path="$1"
  [[ -d "$dir_path" ]] || { err "Missing sketch directory: $dir_path"; exit 1; }
  step "Compiling sketch directory ${dir_path} ..."
  arduino-cli compile --fqbn "$BOARD_PROFILE" --output-dir "$BUILD_DIR" "$dir_path"
}

flash_bin() {
  local port="$1"
  local fw="$2"
  local before_mode="${ESPTOOL_BEFORE:-default_reset}"
  local after_mode="${ESPTOOL_AFTER:-hard_reset}"

  step "Flashing firmware (${fw}) to ${port} (baud=${BAUD_FLASH})..."

  if esptool --chip esp8266 --port "$port" \
      --before "$before_mode" --after "$after_mode" \
      --baud "$BAUD_FLASH" \
      write-flash --flash-mode dio 0x0 "$fw"; then
    log "Flash successful."
    return 0
  fi

  warn "Flash failed with default reset strategy; retrying no_reset + lower baud..."
  if esptool --chip esp8266 --port "$port" \
      --before no_reset --after no_reset \
      --baud "$BAUD_FLASH_FALLBACK" \
      write-flash --flash-mode dio 0x0 "$fw"; then
    log "Flash successful (fallback mode)."
    return 0
  fi

  err "Flash failed in both normal and fallback mode."
  return 1
}

open_monitor() {
  local port="$1"
  step "Opening serial monitor (${BAUD_MONITOR} baud) on ${port}..."
  if require_cmd picocom; then
    picocom -b "$BAUD_MONITOR" "$port"
  elif require_cmd minicom; then
    minicom -D "$port" -b "$BAUD_MONITOR"
  elif require_cmd screen; then
    screen "$port" "$BAUD_MONITOR"
  else
    err "No serial monitor found (install picocom, minicom, or screen)."
    exit 1
  fi
}

# --- Main Logic Window Execution ---
printf "%b========================================================%b\n" "${GREEN}" "${NC}"
printf "%b      UniversalBit ESP8266 Auto-Compile & Flash System   %b\n" "${GREEN}" "${NC}"
printf "%b========================================================%b\n" "${GREEN}" "${NC}"

ensure_path
require_cmd curl || { err "curl is missing. Install: sudo apt install curl"; exit 1; }
require_cmd esptool || { err "esptool is missing. Install: sudo apt install esptool"; exit 1; }

install_arduino_cli_linux
ensure_esp8266_core
mkdir -p "$BUILD_DIR"

if [[ -n "${ESP_PORT:-}" ]]; then
  TARGET_PORT="${ESP_PORT}"
  log "Using manual ESP_PORT override: ${TARGET_PORT}"
else
  step "Detecting serial port..."
  TARGET_PORT="$(detect_serial_port)" || { err "No ESP8266 serial device found on /dev/ttyUSB*."; exit 1; }
  log "Detected hardware target: ${TARGET_PORT}"
fi

[[ -e "$TARGET_PORT" ]] || { err "Selected port does not exist: ${TARGET_PORT}"; exit 1; }

check_port_permissions "$TARGET_PORT"
probe_chip "$TARGET_PORT"

printf "%b--------------------------------------------------------%b\n" "${NC}"
echo "Select Deployment Task:"
echo "1) Build & Flash Baseline NTP Monitor (esp8266_ntp.ino)"
echo "2) Build & Flash Fibonacci Clock (fibonacci_ntp/fibonacci_ntp.ino)"
echo "3) Install display libs + Build & Flash Fibonacci Clock"
echo "4) Launch Real-Time Serial Monitor"
echo "5) Exit"
read -rp "Execute Selection (1-5): " USER_SEL

case "${USER_SEL}" in
  1)
    compile_single_ino_via_temp_sketch "./esp8266_ntp.ino" "esp8266_ntp"
    FW_PATH="$(find_firmware_bin "esp8266_ntp")" || { err "Compiled binary not found in ${BUILD_DIR}"; exit 1; }
    flash_bin "$TARGET_PORT" "$FW_PATH"
    ;;
  2)
    compile_sketch_dir "./fibonacci_ntp"
    FW_PATH="$(find_firmware_bin "fibonacci_ntp")" || { err "Compiled binary not found in ${BUILD_DIR}"; exit 1; }
    flash_bin "$TARGET_PORT" "$FW_PATH"
    ;;
  3)
    log "Installing optional display libraries..."
    arduino-cli lib install "Adafruit ILI9341" "Adafruit GFX Library" >/dev/null 2>&1 || true
    compile_sketch_dir "./fibonacci_ntp"
    FW_PATH="$(find_firmware_bin "fibonacci_ntp")" || { err "Compiled binary not found in ${BUILD_DIR}"; exit 1; }
    flash_bin "$TARGET_PORT" "$FW_PATH"
    ;;
  4)
    open_monitor "$TARGET_PORT"
    ;;
  *)
    log "Exiting diagnostics pipeline."
    exit 0
    ;;
esac

log "Done."
