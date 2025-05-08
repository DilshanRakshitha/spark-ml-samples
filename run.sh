#!/usr/bin/env bash

# Purpose: Build and run the Spark ML Samples API application,
#          and attempt to open the frontend in the browser.
# Place this script in the root directory: dilshanrakshitha-spark-ml-samples/
# Usage: ./run.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
EXPECTED_PORT=9090
FRONTEND_URL="http://localhost:${EXPECTED_PORT}"
# Adjust the pattern if your shadow JAR naming convention is different
JAR_NAME_PATTERN="api-*-SNAPSHOT.jar"
API_BUILD_DIR="api/build/libs"
# Time to wait for the server to start before trying to open the browser (in seconds)
SERVER_START_WAIT_TIME=5

# --- Functions ---
cleanup() {
  echo "" # Newline for better readability after Ctrl+C
  echo "Script interrupted. Stopping application (if running)..."
  # Find and kill the Java process
  # Using pkill with a more specific pattern if possible, or kill the background PID
  if [ -n "$APP_PID" ]; then
    kill "$APP_PID" 2>/dev/null || echo "Application process with PID $APP_PID not found or already stopped."
  else
    # Fallback to pkill if PID wasn't captured (e.g., if run without backgrounding)
    pkill -f "$JAR_NAME_PATTERN" || echo "Application process (pkill) not found or already stopped."
  fi
  exit 1
}

open_browser() {
  local url="$1"
  echo "Attempting to open $url in your default browser..."
  # Cross-platform browser opening
  if [[ "$(uname)" == "Darwin" ]]; then # macOS
    open "$url"
  elif [[ "$(expr substr $(uname -s) 1 5)" == "Linux" ]]; then # Linux
    # Check for common browsers
    if command -v xdg-open &> /dev/null; then
      xdg-open "$url"
    elif command -v gnome-open &> /dev/null; then
      gnome-open "$url"
    elif command -v sensible-browser &> /dev/null; then
      sensible-browser "$url"
    else
      echo "Could not find a command to open the browser (xdg-open, gnome-open, sensible-browser)."
      echo "Please open $url manually."
    fi
  elif [[ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" || "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" || "$(expr substr $(uname -s) 1 5)" == "MSYS_" ]]; then # Git Bash or MSYS on Windows
    start "$url"
  else
    echo "Unsupported OS for automatic browser opening: $(uname)"
    echo "Please open $url manually."
  fi
}

# Trap SIGINT (Ctrl+C) and SIGTERM to attempt cleanup
trap cleanup SIGINT SIGTERM

# --- Main Script ---

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Change to the script's directory (project root)
cd "$SCRIPT_DIR"
echo "Changed directory to: $(pwd)"

echo "----------------------------------------"
echo "1. Building the project (if needed)..."
echo "----------------------------------------"
# Run the build command specified in README, skip tests for faster execution run
./gradlew build shadowJar -x test
echo "Build process finished."
echo "----------------------------------------"


echo "----------------------------------------"
echo "2. Finding the application JAR file..."
echo "----------------------------------------"
JAR_FILE=$(find "$API_BUILD_DIR" -name "$JAR_NAME_PATTERN" -print -quit) # -quit after first find

if [ -z "$JAR_FILE" ]; then
    echo ""
    echo "ERROR: Could not find the application JAR file matching '$JAR_NAME_PATTERN' in '$API_BUILD_DIR/'" >&2
    echo "Please ensure the project is built correctly using './gradlew build shadowJar'." >&2
    exit 1
fi

echo "Found JAR: $JAR_FILE"
echo "----------------------------------------"


echo "----------------------------------------"
echo "3. Starting the Spark ML Samples API application in the background..."
echo "----------------------------------------"
echo "The application will attempt to start on $FRONTEND_URL"
echo "Logs will be in api_app.log. Press Ctrl+C to stop the application and this script."
echo ""

# Run the JAR file using Java in the background and get its PID
java -jar "$JAR_FILE" > api_app.log 2>&1 &
APP_PID=$!
echo "Application started with PID: $APP_PID. Waiting for it to initialize..."

# Wait for a few seconds for the server to start
sleep "$SERVER_START_WAIT_TIME"

# Check if the process is still running
if ! ps -p "$APP_PID" > /dev/null; then
  echo "ERROR: Application failed to start. Check api_app.log for details."
  exit 1
fi
echo "Application likely initialized."
echo "----------------------------------------"

# Attempt to open the browser
open_browser "$FRONTEND_URL"
echo "----------------------------------------"

echo ""
echo "Application is running in the background (PID: $APP_PID)."
echo "View logs with: tail -f api_app.log"
echo "Press Ctrl+C to stop the application and this script."
echo ""

# Wait for the background process to finish (e.g., if it's stopped by Ctrl+C)
# This makes the script wait here, so Ctrl+C in the terminal will trigger the trap.
wait "$APP_PID"
EXIT_STATUS=$? # Capture exit status of the Java app

echo ""
echo "----------------------------------------"
echo "Application (PID: $APP_PID) has finished or been stopped with status $EXIT_STATUS."
echo "----------------------------------------"

exit "$EXIT_STATUS"