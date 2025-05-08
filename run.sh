#!/usr/bin/env bash

# Purpose: Build and run the Spark ML Samples API application.
# Place this script in the root directory: dilshanrakshitha-spark-ml-samples/
# Usage: ./run.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
EXPECTED_PORT=9090
# Adjust the pattern if your shadow JAR naming convention is different
JAR_NAME_PATTERN="api-*-SNAPSHOT.jar"
API_BUILD_DIR="api/build/libs"

# --- Functions ---
cleanup() {
  echo "Script interrupted. Stopping application (if running)..."
  # Find and kill the Java process - might need refinement depending on the system
  # This is a basic attempt; a more robust solution might use PID files.
  pkill -f "$JAR_NAME_PATTERN" || echo "Application process not found or already stopped."
  exit 1
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
# Find the shadow JAR in the api module's build output
# Using the pattern defined above. Takes the first match if multiple exist.
JAR_FILE=$(find "$API_BUILD_DIR" -name "$JAR_NAME_PATTERN" | head -n 1)

if [ -z "$JAR_FILE" ]; then
    echo ""
    echo "ERROR: Could not find the application JAR file matching '$JAR_NAME_PATTERN' in '$API_BUILD_DIR/'" >&2
    echo "Please ensure the project is built correctly using './gradlew build shadowJar'." >&2
    exit 1
fi

echo "Found JAR: $JAR_FILE"
echo "----------------------------------------"


echo "----------------------------------------"
echo "3. Starting the Spark ML Samples API application..."
echo "----------------------------------------"
echo "The application will attempt to start on http://localhost:${EXPECTED_PORT}"
echo "Press Ctrl+C to stop the application."
echo ""

# Run the JAR file using Java
# The application.properties inside the JAR should configure it for the expected port
java -jar "$JAR_FILE"

# Note: The script will stay on the 'java -jar' command until the application
# is stopped (e.g., with Ctrl+C, which is handled by the trap).

echo ""
echo "----------------------------------------"
echo "Application finished or stopped."
echo "----------------------------------------"

exit 0