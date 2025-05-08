const lyricsInput = document.getElementById('lyricsInput');
const modelSelect = document.getElementById('modelSelect');
const predictButton = document.getElementById('predictButton');
const trainButton = document.getElementById('trainButton');
const predictionResultDiv = document.getElementById('predictionResult');
const trainingStatsDiv = document.getElementById('trainingStats');
const statusMessageDiv = document.getElementById('statusMessage');
const chartCanvas = document.getElementById('probabilityChart'); // Get canvas element
const chartCtx = chartCanvas.getContext('2d'); // Get context
let probabilityChart = null; // To hold the chart instance

// --- Event Listeners ---
predictButton.addEventListener('click', handlePrediction);
trainButton.addEventListener('click', handleTraining);

// --- Functions ---

async function handlePrediction() {
    const lyrics = lyricsInput.value.trim();
    const modelName = modelSelect.value;

    if (!lyrics) {
        setStatus("Please enter some lyrics.");
        predictionResultDiv.textContent = 'Please enter lyrics first.';
        clearChart(); // Clear any old chart
        return;
    }

    setLoadingState(predictButton, true, "Predicting");
    predictionResultDiv.textContent = 'Predicting...';
    setStatus(`Predicting genre using ${modelName}...`);
    clearChart(); // Clear previous chart

    try {
        const response = await fetch(`/lyrics/predict?modelName=${encodeURIComponent(modelName)}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'text/plain' // Send as plain text
            },
            body: lyrics
        });

        const result = await response.json();

        if (!response.ok) {
             // Try to get error message from backend response, otherwise use status text
             const errorMsg = result?.message || result?.error || `HTTP error! Status: ${response.status}`;
             throw new Error(errorMsg);
        }

        predictionResultDiv.textContent = `Predicted Genre: ${result.genre || 'Unknown'}`;
        if (result.message) { // Display any additional message from backend
             predictionResultDiv.textContent += ` (${result.message})`;
        }
        updateChart(result.probabilities); // Update chart with probabilities
        setStatus("Prediction complete.");

    } catch (error) {
        console.error('Prediction Error:', error);
        predictionResultDiv.textContent = `Error: ${error.message || 'Prediction failed'}`;
        setStatus(`Prediction failed for ${modelName}.`);
        clearChart(); // Ensure chart is clear on error
    } finally {
        setLoadingState(predictButton, false, "Predict Genre");
    }
}

async function handleTraining() {
    const modelName = modelSelect.value;

    setLoadingState(trainButton, true, "Training");
    trainingStatsDiv.textContent = `Training model: ${modelName}...\nThis might take several minutes. Please wait. Do not close this page.`;
    setStatus(`Training started for ${modelName}...`);

    try {
        const response = await fetch(`/lyrics/train?modelName=${encodeURIComponent(modelName)}`, {
            method: 'GET' // GET request for training trigger
        });

        const result = await response.json();

         if (!response.ok) {
             const errorMsg = result?.error || result?.message || `HTTP error! Status: ${response.status}`;
             throw new Error(errorMsg);
         }

        // Display formatted statistics
        let statsText = `Training complete for: ${result.trainedModel || modelName}\n\nMetrics:\n`;
        let accuracy = 'N/A';
        for (const key in result) {
             // Format key for better readability
             let formattedKey = key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase());
             if (key !== 'trainedModel' && key !== 'Best Parameters Index (0-based)' && key !== 'Metric Name') { // Exclude some verbose/internal keys
                 // Highlight the main metric (assumed to be accuracy or f1)
                 if (formattedKey.includes('Accuracy') || formattedKey.includes('F1')) {
                    statsText += `  * ${formattedKey}: ${parseFloat(result[key]).toFixed(4)}\n`; // Format numbers
                    if (formattedKey.includes('Accuracy')) accuracy = parseFloat(result[key]).toFixed(4);
                 } else if (key === 'Average Metrics Across Folds' || key === 'Best Parameters'){
                     statsText += `\n  ${formattedKey}:\n    ${result[key]}\n`; // Add indentation for complex values
                 }
                 else {
                     statsText += `  ${formattedKey}: ${result[key]}\n`;
                 }
             }
        }
        trainingStatsDiv.textContent = statsText;
        setStatus(`Training complete for ${modelName}. Accuracy: ${accuracy}`);

    } catch (error) {
        console.error('Training Error:', error);
        trainingStatsDiv.textContent = `Error during training: ${error.message || 'Training failed'}`;
        setStatus(`Training failed for ${modelName}.`);
    } finally {
        setLoadingState(trainButton, false, "Train Selected Model");
    }
}

function updateChart(probabilities) {
    clearChart(); // Clear existing chart before drawing new one

    if (!probabilities || Object.keys(probabilities).length === 0) {
        console.log("No probability data to display.");
        // Display a message on the canvas
        chartCtx.clearRect(0, 0, chartCanvas.width, chartCanvas.height); // Clear canvas first
        chartCtx.font = "16px Arial";
        chartCtx.fillStyle = "#888";
        chartCtx.textAlign = "center";
        chartCtx.fillText("No probability data available", chartCanvas.width / 2, chartCanvas.height / 2);
        return;
    }

    // Prepare data for Chart.js
    const labels = Object.keys(probabilities);
    const dataValues = Object.values(probabilities);

    // Define base colors (ensure enough for 8 classes)
     const baseBackgroundColors = [
        'rgba(255, 99, 132, 0.7)',  // Red
        'rgba(54, 162, 235, 0.7)',  // Blue
        'rgba(255, 206, 86, 0.7)',  // Yellow
        'rgba(75, 192, 192, 0.7)',  // Green
        'rgba(153, 102, 255, 0.7)', // Purple
        'rgba(255, 159, 64, 0.7)',  // Orange
        'rgba(99, 255, 132, 0.7)',  // Lime
        'rgba(201, 203, 207, 0.7)', // Grey
        'rgba(255, 138, 101, 0.7)', // Coral
        'rgba(3, 169, 244, 0.7)'    // Light Blue
    ];
    // Use modulo operator to cycle through colors if more labels than colors
    const backgroundColors = labels.map((_, i) => baseBackgroundColors[i % baseBackgroundColors.length]);
    const borderColors = backgroundColors.map(color => color.replace('0.7', '1'));

    // Optional: Sort data for better visualization (e.g., highest probability first)
    const sortedData = labels.map((label, index) => ({ label, value: dataValues[index] }))
                          .sort((a, b) => b.value - a.value); // Sort descending by probability

    const sortedLabels = sortedData.map(item => item.label);
    const sortedValues = sortedData.map(item => item.value);
    // Apply colors based on the original label index to keep colors consistent if sorting
    const sortedBackgroundColors = sortedLabels.map(label => {
         const originalIndex = labels.indexOf(label);
         return baseBackgroundColors[originalIndex % baseBackgroundColors.length];
     });
     const sortedBorderColors = sortedBackgroundColors.map(color => color.replace('0.7', '1'));

    probabilityChart = new Chart(chartCtx, { // Use chartCtx
        type: 'bar', // Vertical bar chart
        data: {
            labels: sortedLabels, // Use sorted labels
            datasets: [{
                label: 'Prediction Probability',
                data: sortedValues, // Use sorted values
                backgroundColor: sortedBackgroundColors, // Use sorted colors
                borderColor: sortedBorderColors,
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'x', // Makes bars vertical (default for 'bar')
            responsive: true,
            maintainAspectRatio: false, // Important for resizing within container
            plugins: {
                 legend: {
                    display: false // Hide legend as label explains it
                },
                tooltip: {
                     callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) { label += ': '; }
                            // For vertical bars, the value is on the y-axis
                            if (context.parsed.y !== null) {
                                label += (context.parsed.y * 100).toFixed(2) + '%';
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                y: { // Probabilities are on the y-axis
                    beginAtZero: true,
                    max: 1, // Probabilities range from 0 to 1
                     ticks: {
                         // Format y-axis ticks as percentages
                         callback: function(value) {
                             return (value * 100) + '%';
                         }
                     }
                },
                 x: { // Genre labels are on the x-axis
                    beginAtZero: true,
                    // Optional: Rotate labels if they become too crowded
                    // ticks: { autoSkip: false, maxRotation: 90, minRotation: 45 }
                 }
            }
        }
    });
}

function clearChart() {
    if (probabilityChart) {
        probabilityChart.destroy();
        probabilityChart = null;
    }
    // Also clear the canvas directly to remove any text messages
    chartCtx.clearRect(0, 0, chartCanvas.width, chartCanvas.height);
}

 function setStatus(message) {
    statusMessageDiv.textContent = message;
    // Auto-clear status after a delay? Optional.
    // setTimeout(() => {
    //     if (statusMessageDiv.textContent === message) { // Clear only if it hasn't changed
    //          statusMessageDiv.textContent = '';
    //     }
    // }, 5000); // Clear after 5 seconds
}

function setLoadingState(button, isLoading, defaultText) {
     if (isLoading) {
        button.disabled = true;
        // Add spinning dots or similar visual cue
        button.innerHTML = `${defaultText.replace('...', '')} <span class="loading-dots"></span>`;
        button.classList.add('loading');
    } else {
        button.disabled = false;
        button.textContent = defaultText; // Restore original text
        button.classList.remove('loading');
    }
}

// Initial clear state
setStatus("Ready.");
predictionResultDiv.textContent = "Prediction will appear here...";
trainingStatsDiv.textContent = "Training stats will appear here...";