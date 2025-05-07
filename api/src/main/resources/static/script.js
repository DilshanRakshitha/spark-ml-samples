const lyricsInput = document.getElementById('lyricsInput');
const modelSelect = document.getElementById('modelSelect');
const predictButton = document.getElementById('predictButton');
const trainButton = document.getElementById('trainButton');
const predictionResultDiv = document.getElementById('predictionResult');
const trainingStatsDiv = document.getElementById('trainingStats');
const statusMessageDiv = document.getElementById('statusMessage');
const chartCanvas = document.getElementById('probabilityChart').getContext('2d');
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
        return;
    }

    setLoadingState(predictButton, true, "Predicting");
    predictionResultDiv.textContent = 'Predicting...';
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
            throw new Error(result.error || `HTTP error! status: ${response.status}`);
        }

        predictionResultDiv.textContent = `Predicted Genre: ${result.genre}`;
        updateChart(result.probabilities);
        setStatus("Prediction complete.");

    } catch (error) {
        console.error('Prediction Error:', error);
        predictionResultDiv.textContent = `Error: ${error.message}`;
        setStatus(`Prediction failed for ${modelName}.`);
    } finally {
        setLoadingState(predictButton, false, "Predict Genre");
    }
}

async function handleTraining() {
    const modelName = modelSelect.value;

    setLoadingState(trainButton, true, "Training");
    trainingStatsDiv.textContent = `Training model: ${modelName}...\nThis might take several minutes. Please wait.`;
    setStatus(`Training started for ${modelName}...`);

    try {
        const response = await fetch(`/lyrics/train?modelName=${encodeURIComponent(modelName)}`, {
            method: 'GET' // GET request for training trigger
        });

        const result = await response.json();

         if (!response.ok) {
            throw new Error(result.error || `HTTP error! status: ${response.status}`);
        }

        // Display formatted statistics
        let statsText = `Training complete for: ${result.trainedModel || modelName}\n\n`;
        for (const key in result) {
            if (key !== 'trainedModel') { // Don't repeat model name
               statsText += `${key}: ${result[key]}\n`;
            }
        }
        trainingStatsDiv.textContent = statsText;
        setStatus(`Training complete for ${modelName}. Accuracy: ${result['Best Cross-Validation Accuracy'] || 'N/A'}`);

    } catch (error) {
        console.error('Training Error:', error);
        trainingStatsDiv.textContent = `Error during training: ${error.message}`;
         setStatus(`Training failed for ${modelName}.`);
    } finally {
        setLoadingState(trainButton, false, "Train Selected Model");
    }
}

function updateChart(probabilities) {
    clearChart(); // Clear existing chart before drawing new one

    if (!probabilities || Object.keys(probabilities).length === 0) {
        console.log("No probability data to display.");
        return;
    }

    // Prepare data for Chart.js (horizontal bar chart)
    const labels = Object.keys(probabilities);
    const dataValues = Object.values(probabilities);

    // Define colors (optional, can use Chart.js defaults)
     const backgroundColors = [
        'rgba(255, 99, 132, 0.7)', // Red
        'rgba(54, 162, 235, 0.7)', // Blue
        'rgba(255, 206, 86, 0.7)', // Yellow
        'rgba(75, 192, 192, 0.7)', // Green
        'rgba(153, 102, 255, 0.7)',// Purple
        'rgba(255, 159, 64, 0.7)', // Orange
        'rgba(99, 255, 132, 0.7)', // Lime
        'rgba(201, 203, 207, 0.7)' // Grey
    ];
     const borderColors = backgroundColors.map(color => color.replace('0.7', '1'));


    probabilityChart = new Chart(chartCanvas, {
        type: 'bar', // Use 'bar' for horizontal bars with indexAxis: 'y'
        data: {
            labels: labels,
            datasets: [{
                label: 'Prediction Probability',
                data: dataValues,
                backgroundColor: backgroundColors.slice(0, labels.length),
                borderColor: borderColors.slice(0, labels.length),
                borderWidth: 1,
                indexAxis: 'y', // Makes the bars horizontal
            }]
        },
        options: {
            indexAxis: 'y', // Crucial for horizontal bars
            responsive: true,
            maintainAspectRatio: false, // Allows chart to fill container height
            plugins: {
                 legend: {
                    display: false // Hide legend if only one dataset
                },
                tooltip: {
                     callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.x !== null) {
                                // Format as percentage for horizontal bars (value is on x-axis)
                                label += (context.parsed.x * 100).toFixed(2) + '%';
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: { // Probabilities are on the x-axis now
                    beginAtZero: true,
                    max: 1, // Probabilities range from 0 to 1
                     ticks: {
                         // Format x-axis ticks as percentages
                         callback: function(value, index, values) {
                             return (value * 100) + '%';
                         }
                     }
                },
                 y: { // Genre labels are on the y-axis
                    beginAtZero: true
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
     // Optionally clear the canvas directly
    // const canvas = document.getElementById('probabilityChart');
    // const ctx = canvas.getContext('2d');
    // ctx.clearRect(0, 0, canvas.width, canvas.height);
}

 function setStatus(message) {
    statusMessageDiv.textContent = message;
}

function setLoadingState(button, isLoading, defaultText) {
     if (isLoading) {
        button.disabled = true;
        button.textContent = defaultText.replace('...', '') + '...'; // Add dots if not already there
        button.classList.add('loading');
    } else {
        button.disabled = false;
        button.textContent = defaultText;
        button.classList.remove('loading');
    }
}

// Initial clear state
setStatus("Ready.");
predictionResultDiv.textContent = "Prediction will appear here...";
trainingStatsDiv.textContent = "Training stats will appear here...";