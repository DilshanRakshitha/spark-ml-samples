const lyricsInput = document.getElementById('lyricsInput');
const modelSelect = document.getElementById('modelSelect');
const predictButton = document.getElementById('predictButton');
const trainButton = document.getElementById('trainButton');
const predictionResultDiv = document.getElementById('predictionResult');
const trainingStatsDiv = document.getElementById('trainingStats');
const statusMessageDiv = document.getElementById('statusMessage');
const chartCanvas = document.getElementById('probabilityChart');
const chartCtx = chartCanvas.getContext('2d');
let probabilityChart = null;

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
        clearChart();
        return;
    }

    setLoadingState(predictButton, true, "Predicting");
    predictionResultDiv.textContent = 'Predicting...';
    setStatus(`Predicting genre using ${modelName}...`);
    clearChart();

    try {
        const response = await fetch(`/lyrics/predict?modelName=${encodeURIComponent(modelName)}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'text/plain'
            },
            body: lyrics
        });

        const result = await response.json();

        if (!response.ok) {
             const errorMsg = result?.genre?.includes("Error:") ? result.genre : (result?.message || result?.error || `HTTP error! Status: ${response.status}`);
             throw new Error(errorMsg);
        }
        
        let predictionText = `Predicted Genre: ${result.genre || 'Unknown'}`;
        // Check if genre itself contains an error message from the backend
        if (result.genre && result.genre.toLowerCase().includes("error:")) {
            predictionText = result.genre; // Display the error message as the main text
        } else if (result.message) { // Display any additional message from backend
             predictionText += ` (${result.message})`;
        }
        predictionResultDiv.textContent = predictionText;

        updateChart(result.probabilities);
        setStatus("Prediction complete.");

    } catch (error) {
        console.error('Prediction Error:', error);
        predictionResultDiv.textContent = `Error: ${error.message || 'Prediction failed'}`;
        setStatus(`Prediction failed for ${modelName}.`);
        clearChart();
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
            method: 'GET'
        });

        const result = await response.json();

         if (!response.ok) {
             const errorMsg = result?.error || result?.message || `HTTP error! Status: ${response.status}`;
             throw new Error(errorMsg);
         }

        let statsText = `Training complete for: ${result.trainedModel || modelName}\n\nMetrics:\n`;
        let accuracy = 'N/A';
        for (const key in result) {
             let formattedKey = key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase());
             if (key !== 'trainedModel' && key !== 'Best Parameters Index (0-based)' && key !== 'Metric Name') {
                 if (formattedKey.includes('Accuracy') || formattedKey.includes('F1')) {
                    statsText += `  * ${formattedKey}: ${parseFloat(result[key]).toFixed(4)}\n`;
                    if (formattedKey.includes('Accuracy')) accuracy = parseFloat(result[key]).toFixed(4);
                 } else if (key === 'Average Metrics Across Folds' || key === 'Best Parameters'){
                     statsText += `\n  ${formattedKey}:\n    ${result[key]}\n`;
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
    clearChart(); 

    if (!probabilities || Object.keys(probabilities).length === 0 || probabilities.error) {
        console.log("No valid probability data to display.");
        chartCtx.clearRect(0, 0, chartCanvas.width, chartCanvas.height);
        chartCtx.font = "16px Arial";
        chartCtx.fillStyle = "#888";
        chartCtx.textAlign = "center";
        chartCtx.fillText(probabilities && probabilities.error ? "Prediction Error" : "No probability data available", chartCanvas.width / 2, chartCanvas.height / 2);
        return;
    }

    const labels = Object.keys(probabilities);
    const dataValues = Object.values(probabilities);

     const baseBackgroundColors = [
        'rgba(255, 99, 132, 0.7)',  // Red
        'rgba(54, 162, 235, 0.7)',  // Blue
        'rgba(255, 206, 86, 0.7)',  // Yellow
        'rgba(75, 192, 192, 0.7)',  // Green
        'rgba(153, 102, 255, 0.7)', // Purple
        'rgba(255, 159, 64, 0.7)',  // Orange
        'rgba(99, 255, 132, 0.7)',  // Lime
        'rgba(201, 203, 207, 0.7)', // Grey (for 8th class: Dance)
        'rgba(255, 138, 101, 0.7)', // Coral (extra)
        'rgba(3, 169, 244, 0.7)'    // Light Blue (extra)
    ];
    const backgroundColors = labels.map((_, i) => baseBackgroundColors[i % baseBackgroundColors.length]);
    const borderColors = backgroundColors.map(color => color.replace('0.7', '1'));

    const sortedData = labels.map((label, index) => ({ label, value: dataValues[index] }))
                          .sort((a, b) => b.value - a.value); 

    const sortedLabels = sortedData.map(item => item.label);
    const sortedValues = sortedData.map(item => item.value);
    const sortedBackgroundColors = sortedLabels.map(label => {
         const originalIndex = labels.indexOf(label);
         return baseBackgroundColors[originalIndex % baseBackgroundColors.length];
     });
     const sortedBorderColors = sortedBackgroundColors.map(color => color.replace('0.7', '1'));

    probabilityChart = new Chart(chartCtx, {
        type: 'bar', 
        data: {
            labels: sortedLabels, 
            datasets: [{
                label: 'Prediction Probability',
                data: sortedValues, 
                backgroundColor: sortedBackgroundColors, 
                borderColor: sortedBorderColors,
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'x', 
            responsive: true,
            maintainAspectRatio: false, 
            plugins: {
                 legend: {
                    display: false 
                },
                tooltip: {
                     callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) { label += ': '; }
                            if (context.parsed.y !== null) {
                                label += (context.parsed.y * 100).toFixed(2) + '%';
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                y: { 
                    beginAtZero: true,
                    max: 1, 
                     ticks: {
                         callback: function(value) {
                             return (value * 100) + '%';
                         }
                     }
                },
                 x: { 
                    beginAtZero: true,
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
    chartCtx.clearRect(0, 0, chartCanvas.width, chartCanvas.height);
}

 function setStatus(message) {
    statusMessageDiv.textContent = message;
}

function setLoadingState(button, isLoading, defaultText) {
     if (isLoading) {
        button.disabled = true;
        button.innerHTML = `${defaultText.replace('...', '')} <span class="loading-dots"></span>`;
        button.classList.add('loading');
    } else {
        button.disabled = false;
        button.textContent = defaultText;
        button.classList.remove('loading');
    }
}

setStatus("Ready.");
predictionResultDiv.textContent = "Prediction will appear here...";
trainingStatsDiv.textContent = "Training stats will appear here...";
clearChart(); // Initial clear of chart with message
updateChart(null); // Call to display "No probability data" initially