// Fetch data from the JSON file
fetch('data.json')
  .then(response => response.json())
  .then(data => {
    // Update the heart rate value
    document.getElementById('heart-rate-value').textContent = data.heartRate;

    // Update the body temperature value
    document.getElementById('body-temp-value').textContent = data.bodyTemp;

    // Update the fall detection status
    const fallDetectionStatus = data.fallDetected ? 'Yes' : 'No';
    document.getElementById('fall-detection-value').textContent = fallDetectionStatus;

    // Optional: Add additional animation if fall is detected
    if (data.fallDetected) {
      document.getElementById('fall-detection').style.animation = "shake 0.5s";
    }
  })
  .catch(error => console.error('Error fetching data:', error));

// Optional: Shake animation for fall detection
const shakeAnimation = `
  @keyframes shake {
    0%, 100% { transform: translateX(0); }
    25%, 75% { transform: translateX(-5px); }
    50% { transform: translateX(5px); }
  }
`;

const styleSheet = document.createElement("style");
styleSheet.type = "text/css";
styleSheet.innerText = shakeAnimation;
document.head.appendChild(styleSheet);
