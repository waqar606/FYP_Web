function displayFileName(input) {
  var fileName = input.files[0].name;
  document.querySelector(".custom-file-label").textContent = fileName;

  var preview = document.getElementById("image-preview");
  var file = input.files[0];
  var reader = new FileReader();

  reader.onload = function () {
    preview.innerHTML =
      '<img src="' + reader.result + '" class="img-fluid" alt="Image Preview">';
  };

  if (file) {
    reader.readAsDataURL(file);
  } else {
    preview.innerHTML = "";
  }
}

function checkResult() {
  var fileName = document.querySelector(".custom-file-label").textContent;
  var errorMessage = document.getElementById("errorMessage");
  var resultMessage = document.getElementById("resultMessage");
  var resultTextArea = document.getElementById("resultTextArea");
  var loadingOverlay = document.getElementById("loadingOverlay");

  if (!fileName || fileName === "Select file") {
    errorMessage.textContent = "Please select a picture first.";
    resultMessage.textContent = "";
    resultTextArea.value = "";
    return;
  }

  errorMessage.textContent = ""; // Clear error message
  resultMessage.textContent = ""; // Clear result message

  loadingOverlay.style.display = "flex"; // Show loading overlay

  // Simulate delay with setTimeout (replace with actual processing logic)
  setTimeout(() => {
    // Randomly generate a result
    let randomNum = Math.random();
    let result;

    if (randomNum < 0.33) {
      result = "You have chances of AD. Please connect with a doctor.";
    } else if (randomNum < 0.66) {
      result = "Normal";
    } else {
      result = "MCI";
    }

    resultTextArea.value = result;
    loadingOverlay.style.display = "none"; // Hide loading overlay
  }, 3000); // Simulated delay of 3 seconds
}
