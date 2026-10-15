document.addEventListener("DOMContentLoaded", () => {
    const audioInput = document.getElementById("audioFile");
    const browseButton = document.getElementById("browseButton");
    const uploadArea = document.getElementById("uploadArea");
    const fileDetails = document.getElementById("fileDetails");
    const fileName = document.getElementById("fileName");
    const fileSize = document.getElementById("fileSize");
    const audioPlayer = document.getElementById("audioPlayer");
    const removeFileButton = document.getElementById("removeFile");
    const analyzeButton = document.getElementById("analyzeButton");
    const buttonText = document.getElementById("buttonText");
    const statusMessage = document.getElementById("statusMessage");
    const results = document.getElementById("results");

    let selectedFile = null;
    let audioUrl = null;

    // Check that the HTML elements needed by this script exist.
    const requiredElements = {
        audioInput,
        browseButton,
        uploadArea,
        fileDetails,
        fileName,
        fileSize,
        audioPlayer,
        removeFileButton,
        analyzeButton,
        buttonText,
        statusMessage,
        results
    };

    for (const [name, element] of Object.entries(requiredElements)) {
        if (!element) {
            console.error(`Missing HTML element: ${name}`);
            return;
        }
    }

    function setStatus(message, isError = false) {
        statusMessage.textContent = message;
        statusMessage.style.color = isError
            ? "#fda4af"
            : "#a2a5bf";
    }

    function clearResults() {
        results.classList.add("hidden");

        document.getElementById("probabilityList").replaceChildren();
        document.getElementById("predictedEmotion").textContent = "—";
        document.getElementById("confidenceScore").textContent = "—";
        document.getElementById("durationValue").textContent = "—";
        document.getElementById("sampleRateValue").textContent = "—";
        document.getElementById("sampleCountValue").textContent = "—";
    }

    function selectAudio(file) {
        if (!file) {
            return;
        }

        if (!file.name.toLowerCase().endsWith(".wav")) {
            setStatus("Please select a WAV audio file.", true);
            return;
        }

        if (file.size === 0) {
            setStatus("The selected file is empty.", true);
            return;
        }

        selectedFile = file;

        if (audioUrl) {
            URL.revokeObjectURL(audioUrl);
        }

        audioUrl = URL.createObjectURL(file);

        fileName.textContent = file.name;
        fileSize.textContent =
            `${(file.size / (1024 * 1024)).toFixed(2)} MB`;

        audioPlayer.src = audioUrl;
        audioPlayer.classList.remove("hidden");

        fileDetails.classList.remove("hidden");
        uploadArea.classList.add("hidden");

        analyzeButton.disabled = false;
        buttonText.textContent = "Analyze emotion";

        clearResults();
        setStatus("Audio selected successfully. Click Analyze emotion.");

        console.log("Selected audio file:", file.name);
    }

    function resetAudio() {
        selectedFile = null;

        if (audioUrl) {
            URL.revokeObjectURL(audioUrl);
            audioUrl = null;
        }

        audioInput.value = "";

        audioPlayer.pause();
        audioPlayer.removeAttribute("src");
        audioPlayer.load();
        audioPlayer.classList.add("hidden");

        fileDetails.classList.add("hidden");
        uploadArea.classList.remove("hidden");

        analyzeButton.disabled = true;
        buttonText.textContent = "Analyze emotion";

        clearResults();
        setStatus("");
    }

    // Open the Windows file picker from a direct button click.
    browseButton.addEventListener("click", () => {
        audioInput.click();
    });

    // Detect the selected file.
    audioInput.addEventListener("change", () => {
        const file = audioInput.files[0];

        if (file) {
            selectAudio(file);
        }
    });

    // Remove the selected file.
    removeFileButton.addEventListener("click", resetAudio);

    // Support drag and drop.
    uploadArea.addEventListener("dragover", (event) => {
        event.preventDefault();
        uploadArea.classList.add("drag-over");
    });

    uploadArea.addEventListener("dragleave", () => {
        uploadArea.classList.remove("drag-over");
    });

    uploadArea.addEventListener("drop", (event) => {
        event.preventDefault();
        uploadArea.classList.remove("drag-over");

        const file = event.dataTransfer.files[0];

        if (file) {
            selectAudio(file);
        }
    });

    // Send the selected file to the Flask prediction endpoint.
    analyzeButton.addEventListener("click", async () => {
        if (!selectedFile) {
            setStatus("Please select a WAV file first.", true);
            return;
        }

        const formData = new FormData();
        formData.append("audio", selectedFile);

        analyzeButton.disabled = true;
        buttonText.textContent = "Analyzing...";
        setStatus("Analyzing audio. Please wait...");
        clearResults();

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(
                    data.error || "The server could not analyze the file."
                );
            }

            document.getElementById("predictedEmotion").textContent =
                String(data.emotion)
                    .replaceAll("_", " ")
                    .toUpperCase();

            document.getElementById("confidenceScore").textContent =
                `${Number(data.confidence).toFixed(2)}%`;

            document.getElementById("durationValue").textContent =
                `${Number(data.duration).toFixed(2)} seconds`;

            document.getElementById("sampleRateValue").textContent =
                `${Number(data.sample_rate).toLocaleString()} Hz`;

            document.getElementById("sampleCountValue").textContent =
                Number(data.samples).toLocaleString();

            const probabilityList =
                document.getElementById("probabilityList");

            probabilityList.replaceChildren();

            const sortedProbabilities = Object.entries(
                data.probabilities || {}
            ).sort((a, b) => Number(b[1]) - Number(a[1]));

            for (const [emotion, probability] of sortedProbabilities) {
                const row = document.createElement("div");
                row.className = "probability-row";

                const info = document.createElement("div");
                info.className = "probability-info";

                const name = document.createElement("span");
                name.textContent = emotion;

                const value = document.createElement("span");
                value.textContent =
                    `${Number(probability).toFixed(2)}%`;

                const track = document.createElement("div");
                track.className = "probability-track";

                const fill = document.createElement("div");
                fill.className = "probability-fill";

                const safeProbability = Math.max(
                    0,
                    Math.min(100, Number(probability) || 0)
                );

                fill.style.width = `${safeProbability}%`;

                info.append(name, value);
                track.append(fill);
                row.append(info, track);
                probabilityList.append(row);
            }

            results.classList.remove("hidden");
            setStatus("Analysis completed successfully!");

            results.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        } catch (error) {
            console.error("Audio analysis failed:", error);

            setStatus(
                error.message ||
                "Unable to analyze the audio. Check the Flask terminal.",
                true
            );

        } finally {
            analyzeButton.disabled = !selectedFile;
            buttonText.textContent = "Analyze emotion";
        }
    });

    // Initial state.
    analyzeButton.disabled = true;
    clearResults();

    console.log("EchoMind JavaScript loaded successfully.");
});
