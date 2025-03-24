<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Brain Tumor Detection</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            margin: 0;
            padding: 0;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            background: #f4f4f4;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.2);
            width: 80%;
            max-width: 500px;
        }
        input, select, button {
            width: 90%;
            padding: 10px;
            margin: 10px;
            font-size: 16px;
        }
        #imageUploadSection, #reportSection {
            display: none;
        }
        #imagePreview {
            margin-top: 20px;
            max-width: 100%;
            border-radius: 10px;
            display: none;
        }
        #report {
            text-align: left;
            padding: 10px;
            background: #eef;
            border-radius: 5px;
        }
    </style>
</head>
<body>

<div class="container">
    <!-- Patient Details Form -->
    <div id="patientDetails">
        <h2>Enter Patient Details</h2>
        <input type="text" id="patientName" placeholder="Patient Name" required>
        <input type="number" id="patientAge" placeholder="Age" required>
        <select id="patientGender">
            <option value="Male">Male</option>
            <option value="Female">Female</option>
        </select>
        <textarea id="clinicalHistory" placeholder="Clinical History"></textarea>
        <button onclick="showImageUpload()">Next</button>
    </div>

    <!-- Image Upload Section -->
    <div id="imageUploadSection">
        <h2>Upload MRI Scan</h2>
        <input type="file" id="fileInput" accept="image/*">
        <br>
        <img id="imagePreview" src="#" alt="Uploaded Image">
        <br>
        <button id="predictButton">Predict</button>
    </div>

    <!-- Report Section -->
    <div id="reportSection">
        <h2>MRI Scan Report</h2>
        <div id="report"></div>
    </div>
</div>

<script>
    function showImageUpload() {
        $("#patientDetails").hide();
        $("#imageUploadSection").show();
    }

    $("#fileInput").change(function(event){
        var reader = new FileReader();
        reader.onload = function(){
            var output = document.getElementById("imagePreview");
            output.src = reader.result;
            output.style.display = "block";
        }
        reader.readAsDataURL(event.target.files[0]);
    });

    $("#predictButton").click(function(){
        var file = $("#fileInput")[0].files[0];
        if (!file) {
            alert("Please select an image.");
            return;
        }

        var formData = new FormData();
        formData.append("file", file);
        formData.append("name", $("#patientName").val());
        formData.append("age", $("#patientAge").val());
        formData.append("gender", $("#patientGender").val());
        formData.append("history", $("#clinicalHistory").val());

        $.ajax({
            url: "/predict",
            type: "POST",
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                if (response.error) {
                    alert("Error: " + response.error);
                } else {
                    $("#imageUploadSection").hide();
                    $("#reportSection").show();
                    $("#report").html(`
                        <p><strong>Date of Report:</strong> ${new Date().toLocaleDateString()}</p>
                        <p><strong>Patient Details:</strong></p>
                        <p>Name: ${response.name}</p>
                        <p>Age/Gender: ${response.age}, ${response.gender}</p>
                        <p>Clinical History: ${response.history}</p>
                        <p><strong>Findings:</strong> ${response.prediction} (${(response.confidence * 100).toFixed(2)}%)</p>
                    `);
                }
            },
            error: function() {
                alert("Error in prediction. Ensure the server is running.");
            }
        });
    });
</script>

</body>
</html>
