function onClickedPredictPrice() {
    console.log("Predict price button clicked");
    
    try {
        // Create FormData object
        var formData = new FormData();
        
        // Add numerical inputs with validation
        const numericalFields = [
            'symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight',
            'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio',
            'horsepower', 'peakrpm', 'citympg', 'highwaympg'
        ];
        
        for (const field of numericalFields) {
            const value = document.getElementById('ui' + field).value;
            if (!value) {
                alert(`Vui lòng nhập ${field}`);
                return;
            }
            formData.append(field, parseFloat(value));
        }
        
        // Add categorical inputs with validation
        const categoricalFields = [
            'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
            'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem',
            'brand', 'model'
        ];
        
        for (const field of categoricalFields) {
            const value = document.getElementById('ui' + field).value;
            if (!value) {
                alert(`Vui lòng chọn ${field}`);
                return;
            }
            formData.append(field, value);
        }

        // Show loading state
        const resultDiv = document.getElementById("uiPredictPrice");
        resultDiv.innerHTML = "<h2>Đang tính toán...</h2>";
        
        // Make API call
        $.ajax({
            url: "http://127.0.0.1:5000/predict_price_car",
            type: "POST",
            data: formData,
            processData: false,
            contentType: false,
            success: function(data, status) {
                console.log("Success:", data);
                if (data && data.predict_price) {
                    const price = new Intl.NumberFormat('vi-VN', {
                        style: 'currency',
                        currency: 'USD'
                    }).format(data.predict_price);
                    resultDiv.innerHTML = `<h2>Giá dự đoán: ${price}</h2>`;
                } else {
                    resultDiv.innerHTML = "<h2>Không thể dự đoán giá. Vui lòng thử lại!</h2>";
                }
            },
            error: function(xhr, status, error) {
                console.error("Error:", error);
                resultDiv.innerHTML = "<h2>Có lỗi xảy ra. Vui lòng thử lại!</h2>";
            }
        });
    } catch (error) {
        console.error("Error:", error);
        document.getElementById("uiPredictPrice").innerHTML = 
            "<h2>Có lỗi xảy ra. Vui lòng thử lại!</h2>";
    }
}

function onPageLoad() {
    console.log("Document loaded");
    
    // Get column values from backend
    $.ajax({
        url: "http://127.0.0.1:5000/get_full_columns",
        type: "GET",
        success: function(data, status) {
            console.log("Got response for get_full_columns request");
            if(data) {
                // Map of field IDs to their values
                const fieldMap = {
                    'fueltype': data.fueltype,
                    'aspiration': data.aspiration,
                    'doornumber': data.doornumber,
                    'carbody': data.carbody,
                    'drivewheel': data.drivewheel,
                    'enginelocation': data.enginelocation,
                    'enginetype': data.enginetype,
                    'cylindernumber': data.cylindernumber,
                    'fuelsystem': data.fuelsystem,
                    'brand': data.brand,
                    'model': data.model
                };

                // Populate dropdowns
                for (const [field, values] of Object.entries(fieldMap)) {
                    const select = document.getElementById('ui' + field);
                    if (select && values) {
                        select.innerHTML = ''; // Clear existing options
                        values.forEach(value => {
                            const option = new Option(value);
                            select.add(option);
                        });
                    }
                }
            }
        },
        error: function(xhr, status, error) {
            console.error("Error loading options:", error);
            alert("Không thể tải dữ liệu. Vui lòng tải lại trang!");
        }
    });
}

// Attach event listener for page load
window.onload = onPageLoad;
